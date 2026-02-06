"""
Agent Factory System
====================

The factory that lets agents create agents.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  Agent Factory                                           │
    │  ┌────────────────┐  ┌────────────────┐                  │
    │  │ Prompt Registry │  │  Tool Registry  │                 │
    │  │  (what to be)   │  │  (what to use)  │                │
    │  └───────┬────────┘  └───────┬─────────┘                 │
    │          │                   │                            │
    │          ▼                   ▼                            │
    │  ┌─────────────────────────────────────┐                 │
    │  │         Prompt Composer              │                 │
    │  │  Identity + Tools + Reasoning +      │                 │
    │  │  Contract + Guardrails               │                 │
    │  └───────────────┬─────────────────────┘                 │
    │                  │                                        │
    │                  ▼                                        │
    │  ┌─────────────────────────────────────┐                 │
    │  │     deepagents runtime              │                 │
    │  │  create_deep_agent(                 │                 │
    │  │    system_prompt=composed,           │                 │
    │  │    tools=resolved_tools,            │                 │
    │  │    subagents=sub_agents             │                 │
    │  │  )                                  │                 │
    │  └─────────────────────────────────────┘                 │
    └──────────────────────────────────────────────────────────┘

Usage:
    from agent_factory import AgentFactory

    factory = AgentFactory()
    agent = factory.create("financial-analyst-v2", tool_overrides=["web_search"])
    result = agent.invoke({"messages": [...]})
"""

from __future__ import annotations

import os
import re
import yaml
import json
import logging
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from enum import Enum

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# ============================================================
# RUNTIME DETECTION
# ============================================================
# Try deepagents first (full features: planning, filesystem, sub-agents)
# Fall back to LangGraph's create_react_agent (basic ReAct loop)

try:
    from deepagents import create_deep_agent
    AGENT_RUNTIME = "deepagents"
    logger.info("Agent runtime: deepagents (full features)")
except ImportError:
    create_deep_agent = None
    AGENT_RUNTIME = "langgraph"
    logger.info("Agent runtime: langgraph (basic ReAct fallback)")


# ============================================================
# 1. DATA MODELS
# ============================================================

class PromptType(str, Enum):
    PERSONA = "persona"
    TASK = "task"
    COMPOSITE = "composite"
    TOOL_WRAPPER = "tool-wrapper"


class ReasoningStyle(str, Enum):
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    ADVERSARIAL = "adversarial"
    METHODICAL = "methodical"
    EXPLORATORY = "exploratory"
    CONVERSATIONAL = "conversational"
    SIMULATIVE = "simulative"


class GraphType(str, Enum):
    REACT = "react"
    CHAIN = "chain"
    PLAN_EXECUTE = "plan-execute"


class Complexity(str, Enum):
    ATOMIC = "atomic"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ReasoningFramework(str, Enum):
    """Meta-reasoning layer injected on top of domain reasoning."""
    NONE = "none"
    SIRP = "sirp"           # Structured Iterative Reasoning Protocol
    STANDARD = "standard"   # Basic self-reflection


@dataclass
class PromptTemplate:
    """Complete specification for an agent persona/task."""
    # Identity
    id: str
    name: str
    version: str
    description: str

    # The prompt
    system_prompt: str
    prompt_type: PromptType

    # Classification
    domain_tags: list[str]
    reasoning_style: ReasoningStyle
    complexity: Complexity
    composable: bool

    # Tool dependencies
    required_tools: list[str]
    optional_tools: list[str]

    # Execution
    recommended_graph: GraphType
    max_iterations: int | None = 15

    # Contract
    input_schema: dict | None = None
    output_schema: dict | None = None

    # Quality
    quality_score: float | None = None
    tested: bool = False
    author: str = "unknown"
    source: str = "curated"

    # Reasoning layer
    reasoning_framework: ReasoningFramework = ReasoningFramework.NONE

    # Relationships
    parent_template: str | None = None
    compatible_with: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)


@dataclass
class ToolRegistryEntry:
    """Specification for a tool/MCP server in the registry."""
    id: str
    name: str
    description: str
    tool_type: str  # "langchain_tool" | "mcp_server" | "function"

    # What gets injected into the system prompt
    prompt_instructions: str

    # Connection info
    config: dict = field(default_factory=dict)

    # The actual tool object (populated at runtime)
    tool_instance: BaseTool | None = None

    # Metadata
    rate_limit: str | None = None
    required_env_vars: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)


@dataclass
class SpawnConfig:
    """Configuration for a factory spawn request."""
    template_id: str
    tool_overrides: list[str] | None = None     # Override template's tool list
    model: str | None = None                     # Override default model
    max_iterations: int | None = None            # Override template's max
    reasoning_framework: ReasoningFramework | None = None  # Override template's framework
    budget: dict | None = None                   # Resource limits
    parent_agent_id: str | None = None           # Who spawned this
    task_context: str | None = None              # Additional context from orchestrator


@dataclass
class SpawnResult:
    """What the factory returns after creating an agent."""
    agent: Any              # The compiled LangGraph agent
    agent_id: str           # Unique ID for tracking
    template_id: str        # Which template was used
    tools_attached: list[str]  # Which tools were resolved
    tools_missing: list[str]   # Which required tools were unavailable
    composed_prompt: str    # The final system prompt (for debugging)
    genealogy: dict         # Spawn chain tracking


# ============================================================
# 2. PROMPT REGISTRY (Tool 1)
# ============================================================

class PromptRegistry:
    """
    Searchable library of prompt templates.

    Think of this as a parts catalog — the factory looks up
    what kind of agent to build based on task requirements.
    """

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}

    def load_from_yaml(self, path: str | Path) -> int:
        """Load templates from a YAML file (the enriched registry)."""
        path = Path(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        count = 0
        for entry in data:
            template = PromptTemplate(
                id=entry['id'],
                name=entry['name'],
                version=entry.get('version', '1.0.0'),
                description=entry.get('description', ''),
                system_prompt=entry['system_prompt'],
                prompt_type=PromptType(entry.get('prompt_type', 'persona')),
                domain_tags=entry.get('domain_tags', []),
                reasoning_style=ReasoningStyle(entry.get('reasoning_style', 'analytical')),
                complexity=Complexity(entry.get('complexity', 'moderate')),
                composable=entry.get('composable', True),
                required_tools=entry.get('required_tools', []),
                optional_tools=entry.get('optional_tools', []),
                recommended_graph=GraphType(entry.get('recommended_graph', 'react')),
                max_iterations=entry.get('max_iterations', 15),
                input_schema=entry.get('input_schema'),
                output_schema=entry.get('output_schema'),
                quality_score=entry.get('quality_score'),
                tested=entry.get('tested', False),
                author=entry.get('author', 'unknown'),
                source=entry.get('source', 'hf-import'),
                reasoning_framework=ReasoningFramework(entry.get('reasoning_framework', 'none')),
            )
            self._templates[template.id] = template
            count += 1

        logger.info(f"Loaded {count} templates from {path}")
        return count

    def load_from_dict(self, templates: dict[str, dict]) -> int:
        """Load templates from a Python dict (like expanded_system_prompts.py)."""
        count = 0
        for tid, entry in templates.items():
            template = PromptTemplate(
                id=entry['id'],
                name=entry['name'],
                version=entry.get('version', '1.0.0'),
                description=entry.get('description', ''),
                system_prompt=entry['system_prompt'],
                prompt_type=PromptType(entry.get('prompt_type', 'persona')),
                domain_tags=entry.get('domain_tags', []),
                reasoning_style=ReasoningStyle(entry.get('reasoning_style', 'analytical')),
                complexity=Complexity(entry.get('complexity', 'moderate')),
                composable=entry.get('composable', True),
                required_tools=entry.get('required_tools', []),
                optional_tools=entry.get('optional_tools', []),
                recommended_graph=GraphType(entry.get('recommended_graph', 'react')),
                max_iterations=entry.get('max_iterations', 15),
                input_schema=entry.get('input_schema'),
                output_schema=entry.get('output_schema'),
                quality_score=entry.get('quality_score'),
                tested=entry.get('tested', False),
                author=entry.get('author', 'unknown'),
                source=entry.get('source', 'curated'),
                reasoning_framework=ReasoningFramework(entry.get('reasoning_framework', 'none')),
            )
            self._templates[template.id] = template
            count += 1

        logger.info(f"Loaded {count} templates from dict")
        return count

    def get(self, template_id: str) -> PromptTemplate | None:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def search(
        self,
        query: str | None = None,
        domain_tags: list[str] | None = None,
        reasoning_style: ReasoningStyle | None = None,
        prompt_type: PromptType | None = None,
        requires_tools: list[str] | None = None,
        composable_only: bool = False,
        min_quality: float = 0.0,
        max_complexity: Complexity | None = None,
    ) -> list[PromptTemplate]:
        """
        Search the registry with filters.

        This is what an orchestrator agent calls when it needs
        to find the right template for a task it wants to delegate.
        """
        results = list(self._templates.values())

        # Filter by domain tags (any match)
        if domain_tags:
            results = [t for t in results
                       if any(d in t.domain_tags for d in domain_tags)]

        # Filter by reasoning style
        if reasoning_style:
            results = [t for t in results
                       if t.reasoning_style == reasoning_style]

        # Filter by prompt type
        if prompt_type:
            results = [t for t in results
                       if t.prompt_type == prompt_type]

        # Filter by required tools (template must need all specified tools)
        if requires_tools:
            results = [t for t in results
                       if all(tool in t.required_tools + t.optional_tools
                              for tool in requires_tools)]

        # Composable only
        if composable_only:
            results = [t for t in results if t.composable]

        # Quality floor
        if min_quality > 0:
            results = [t for t in results
                       if t.quality_score is not None and t.quality_score >= min_quality]

        # Complexity ceiling
        if max_complexity:
            complexity_order = {Complexity.ATOMIC: 0, Complexity.MODERATE: 1, Complexity.COMPLEX: 2}
            max_level = complexity_order[max_complexity]
            results = [t for t in results
                       if complexity_order.get(t.complexity, 0) <= max_level]

        # Keyword search in name + description + domain tags
        if query:
            query_lower = query.lower()
            scored = []
            for t in results:
                searchable = f"{t.name} {t.description} {' '.join(t.domain_tags)}".lower()
                # Simple relevance: count keyword matches
                score = sum(1 for word in query_lower.split() if word in searchable)
                if score > 0:
                    scored.append((score, t))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [t for _, t in scored]

        return results

    def list_all(self) -> list[str]:
        """List all template IDs."""
        return list(self._templates.keys())

    @property
    def count(self) -> int:
        return len(self._templates)


# ============================================================
# 3. TOOL REGISTRY (Tool 2)
# ============================================================

class ToolRegistry:
    """
    Catalog of available tools and MCP servers.

    Each entry knows:
    - What the tool does (for the factory to match against templates)
    - How to describe it to the agent (prompt_instructions)
    - How to connect to it (config)
    - The actual callable (tool_instance, populated at runtime)
    """

    def __init__(self):
        self._tools: dict[str, ToolRegistryEntry] = {}

    def register(self, entry: ToolRegistryEntry) -> None:
        """Register a tool in the catalog."""
        self._tools[entry.id] = entry
        logger.info(f"Registered tool: {entry.id} ({entry.tool_type})")

    def register_langchain_tool(
        self,
        tool_id: str,
        tool: BaseTool,
        prompt_instructions: str | None = None,
        domain_tags: list[str] | None = None,
    ) -> None:
        """Convenience method: register an existing LangChain tool."""
        entry = ToolRegistryEntry(
            id=tool_id,
            name=tool.name,
            description=tool.description,
            tool_type="langchain_tool",
            prompt_instructions=prompt_instructions or self._auto_instructions(tool),
            tool_instance=tool,
            domain_tags=domain_tags or [],
        )
        self.register(entry)

    def register_function(
        self,
        tool_id: str,
        func: callable,
        name: str,
        description: str,
        prompt_instructions: str,
        domain_tags: list[str] | None = None,
    ) -> None:
        """Register a plain Python function as a tool."""
        # Wrap as LangChain StructuredTool
        tool = StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
        )
        entry = ToolRegistryEntry(
            id=tool_id,
            name=name,
            description=description,
            tool_type="function",
            prompt_instructions=prompt_instructions,
            tool_instance=tool,
            domain_tags=domain_tags or [],
        )
        self.register(entry)

    def get(self, tool_id: str) -> ToolRegistryEntry | None:
        """Look up a tool by ID."""
        return self._tools.get(tool_id)

    def resolve(self, tool_ids: list[str]) -> tuple[list[BaseTool], list[str], list[str]]:
        """
        Resolve a list of tool IDs into actual tool instances.

        Returns:
            (resolved_tools, resolved_ids, missing_ids)
        """
        resolved = []
        resolved_ids = []
        missing = []

        for tid in tool_ids:
            entry = self._tools.get(tid)
            if entry and entry.tool_instance:
                resolved.append(entry.tool_instance)
                resolved_ids.append(tid)
            else:
                missing.append(tid)

        return resolved, resolved_ids, missing

    def get_prompt_instructions(self, tool_id: str) -> str | None:
        """Get the system prompt instructions for a tool."""
        entry = self._tools.get(tool_id)
        return entry.prompt_instructions if entry else None

    def search(
        self,
        query: str | None = None,
        domain_tags: list[str] | None = None,
    ) -> list[ToolRegistryEntry]:
        """Search available tools by keyword or domain."""
        results = list(self._tools.values())

        if domain_tags:
            results = [t for t in results
                       if any(d in t.domain_tags for d in domain_tags)]

        if query:
            query_lower = query.lower()
            results = [t for t in results
                       if query_lower in t.name.lower()
                       or query_lower in t.description.lower()]

        return results

    def list_all(self) -> list[str]:
        """List all registered tool IDs."""
        return list(self._tools.keys())

    @staticmethod
    def _auto_instructions(tool: BaseTool) -> str:
        """Generate basic prompt instructions from a LangChain tool."""
        schema = tool.args_schema.schema() if tool.args_schema else {}
        params = schema.get('properties', {})

        lines = [f"## Tool: {tool.name}", f"{tool.description}", ""]
        if params:
            lines.append("Parameters:")
            for pname, pinfo in params.items():
                ptype = pinfo.get('type', 'any')
                pdesc = pinfo.get('description', '')
                lines.append(f"  - {pname} ({ptype}): {pdesc}")

        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._tools)


# ============================================================
# 4. PROMPT COMPOSER
# ============================================================

# SIRP meta-reasoning injection
SIRP_INJECTION = """
# Meta-Reasoning Layer: Structured Iterative Reasoning Protocol (SIRP)

In addition to your domain-specific reasoning protocol, apply this
self-monitoring discipline throughout your work:

Enclose exploratory thoughts in <thinking> tags before committing to an approach.
Track your step budget with <count> tags (starting budget: {step_budget}).
After every 3-4 steps, pause for <reflection>:
  - Score your progress 0.0-1.0
  - If quality > 0.8: continue current approach
  - If quality 0.5-0.8: adjust strategy, consider alternative angles
  - If quality < 0.5: backtrack and try a fundamentally different approach

When you reach your final answer, wrap a <reward> tag with your
self-assessed quality score for the overall solution.

Important: SIRP tags are for your internal reasoning. Your final
output should still match the Output Contract specified above.
"""

STANDARD_REFLECTION_INJECTION = """
# Self-Reflection

After completing your analysis, briefly assess:
- Did I address the core question?
- What is my confidence in this answer?
- What would I do differently with more time/data?
"""


class PromptComposer:
    """
    Composes a complete system prompt from template + tool instructions.

    This is the assembly line that takes a blueprint (template) and
    parts (tool instructions) and produces a complete agent specification.

    Composition layers:
    1. Identity + Mandate (from template)
    2. Tool Instructions (injected from tool registry)
    3. Reasoning Protocol (from template)
    4. Input/Output Contract (from template)
    5. Guardrails (from template)
    6. Meta-reasoning (SIRP or standard, injected if configured)
    7. Task Context (from orchestrator, if provided)
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def compose(
        self,
        template: PromptTemplate,
        config: SpawnConfig,
    ) -> str:
        """
        Compose the final system prompt.

        Replaces {{TOOL_BLOCK:tool_id}} placeholders with actual
        tool instructions, injects meta-reasoning if configured,
        and appends task context if provided.
        """
        prompt = template.system_prompt

        # ── Step 1: Resolve tool blocks ──────────────────────
        prompt = self._inject_tool_blocks(prompt, template, config)

        # ── Step 2: Inject meta-reasoning framework ──────────
        framework = config.reasoning_framework or template.reasoning_framework
        if framework == ReasoningFramework.SIRP:
            step_budget = config.max_iterations or template.max_iterations or 20
            prompt += SIRP_INJECTION.format(step_budget=step_budget)
        elif framework == ReasoningFramework.STANDARD:
            prompt += STANDARD_REFLECTION_INJECTION

        # ── Step 3: Inject task context from orchestrator ────
        if config.task_context:
            prompt += f"\n\n# Task Context (from orchestrator)\n\n{config.task_context}\n"

        # ── Step 4: Inject budget constraints ────────────────
        if config.budget:
            budget_lines = ["\n# Resource Budget"]
            if 'max_sub_agents' in config.budget:
                budget_lines.append(f"- Maximum sub-agents you may spawn: {config.budget['max_sub_agents']}")
            if 'timeout' in config.budget:
                budget_lines.append(f"- Time budget: {config.budget['timeout']} seconds")
            if 'max_tool_calls' in config.budget:
                budget_lines.append(f"- Maximum tool calls: {config.budget['max_tool_calls']}")
            prompt += "\n".join(budget_lines) + "\n"

        return prompt

    def _inject_tool_blocks(
        self,
        prompt: str,
        template: PromptTemplate,
        config: SpawnConfig,
    ) -> str:
        """Replace {{TOOL_BLOCK:tool_id}} with actual instructions."""

        # Determine which tools to resolve
        if config.tool_overrides is not None:
            tool_ids = config.tool_overrides
        else:
            tool_ids = template.required_tools + template.optional_tools

        # Find all {{TOOL_BLOCK:xxx}} placeholders in the prompt
        pattern = r'\{\{TOOL_BLOCK:(\w+)\}\}'
        found_blocks = re.findall(pattern, prompt)

        for tool_id in found_blocks:
            placeholder = f"{{{{TOOL_BLOCK:{tool_id}}}}}"

            if tool_id in tool_ids:
                instructions = self.tool_registry.get_prompt_instructions(tool_id)
                if instructions:
                    prompt = prompt.replace(placeholder, instructions)
                elif tool_id in template.required_tools:
                    # Required tool exists in registry but has no instructions
                    prompt = prompt.replace(
                        placeholder,
                        f"## Tool: {tool_id}\n(Available but no detailed instructions. Use standard calling conventions.)"
                    )
                else:
                    # Optional tool, no instructions — remove silently
                    prompt = prompt.replace(placeholder, "")
            else:
                # Tool not in the resolved list — check if required
                if tool_id in template.required_tools:
                    prompt = prompt.replace(
                        placeholder,
                        f"## Tool: {tool_id} — ⚠️ NOT AVAILABLE\nThis tool was expected but is not available. Adapt your approach accordingly."
                    )
                else:
                    # Optional, not available — remove silently
                    prompt = prompt.replace(placeholder, "")

        # Clean up any double blank lines from removed blocks
        prompt = re.sub(r'\n{3,}', '\n\n', prompt)

        return prompt


# ============================================================
# 5. AGENT FACTORY
# ============================================================

class AgentFactory:
    """
    The factory that creates agents from templates and tools.

    This is the main entry point. An orchestrator (Clara) calls
    factory.create() with a template ID and gets back a configured,
    ready-to-run agent.
    """

    def __init__(
        self,
        prompt_registry: PromptRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        default_model: str = "anthropic:claude-sonnet-4-5-20250929",
        max_recursion_depth: int = 3,
    ):
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.tool_registry = tool_registry or ToolRegistry()
        self.composer = PromptComposer(self.tool_registry)
        self.default_model = default_model
        self.max_recursion_depth = max_recursion_depth

        # Spawn tracking
        self._spawn_counter = 0
        self._spawn_history: list[dict] = []

    def create(self, config: SpawnConfig | str, **kwargs) -> SpawnResult:
        """
        Create an agent from a template.

        Args:
            config: Either a SpawnConfig or a template_id string.
                    If string, a SpawnConfig is created with defaults.
            **kwargs: Override fields on SpawnConfig if config is a string.

        Returns:
            SpawnResult with the compiled agent and metadata.
        """
        # Normalize config
        if isinstance(config, str):
            config = SpawnConfig(template_id=config, **kwargs)

        # ── Step 1: Look up the template ─────────────────────
        template = self.prompt_registry.get(config.template_id)
        if template is None:
            raise ValueError(
                f"Template '{config.template_id}' not found in registry. "
                f"Available: {self.prompt_registry.list_all()[:10]}..."
            )

        # ── Step 2: Resolve tools ────────────────────────────
        tool_ids = config.tool_overrides if config.tool_overrides is not None \
            else template.required_tools + template.optional_tools

        resolved_tools, resolved_ids, missing_ids = \
            self.tool_registry.resolve(tool_ids)

        # Check for missing required tools
        missing_required = [t for t in template.required_tools if t in missing_ids]
        if missing_required:
            logger.warning(
                f"Missing required tools for {config.template_id}: {missing_required}. "
                f"Agent will be created but may not function correctly."
            )

        # ── Step 3: Compose the system prompt ────────────────
        composed_prompt = self.composer.compose(template, config)

        # ── Step 4: Determine model ──────────────────────────
        model = config.model or self.default_model

        # ── Step 5: Build the agent ──────────────────────────
        agent = self._build_agent(
            model=model,
            system_prompt=composed_prompt,
            tools=resolved_tools,
            template=template,
            config=config,
        )

        # ── Step 6: Track the spawn ──────────────────────────
        self._spawn_counter += 1
        agent_id = f"{config.template_id}-{self._spawn_counter}"

        genealogy = {
            "agent_id": agent_id,
            "template_id": config.template_id,
            "parent_agent_id": config.parent_agent_id,
            "depth": self._get_depth(config.parent_agent_id),
            "model": model,
            "tools_attached": resolved_ids,
            "tools_missing": missing_ids,
        }
        self._spawn_history.append(genealogy)

        logger.info(
            f"Spawned agent '{agent_id}' from template '{config.template_id}' "
            f"with {len(resolved_ids)} tools ({len(missing_ids)} missing)"
        )

        return SpawnResult(
            agent=agent,
            agent_id=agent_id,
            template_id=config.template_id,
            tools_attached=resolved_ids,
            tools_missing=missing_ids,
            composed_prompt=composed_prompt,
            genealogy=genealogy,
        )

    def _build_agent(
        self,
        model: str,
        system_prompt: str,
        tools: list[BaseTool],
        template: PromptTemplate,
        config: SpawnConfig,
    ) -> Any:
        """
        Build the actual LangGraph agent.

        Runtime selection:
        - deepagents (if available): Full features including planning,
          filesystem context, and sub-agent spawning
        - langgraph fallback: Basic ReAct agent loop

        The factory abstracts this — callers get the same interface.
        """
        if AGENT_RUNTIME == "deepagents" and create_deep_agent is not None:
            return self._build_deepagent(model, system_prompt, tools, template, config)
        else:
            return self._build_langgraph_agent(model, system_prompt, tools)

    def _build_deepagent(
        self,
        model: str,
        system_prompt: str,
        tools: list[BaseTool],
        template: PromptTemplate,
        config: SpawnConfig,
    ) -> Any:
        """
        Build agent using deepagents runtime (full features).

        Features:
        - Planning (write_todos)
        - Filesystem (context management)
        - Sub-agent spawning (task tool)
        """
        # Determine if this agent should be able to spawn sub-agents
        # (only if it's not already at max depth)
        subagents = None
        current_depth = self._get_depth(config.parent_agent_id)
        if current_depth < self.max_recursion_depth and not template.composable:
            # Complex, non-composable agents can delegate
            # Build sub-agent specs from composable templates
            subagents = self._build_subagent_specs(template)

        agent = create_deep_agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            subagents=subagents,
            name=config.template_id,
        )

        return agent

    def _build_langgraph_agent(
        self,
        model: str,
        system_prompt: str,
        tools: list[BaseTool],
    ) -> Any:
        """
        Build agent using LangGraph's create_react_agent (fallback).

        This provides a basic ReAct loop without:
        - Planning capabilities
        - Filesystem context
        - Sub-agent spawning

        Useful for development/testing or when deepagents is unavailable.
        """
        from langgraph.prebuilt import create_react_agent
        from langchain.chat_models import init_chat_model

        # Initialize the LLM from model string (e.g., "anthropic:claude-sonnet-4-5-20250929")
        llm = init_chat_model(model)

        # Create basic ReAct agent
        agent = create_react_agent(
            llm,
            tools,
            prompt=system_prompt,
        )

        logger.debug(f"Built LangGraph ReAct agent with {len(tools)} tools")
        return agent

    def _build_subagent_specs(self, parent_template: PromptTemplate) -> list[dict] | None:
        """
        Build sub-agent specifications for a parent agent.

        The parent can delegate to composable templates in related domains.
        """
        # Find composable templates in overlapping domains
        candidates = self.prompt_registry.search(
            domain_tags=parent_template.domain_tags,
            composable_only=True,
        )

        # Don't include the parent itself
        candidates = [c for c in candidates if c.id != parent_template.id]

        if not candidates:
            return None

        subagents = []
        for candidate in candidates[:5]:  # Limit to 5 sub-agent types
            # Resolve tools for this sub-agent
            tool_ids = candidate.required_tools + candidate.optional_tools
            tools, _, _ = self.tool_registry.resolve(tool_ids)

            subagents.append({
                "name": candidate.id,
                "description": f"{candidate.name}: {candidate.description}. "
                               f"Domains: {', '.join(candidate.domain_tags)}. "
                               f"Style: {candidate.reasoning_style.value}.",
                "system_prompt": candidate.system_prompt,
                "tools": tools,
            })

        return subagents if subagents else None

    def _get_depth(self, parent_agent_id: str | None) -> int:
        """Calculate spawn depth from genealogy."""
        if parent_agent_id is None:
            return 0

        for entry in self._spawn_history:
            if entry['agent_id'] == parent_agent_id:
                return entry['depth'] + 1

        return 0

    # ── Factory as a LangChain Tool ──────────────────────────
    # This is what makes the factory callable by an orchestrator agent

    def as_tool(self) -> BaseTool:
        """
        Expose the factory as a LangChain tool that an orchestrator
        agent (Clara) can call.

        The orchestrator calls this tool with a template_id and optional
        task description, and gets back a configured agent that it can
        then invoke with a specific task.
        """
        from langchain_core.tools import tool

        factory = self  # Closure over self

        @tool
        def spawn_agent(
            template_id: str,
            task: str,
            tool_overrides: list[str] | None = None,
            reasoning_framework: str | None = None,
        ) -> str:
            """Spawn a specialized agent from the registry to handle a specific task.

            Use this when the current task requires specialized expertise
            that a purpose-built agent would handle better than the orchestrator.

            Args:
                template_id: ID of the prompt template to use (search the registry first)
                task: Detailed description of what the agent should accomplish
                tool_overrides: Optional list of specific tool IDs to attach
                reasoning_framework: Optional - "sirp" for complex reasoning, "standard" for basic reflection

            Returns:
                The agent's response to the task.
            """
            config = SpawnConfig(
                template_id=template_id,
                tool_overrides=tool_overrides,
                reasoning_framework=ReasoningFramework(reasoning_framework) if reasoning_framework else None,
                task_context=task,
            )

            try:
                result = factory.create(config)
                # Invoke the spawned agent with the task
                response = result.agent.invoke({
                    "messages": [HumanMessage(content=task)]
                })
                # Extract the final message
                final = response.get("messages", [])
                if final:
                    return f"[Agent {result.agent_id}] {final[-1].content}"
                return f"[Agent {result.agent_id}] Task completed but no response generated."
            except Exception as e:
                return f"[Factory Error] Failed to spawn agent: {str(e)}"

        return spawn_agent

    def search_registry_tool(self) -> BaseTool:
        """
        Expose registry search as a tool for the orchestrator.

        Clara can search for the right template before spawning.
        """
        registry = self.prompt_registry

        from langchain_core.tools import tool

        @tool
        def search_prompt_registry(
            query: str | None = None,
            domain_tags: list[str] | None = None,
            reasoning_style: str | None = None,
            composable_only: bool = False,
        ) -> str:
            """Search the prompt template registry to find the right agent type for a task.

            Use this before spawn_agent to find the best template ID.

            Args:
                query: Keyword search (e.g., "financial analysis", "code review")
                domain_tags: Filter by domain (e.g., ["finance", "trading"])
                reasoning_style: Filter by style ("analytical", "creative", "exploratory", etc.)
                composable_only: Only return templates that can be used as sub-agents

            Returns:
                List of matching templates with their IDs and descriptions.
            """
            style = ReasoningStyle(reasoning_style) if reasoning_style else None
            results = registry.search(
                query=query,
                domain_tags=domain_tags,
                reasoning_style=style,
                composable_only=composable_only,
            )

            if not results:
                return "No templates found matching your criteria."

            lines = [f"Found {len(results)} matching templates:\n"]
            for t in results[:10]:  # Limit output
                lines.append(
                    f"  [{t.id}] {t.name}\n"
                    f"    Type: {t.prompt_type.value} | Style: {t.reasoning_style.value} | "
                    f"Graph: {t.recommended_graph.value}\n"
                    f"    Domains: {', '.join(t.domain_tags)}\n"
                    f"    Required tools: {t.required_tools}\n"
                    f"    Composable: {t.composable}\n"
                )

            return "\n".join(lines)

        return search_prompt_registry

    # ── Observability ────────────────────────────────────────

    @property
    def runtime(self) -> str:
        """Return the active agent runtime ('deepagents' or 'langgraph')."""
        return AGENT_RUNTIME

    @property
    def spawn_history(self) -> list[dict]:
        """Full spawn genealogy for debugging."""
        return self._spawn_history

    def print_genealogy(self) -> str:
        """Pretty-print the spawn tree."""
        if not self._spawn_history:
            return "No agents spawned yet."

        lines = ["Agent Spawn Genealogy:", "=" * 50]
        for entry in self._spawn_history:
            indent = "  " * entry['depth']
            parent = entry['parent_agent_id'] or "ROOT"
            lines.append(
                f"{indent}+-- {entry['agent_id']} "
                f"(from: {entry['template_id']}, parent: {parent}, "
                f"tools: {len(entry['tools_attached'])})"
            )
        return "\n".join(lines)
