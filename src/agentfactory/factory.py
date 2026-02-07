"""
Agent Factory — the core engine that creates agents from templates and tools.

An orchestrator (Clara) calls factory.create() with a template ID
and gets back a configured, ready-to-run agent.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

from .models import (
    PromptTemplate,
    ReasoningFramework,
    SpawnConfig,
    SpawnResult,
)
from .registry import PromptRegistry, ToolRegistry
from .composer import PromptComposer

logger = logging.getLogger(__name__)


# ── Runtime detection ────────────────────────────────────────
# Try deepagents first (full features: planning, filesystem, sub-agents)
# Fall back to LangGraph's create_react_agent (basic ReAct loop)

try:
    from deepagents import create_deep_agent  # type: ignore[import-untyped]
    AGENT_RUNTIME = "deepagents"
    logger.info("Agent runtime: deepagents (full features)")
except ImportError:
    create_deep_agent = None
    AGENT_RUNTIME = "langgraph"
    logger.info("Agent runtime: langgraph (ReAct fallback)")


class AgentFactory:
    """
    The factory that creates agents from templates and tools.

    Usage:
        factory = AgentFactory(prompt_registry, tool_registry)
        result = factory.create("code-review-agent-v2", task_context="Review X")
        response = result.agent.invoke({"messages": [HumanMessage(content="...")]})
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

        self._spawn_counter = 0
        self._spawn_history: list[dict] = []

    def create(self, config: SpawnConfig | str, **kwargs) -> SpawnResult:
        """
        Create an agent from a template.

        Args:
            config: SpawnConfig or a template_id string.
            **kwargs: Override fields on SpawnConfig if config is a string.

        Returns:
            SpawnResult with the compiled agent and metadata.
        """
        if isinstance(config, str):
            config = SpawnConfig(template_id=config, **kwargs)

        # 1. Look up template
        template = self.prompt_registry.get(config.template_id)
        if template is None:
            available = self.prompt_registry.list_all()[:10]
            raise ValueError(
                f"Template '{config.template_id}' not found. "
                f"Available: {available}..."
            )

        # 2. Resolve tools
        tool_ids = (
            config.tool_overrides
            if config.tool_overrides is not None
            else template.required_tools + template.optional_tools
        )
        resolved_tools, resolved_ids, missing_ids = self.tool_registry.resolve(
            tool_ids
        )

        missing_required = [t for t in template.required_tools if t in missing_ids]
        if missing_required:
            logger.warning(
                f"Missing required tools for {config.template_id}: {missing_required}"
            )

        # 3. Compose system prompt
        composed_prompt = self.composer.compose(template, config)

        # 4. Build the agent
        model = config.model or self.default_model
        agent = self._build_agent(model, composed_prompt, resolved_tools, template, config)

        # 5. Track the spawn
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
            f"Spawned '{agent_id}' with {len(resolved_ids)} tools "
            f"({len(missing_ids)} missing)"
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

    # ── Agent building ───────────────────────────────────────

    def _build_agent(
        self,
        model: str,
        system_prompt: str,
        tools: list[BaseTool],
        template: PromptTemplate,
        config: SpawnConfig,
    ) -> Any:
        if AGENT_RUNTIME == "deepagents" and create_deep_agent is not None:
            return self._build_deepagent(
                model, system_prompt, tools, template, config
            )
        return self._build_langgraph_agent(model, system_prompt, tools)

    def _build_deepagent(
        self,
        model: str,
        system_prompt: str,
        tools: list[BaseTool],
        template: PromptTemplate,
        config: SpawnConfig,
    ) -> Any:
        """Build agent using deepagents runtime (full features)."""
        subagents = None
        current_depth = self._get_depth(config.parent_agent_id)
        if current_depth < self.max_recursion_depth and not template.composable:
            subagents = self._build_subagent_specs(template)

        return create_deep_agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            subagents=subagents,
            name=config.template_id,
        )

    def _build_langgraph_agent(
        self,
        model: str,
        system_prompt: str,
        tools: list[BaseTool],
    ) -> Any:
        """Build agent using LangGraph's create_react_agent (fallback)."""
        from langgraph.prebuilt import create_react_agent
        from langchain.chat_models import init_chat_model

        llm = init_chat_model(model)
        agent = create_react_agent(llm, tools, prompt=system_prompt)
        logger.debug(f"Built LangGraph ReAct agent with {len(tools)} tools")
        return agent

    def _build_subagent_specs(
        self, parent_template: PromptTemplate
    ) -> list[dict] | None:
        """Build sub-agent specs from composable templates in related domains."""
        candidates = self.prompt_registry.search(
            domain_tags=parent_template.domain_tags,
            composable_only=True,
        )
        candidates = [c for c in candidates if c.id != parent_template.id]
        if not candidates:
            return None

        subagents = []
        for candidate in candidates[:5]:
            tool_ids = candidate.required_tools + candidate.optional_tools
            tools, _, _ = self.tool_registry.resolve(tool_ids)
            subagents.append(
                {
                    "name": candidate.id,
                    "description": (
                        f"{candidate.name}: {candidate.description}. "
                        f"Domains: {', '.join(candidate.domain_tags)}."
                    ),
                    "system_prompt": candidate.system_prompt,
                    "tools": tools,
                }
            )
        return subagents if subagents else None

    def _get_depth(self, parent_agent_id: str | None) -> int:
        if parent_agent_id is None:
            return 0
        for entry in self._spawn_history:
            if entry["agent_id"] == parent_agent_id:
                return entry["depth"] + 1
        return 0

    # ── Expose as LangChain tools for orchestrator agents ────

    def as_tool(self) -> BaseTool:
        """
        Expose the factory as a LangChain tool.

        An orchestrator agent (Clara) calls this to spawn
        specialized agents for tasks.
        """
        from langchain_core.tools import tool

        factory = self

        @tool
        def spawn_agent(
            template_id: str,
            task: str,
            tool_overrides: list[str] | None = None,
            reasoning_framework: str | None = None,
        ) -> str:
            """Spawn a specialized agent to handle a specific task.

            Use this when the current task requires specialized expertise.

            Args:
                template_id: ID of the prompt template (search the registry first)
                task: Detailed description of what the agent should accomplish
                tool_overrides: Optional list of specific tool IDs to attach
                reasoning_framework: Optional - "sirp" or "standard"

            Returns:
                The agent's response to the task.
            """
            config = SpawnConfig(
                template_id=template_id,
                tool_overrides=tool_overrides,
                reasoning_framework=(
                    ReasoningFramework(reasoning_framework)
                    if reasoning_framework
                    else None
                ),
                task_context=task,
            )

            try:
                result = factory.create(config)
                response = result.agent.invoke(
                    {"messages": [HumanMessage(content=task)]}
                )
                messages = response.get("messages", [])
                if messages:
                    return f"[Agent {result.agent_id}] {messages[-1].content}"
                return f"[Agent {result.agent_id}] Task completed (no output)."
            except Exception as e:
                return f"[Factory Error] {e}"

        return spawn_agent

    def search_registry_tool(self) -> BaseTool:
        """Expose registry search as a tool for the orchestrator."""
        registry = self.prompt_registry

        from langchain_core.tools import tool

        @tool
        def search_prompt_registry(
            query: str | None = None,
            domain_tags: list[str] | None = None,
            reasoning_style: str | None = None,
            composable_only: bool = False,
        ) -> str:
            """Search the prompt template registry to find the right agent type.

            Use this before spawn_agent to find the best template ID.

            Args:
                query: Keyword search (e.g., "code review", "financial analysis")
                domain_tags: Filter by domain (e.g., ["finance", "trading"])
                reasoning_style: Filter by style ("analytical", "creative", etc.)
                composable_only: Only return templates usable as sub-agents
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
            for t in results[:10]:
                lines.append(
                    f"  [{t.id}] {t.name}\n"
                    f"    Domains: {', '.join(t.domain_tags)}\n"
                    f"    Required tools: {t.required_tools}\n"
                    f"    Composable: {t.composable}\n"
                )
            return "\n".join(lines)

        return search_prompt_registry

    # ── Observability ────────────────────────────────────────

    @property
    def runtime(self) -> str:
        return AGENT_RUNTIME

    @property
    def spawn_history(self) -> list[dict]:
        return self._spawn_history

    def print_genealogy(self) -> str:
        if not self._spawn_history:
            return "No agents spawned yet."
        lines = ["Agent Spawn Genealogy:", "=" * 50]
        for entry in self._spawn_history:
            indent = "  " * entry["depth"]
            parent = entry["parent_agent_id"] or "ROOT"
            lines.append(
                f"{indent}+-- {entry['agent_id']} "
                f"(parent: {parent}, tools: {len(entry['tools_attached'])})"
            )
        return "\n".join(lines)
