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
from .spec import AgentSpec, ServerRegistry, default_server_registry

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
        server_registry: ServerRegistry | None = None,
        default_model: str = "anthropic:claude-sonnet-4-5-20250929",
        max_recursion_depth: int = 3,
    ):
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.tool_registry = tool_registry or ToolRegistry()
        self.server_registry = server_registry or default_server_registry()
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

    def from_spec(
        self,
        spec: AgentSpec | str,
        task: str | None = None,
    ) -> SpawnResult:
        """
        Create an agent from a declarative YAML spec.

        This is the "compiler" entry point: give it a spec (or path to one),
        and it assembles template + MCP servers + knowledge paths into a
        running agent.

        Args:
            spec: An AgentSpec, or a path to a YAML spec file.
            task: Override the task from the spec (optional).

        Returns:
            SpawnResult with the compiled agent and metadata.
        """
        from pathlib import Path
        from .mcp.manager import ToolServerManager
        from .mcp.bridge import register_mcp_tools
        import os

        if isinstance(spec, (str, Path)):
            spec = AgentSpec.from_yaml(spec)

        # 1. Resolve the template
        if spec.template_is_file:
            template = PromptTemplate.from_markdown(spec.template, name=spec.name)
            # Register it in the prompt registry so the rest of the pipeline works
            self.prompt_registry._templates[template.id] = template
        else:
            template = self.prompt_registry.get(spec.template)
            if template is None:
                raise ValueError(
                    f"Template '{spec.template}' not found in registry. "
                    f"Use a registry ID or a path ending in .md"
                )

        # 2. Start MCP servers declared in the spec
        manager = ToolServerManager()
        if spec.mcp_servers:
            project_root = Path(__file__).parent.parent.parent
            base_env = {
                **os.environ,
                "PYTHONPATH": str(project_root / "src"),
            }

            # Pass knowledge_paths to KB server via env var
            if spec.knowledge_paths:
                base_env["KNOWLEDGE_PATHS"] = ",".join(spec.knowledge_paths)

            # Merge any extra env from the spec
            base_env.update(spec.env)

            for server_name in spec.mcp_servers:
                server_def = self.server_registry.get(server_name)
                if server_def is None:
                    logger.warning(
                        f"MCP server '{server_name}' not found in server registry, skipping"
                    )
                    continue

                server_env = {**base_env, **server_def.env}
                manager.register_server(server_name, server_def.command, env=server_env)

                try:
                    tools = manager.start(server_name)
                    tool_names = [t["name"] for t in tools]
                    logger.info(f"  [{server_name}] started: {tool_names}")
                except Exception as e:
                    logger.error(f"  [{server_name}] failed to start: {e}")

            # Register discovered tools in the tool registry
            bridge_configs = self.server_registry.to_bridge_configs(spec.mcp_servers)
            register_mcp_tools(manager, self.tool_registry, bridge_configs)

        # 3. Build SpawnConfig from the spec
        task_context = task or spec.task
        reasoning = ReasoningFramework(spec.reasoning) if spec.reasoning else None

        config = SpawnConfig(
            template_id=template.id,
            tool_overrides=spec.tool_overrides,
            model=spec.model,
            max_iterations=spec.max_iterations,
            reasoning_framework=reasoning,
            task_context=task_context,
        )

        # 4. Delegate to create() for the rest of the pipeline
        result = self.create(config)

        # Attach the manager so callers can shut down MCP servers
        result._mcp_manager = manager

        return result

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
