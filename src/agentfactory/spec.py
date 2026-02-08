"""
Agent Spec — declarative YAML-based agent definitions.

An agent spec is a thin YAML file that declares what an agent IS:
  - name, template, MCP servers, knowledge paths, model, reasoning

The factory reads a spec and "compiles" it into a running agent.

Example spec (agent-pm.yaml):
    name: project-manager
    template: project-manager-v2
    model: anthropic:claude-sonnet-4-5-20250929
    mcp_servers:
      - calculator
      - knowledge_base
    knowledge_paths:
      - aars/
      - sops/
    reasoning: sirp
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ── Agent Spec ──────────────────────────────────────────────

@dataclass
class AgentSpec:
    """
    Declarative agent definition loaded from YAML.

    Fields:
        name: Human-readable agent name (used for agent_id prefix)
        template: Template registry ID or path to a .md file
        mcp_servers: List of MCP server names (resolved via ServerRegistry)
        knowledge_paths: Subdirectories within the KB to expose to this agent
        model: LLM model string (e.g., "anthropic:claude-sonnet-4-5-20250929")
        reasoning: Reasoning framework ("none", "standard", "sirp")
        task: Default task context (can be overridden at runtime)
        tool_overrides: Explicit list of tool IDs (bypasses template defaults)
        max_iterations: Max agent iterations
        env: Extra environment variables for MCP server subprocesses
    """
    name: str
    template: str  # registry ID or file path ending in .md

    mcp_servers: list[str] = field(default_factory=list)
    knowledge_paths: list[str] = field(default_factory=list)
    model: str | None = None
    reasoning: str = "none"
    task: str | None = None
    tool_overrides: list[str] | None = None
    max_iterations: int | None = None
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentSpec:
        """Load an agent spec from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Agent spec not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Agent spec must be a YAML mapping, got {type(data).__name__}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> AgentSpec:
        """Create an AgentSpec from a plain dict."""
        if "name" not in data:
            raise ValueError("Agent spec requires a 'name' field")
        if "template" not in data:
            raise ValueError("Agent spec requires a 'template' field")

        return cls(
            name=data["name"],
            template=data["template"],
            mcp_servers=data.get("mcp_servers", []),
            knowledge_paths=data.get("knowledge_paths", []),
            model=data.get("model"),
            reasoning=data.get("reasoning", "none"),
            task=data.get("task"),
            tool_overrides=data.get("tool_overrides"),
            max_iterations=data.get("max_iterations"),
            env=data.get("env", {}),
        )

    @property
    def template_is_file(self) -> bool:
        """True if the template field points to a file path (vs. registry ID)."""
        return self.template.endswith(".md")


# ── MCP Server Registry ────────────────────────────────────

@dataclass
class ServerDef:
    """Definition of an MCP tool server."""
    name: str
    command: list[str]
    domain_tags: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    # Maps MCP tool names to factory tool IDs
    id_map: dict[str, str] = field(default_factory=dict)

    # Prompt instructions keyed by factory tool ID
    prompt_instructions: dict[str, str] = field(default_factory=dict)


class ServerRegistry:
    """
    Registry of available MCP tool servers.

    Maps server names to their launch commands and metadata,
    so agent specs can reference servers by name.
    """

    def __init__(self):
        self._servers: dict[str, ServerDef] = {}

    def register(self, server_def: ServerDef) -> None:
        """Register a server definition."""
        self._servers[server_def.name] = server_def
        logger.info(f"Registered server def: {server_def.name}")

    def get(self, name: str) -> ServerDef | None:
        return self._servers.get(name)

    def list_all(self) -> list[str]:
        return list(self._servers.keys())

    @property
    def count(self) -> int:
        return len(self._servers)

    def load_from_dict(self, servers: dict[str, dict]) -> int:
        """
        Load server definitions from a dict.

        Format:
            {
                "calculator": {
                    "command": ["python", "-m", "agentfactory.mcp.servers.calculator"],
                    "domain_tags": ["math", "finance"],
                    "id_map": {"calculate": "calculator", ...},
                    "prompt_instructions": {"calculator": "## Tool: calculator\n..."},
                },
                ...
            }
        """
        count = 0
        for name, config in servers.items():
            command = config.get("command", [])
            if not command:
                logger.warning(f"Server '{name}' has no command, skipping")
                continue

            self.register(ServerDef(
                name=name,
                command=command,
                domain_tags=config.get("domain_tags", []),
                env=config.get("env", {}),
                id_map=config.get("id_map", {}),
                prompt_instructions=config.get("prompt_instructions", {}),
            ))
            count += 1

        return count

    def load_from_yaml(self, path: str | Path) -> int:
        """Load server definitions from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Server registry not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if isinstance(data, dict):
            return self.load_from_dict(data)

        raise ValueError(f"Server registry YAML must be a mapping, got {type(data).__name__}")

    def to_bridge_configs(self, server_names: list[str] | None = None) -> dict[str, dict]:
        """
        Convert server defs to the dict format expected by register_mcp_tools().

        Args:
            server_names: Subset of servers to include (None = all).

        Returns:
            {server_name: {"domain_tags": [...], "id_map": {...}, "prompt_instructions": {...}}}
        """
        names = server_names or list(self._servers.keys())
        configs = {}
        for name in names:
            sdef = self._servers.get(name)
            if sdef:
                configs[name] = {
                    "command": sdef.command,
                    "domain_tags": sdef.domain_tags,
                    "id_map": sdef.id_map,
                    "prompt_instructions": sdef.prompt_instructions,
                }
        return configs


def default_server_registry() -> ServerRegistry:
    """
    Create a ServerRegistry pre-loaded with the built-in MCP servers.

    These are the servers that ship with agentfactory:
      - calculator: math + unit conversion
      - knowledge_base: markdown knowledge base reader
    """
    registry = ServerRegistry()
    python = sys.executable

    registry.register(ServerDef(
        name="calculator",
        command=[python, "-m", "agentfactory.mcp.servers.calculator"],
        domain_tags=["math", "finance", "data-science"],
        id_map={
            "calculate": "calculator",
            "convert_units": "convert_units",
        },
        prompt_instructions={
            "calculator": (
                "## Tool: calculator\n"
                "Evaluate a mathematical expression.\n\n"
                "Parameters:\n"
                "  - expression (str): Math expression to evaluate.\n"
                "    Supports: +, -, *, /, **, sqrt(), log(), sin(), cos(), pi, e\n\n"
                "Returns: JSON with 'expression' and 'result' (numeric)."
            ),
            "convert_units": (
                "## Tool: convert_units\n"
                "Convert between common units (length, weight, temperature).\n\n"
                "Parameters:\n"
                "  - value (number): The value to convert\n"
                "  - from_unit (str): Source unit (km, miles, kg, lb, celsius, fahrenheit, m, ft)\n"
                "  - to_unit (str): Target unit\n\n"
                "Returns: JSON with 'value', 'from', 'to', 'result'."
            ),
        },
    ))

    registry.register(ServerDef(
        name="knowledge_base",
        command=[python, "-m", "agentfactory.mcp.servers.knowledge_base"],
        domain_tags=["general", "business", "development"],
        id_map={
            "list_documents": "kb_list",
            "read_document": "kb_read",
            "search_knowledge": "knowledge_base",
        },
        prompt_instructions={
            "knowledge_base": (
                "## Tool: knowledge_base\n"
                "Search the team's knowledge base for lessons learned, SOPs, AARs, and conventions.\n\n"
                "Parameters:\n"
                "  - query (str): Search term or phrase\n\n"
                "Returns: Matching excerpts with filenames and context.\n\n"
                "Use this to check if there are existing lessons learned or SOPs\n"
                "before starting a new task. The knowledge base contains AARs\n"
                "(After Action Reviews), SOPs, and team conventions."
            ),
            "kb_list": (
                "## Tool: kb_list\n"
                "List all available documents in the knowledge base.\n\n"
                "Returns: Document filenames with summaries."
            ),
            "kb_read": (
                "## Tool: kb_read\n"
                "Read the full contents of a specific knowledge base document.\n\n"
                "Parameters:\n"
                "  - filename (str): Document filename (use kb_list to discover available files)\n\n"
                "Returns: Full document content."
            ),
        },
    ))

    return registry
