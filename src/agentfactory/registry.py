"""
Prompt Registry and Tool Registry.

PromptRegistry: Searchable library of agent prompt templates.
ToolRegistry:   Catalog of available tools and MCP servers.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .models import (
    Complexity,
    GraphType,
    PromptTemplate,
    PromptType,
    ReasoningFramework,
    ReasoningStyle,
    ToolRegistryEntry,
)

logger = logging.getLogger(__name__)


# ── Prompt Registry ──────────────────────────────────────────

class PromptRegistry:
    """
    Searchable library of prompt templates.

    The factory looks up what kind of agent to build
    based on task requirements.
    """

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}

    def load_from_yaml(self, path: str | Path) -> int:
        """Load templates from a YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        count = 0
        for entry in data:
            template = self._entry_to_template(entry, source="yaml")
            self._templates[template.id] = template
            count += 1

        logger.info(f"Loaded {count} templates from {path}")
        return count

    def load_from_dict(self, templates: dict[str, dict]) -> int:
        """Load templates from a Python dict (overrides existing entries)."""
        count = 0
        for tid, entry in templates.items():
            template = self._entry_to_template(entry, source="curated")
            self._templates[template.id] = template
            count += 1

        logger.info(f"Loaded {count} templates from dict")
        return count

    def get(self, template_id: str) -> PromptTemplate | None:
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
        """Search the registry with filters."""
        results = list(self._templates.values())

        if domain_tags:
            results = [
                t for t in results if any(d in t.domain_tags for d in domain_tags)
            ]
        if reasoning_style:
            results = [t for t in results if t.reasoning_style == reasoning_style]
        if prompt_type:
            results = [t for t in results if t.prompt_type == prompt_type]
        if requires_tools:
            results = [
                t
                for t in results
                if all(
                    tool in t.required_tools + t.optional_tools
                    for tool in requires_tools
                )
            ]
        if composable_only:
            results = [t for t in results if t.composable]
        if min_quality > 0:
            results = [
                t
                for t in results
                if t.quality_score is not None and t.quality_score >= min_quality
            ]
        if max_complexity:
            order = {Complexity.ATOMIC: 0, Complexity.MODERATE: 1, Complexity.COMPLEX: 2}
            ceiling = order[max_complexity]
            results = [t for t in results if order.get(t.complexity, 0) <= ceiling]

        if query:
            query_lower = query.lower()
            scored = []
            for t in results:
                searchable = f"{t.name} {t.description} {' '.join(t.domain_tags)}".lower()
                score = sum(1 for word in query_lower.split() if word in searchable)
                if score > 0:
                    scored.append((score, t))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [t for _, t in scored]

        return results

    def list_all(self) -> list[str]:
        return list(self._templates.keys())

    @property
    def count(self) -> int:
        return len(self._templates)

    @staticmethod
    def _entry_to_template(entry: dict, source: str = "curated") -> PromptTemplate:
        return PromptTemplate(
            id=entry["id"],
            name=entry["name"],
            version=entry.get("version", "1.0.0"),
            description=entry.get("description", ""),
            system_prompt=entry["system_prompt"],
            prompt_type=PromptType(entry.get("prompt_type", "persona")),
            domain_tags=entry.get("domain_tags", []),
            reasoning_style=ReasoningStyle(entry.get("reasoning_style", "analytical")),
            complexity=Complexity(entry.get("complexity", "moderate")),
            composable=entry.get("composable", True),
            required_tools=entry.get("required_tools", []),
            optional_tools=entry.get("optional_tools", []),
            recommended_graph=GraphType(entry.get("recommended_graph", "react")),
            max_iterations=entry.get("max_iterations", 15),
            input_schema=entry.get("input_schema"),
            output_schema=entry.get("output_schema"),
            quality_score=entry.get("quality_score"),
            tested=entry.get("tested", False),
            author=entry.get("author", "unknown"),
            source=entry.get("source", source),
            reasoning_framework=ReasoningFramework(
                entry.get("reasoning_framework", "none")
            ),
        )


# ── Tool Registry ────────────────────────────────────────────

class ToolRegistry:
    """
    Catalog of available tools and MCP servers.

    Each entry knows:
    - What the tool does (for matching against templates)
    - How to describe it to the agent (prompt_instructions)
    - The actual callable (tool_instance, populated at runtime)
    """

    def __init__(self):
        self._tools: dict[str, ToolRegistryEntry] = {}

    def register(self, entry: ToolRegistryEntry) -> None:
        self._tools[entry.id] = entry
        logger.info(f"Registered tool: {entry.id} ({entry.tool_type})")

    def register_langchain_tool(
        self,
        tool_id: str,
        tool: Any,  # BaseTool
        prompt_instructions: str | None = None,
        domain_tags: list[str] | None = None,
    ) -> None:
        """Register an existing LangChain tool."""
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
        from langchain_core.tools import StructuredTool

        tool = StructuredTool.from_function(
            func=func, name=name, description=description
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
        return self._tools.get(tool_id)

    def resolve(
        self, tool_ids: list[str]
    ) -> tuple[list[Any], list[str], list[str]]:
        """
        Resolve tool IDs into actual tool instances.

        Returns: (resolved_tools, resolved_ids, missing_ids)
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
        entry = self._tools.get(tool_id)
        return entry.prompt_instructions if entry else None

    def search(
        self,
        query: str | None = None,
        domain_tags: list[str] | None = None,
    ) -> list[ToolRegistryEntry]:
        results = list(self._tools.values())

        if domain_tags:
            results = [
                t for t in results if any(d in t.domain_tags for d in domain_tags)
            ]
        if query:
            q = query.lower()
            results = [
                t
                for t in results
                if q in t.name.lower() or q in t.description.lower()
            ]
        return results

    def list_all(self) -> list[str]:
        return list(self._tools.keys())

    @staticmethod
    def _auto_instructions(tool: Any) -> str:
        """Generate prompt instructions from a LangChain tool."""
        schema = tool.args_schema.schema() if tool.args_schema else {}
        params = schema.get("properties", {})
        lines = [f"## Tool: {tool.name}", f"{tool.description}", ""]
        if params:
            lines.append("Parameters:")
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                lines.append(f"  - {pname} ({ptype}): {pdesc}")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._tools)
