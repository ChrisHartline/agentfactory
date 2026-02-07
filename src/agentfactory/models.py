"""
Data models for Agent Factory.

Enums, dataclasses, and type definitions used across the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ────────────────────────────────────────────────────

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
    SIRP = "sirp"
    STANDARD = "standard"


# ── Core data models ─────────────────────────────────────────

@dataclass
class PromptTemplate:
    """Complete specification for an agent persona/task."""
    id: str
    name: str
    version: str
    description: str
    system_prompt: str
    prompt_type: PromptType

    domain_tags: list[str]
    reasoning_style: ReasoningStyle
    complexity: Complexity
    composable: bool

    required_tools: list[str]
    optional_tools: list[str]

    recommended_graph: GraphType
    max_iterations: int | None = 15

    input_schema: dict | None = None
    output_schema: dict | None = None

    quality_score: float | None = None
    tested: bool = False
    author: str = "unknown"
    source: str = "curated"

    reasoning_framework: ReasoningFramework = ReasoningFramework.NONE

    parent_template: str | None = None
    compatible_with: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)


@dataclass
class ToolRegistryEntry:
    """Specification for a tool in the registry."""
    id: str
    name: str
    description: str
    tool_type: str  # "langchain_tool" | "mcp_server" | "function"

    # What gets injected into the system prompt
    prompt_instructions: str

    # Connection info
    config: dict = field(default_factory=dict)

    # The actual tool object (populated at runtime)
    tool_instance: Any = None  # BaseTool when resolved

    # Metadata
    rate_limit: str | None = None
    required_env_vars: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)


@dataclass
class SpawnConfig:
    """Configuration for a factory spawn request."""
    template_id: str
    tool_overrides: list[str] | None = None
    model: str | None = None
    max_iterations: int | None = None
    reasoning_framework: ReasoningFramework | None = None
    budget: dict | None = None
    parent_agent_id: str | None = None
    task_context: str | None = None


@dataclass
class SpawnResult:
    """What the factory returns after creating an agent."""
    agent: Any
    agent_id: str
    template_id: str
    tools_attached: list[str]
    tools_missing: list[str]
    composed_prompt: str
    genealogy: dict
