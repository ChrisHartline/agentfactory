"""
Agent Factory — agents that create specialized agents with tools.

Usage:
    from agentfactory import AgentFactory, SpawnConfig, PromptRegistry, ToolRegistry
    from agentfactory.templates import load_all_templates

    # Load templates
    registry = load_all_templates()

    # Create factory
    factory = AgentFactory(prompt_registry=registry)

    # Spawn an agent
    result = factory.create("project-manager-v2", task_context="Write a PRD for X")

    # Or expose as a tool for an orchestrator agent (Clara)
    spawn_tool = factory.as_tool()
    search_tool = factory.search_registry_tool()
"""

from .models import (
    Complexity,
    GraphType,
    PromptTemplate,
    PromptType,
    ReasoningFramework,
    ReasoningStyle,
    SpawnConfig,
    SpawnResult,
    ToolRegistryEntry,
)
from .registry import PromptRegistry, ToolRegistry
from .composer import PromptComposer
from .factory import AgentFactory
from .spec import AgentSpec, ServerDef, ServerRegistry, default_server_registry

__version__ = "0.1.0"

__all__ = [
    # Core
    "AgentFactory",
    "PromptComposer",
    "PromptRegistry",
    "ToolRegistry",
    # Spec-based agents
    "AgentSpec",
    "ServerDef",
    "ServerRegistry",
    "default_server_registry",
    # Config
    "SpawnConfig",
    "SpawnResult",
    # Models
    "PromptTemplate",
    "ToolRegistryEntry",
    # Enums
    "Complexity",
    "GraphType",
    "PromptType",
    "ReasoningFramework",
    "ReasoningStyle",
]
