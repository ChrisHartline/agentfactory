"""
Unit tests for the Agent Factory core components.

These tests verify the registry, composer, and factory logic
WITHOUT requiring LLM API calls or MCP servers.
"""

import pytest
from pathlib import Path

from agentfactory.models import (
    Complexity,
    PromptTemplate,
    PromptType,
    ReasoningFramework,
    ReasoningStyle,
    GraphType,
    SpawnConfig,
    ToolRegistryEntry,
)
from agentfactory.registry import PromptRegistry, ToolRegistry
from agentfactory.composer import PromptComposer
from agentfactory.templates import load_all_templates


# ── Fixtures ─────────────────────────────────────────────────

def _make_template(**overrides) -> PromptTemplate:
    """Create a test template with sensible defaults."""
    defaults = {
        "id": "test-agent",
        "name": "Test Agent",
        "version": "1.0.0",
        "description": "A test agent",
        "system_prompt": "You are a test agent.\n\n{{TOOL_BLOCK:calculator}}\n\n{{TOOL_BLOCK:web_search}}",
        "prompt_type": PromptType.PERSONA,
        "domain_tags": ["testing"],
        "reasoning_style": ReasoningStyle.ANALYTICAL,
        "complexity": Complexity.MODERATE,
        "composable": True,
        "required_tools": ["calculator"],
        "optional_tools": ["web_search"],
        "recommended_graph": GraphType.REACT,
    }
    defaults.update(overrides)
    return PromptTemplate(**defaults)


# ── PromptRegistry Tests ─────────────────────────────────────

class TestPromptRegistry:
    def test_load_from_dict(self):
        registry = PromptRegistry()
        templates = {
            "test-agent": {
                "id": "test-agent",
                "name": "Test Agent",
                "system_prompt": "You are a test agent.",
                "domain_tags": ["testing"],
            }
        }
        count = registry.load_from_dict(templates)
        assert count == 1
        assert registry.get("test-agent") is not None
        assert registry.get("test-agent").name == "Test Agent"

    def test_get_missing_returns_none(self):
        registry = PromptRegistry()
        assert registry.get("nonexistent") is None

    def test_search_by_domain(self):
        registry = PromptRegistry()
        registry.load_from_dict({
            "a": {"id": "a", "name": "A", "system_prompt": "a", "domain_tags": ["finance"]},
            "b": {"id": "b", "name": "B", "system_prompt": "b", "domain_tags": ["dev"]},
        })
        results = registry.search(domain_tags=["finance"])
        assert len(results) == 1
        assert results[0].id == "a"

    def test_search_by_query(self):
        registry = PromptRegistry()
        registry.load_from_dict({
            "a": {"id": "a", "name": "Code Reviewer", "system_prompt": "a", "domain_tags": ["dev"]},
            "b": {"id": "b", "name": "Data Analyst", "system_prompt": "b", "domain_tags": ["data"]},
        })
        results = registry.search(query="code")
        assert len(results) == 1
        assert results[0].id == "a"

    def test_load_all_templates(self):
        """Verify all bundled templates load without error."""
        registry = load_all_templates()
        assert registry.count > 0
        # Should have the expanded v2 templates
        assert registry.get("project-manager-v2") is not None
        assert registry.get("code-review-agent-v2") is not None

    def test_yaml_templates_load(self):
        """Verify YAML templates load from bundled file."""
        yaml_path = Path(__file__).parent.parent / "src" / "agentfactory" / "templates" / "registry.yaml"
        if yaml_path.exists():
            registry = PromptRegistry()
            count = registry.load_from_yaml(yaml_path)
            assert count > 0


# ── ToolRegistry Tests ───────────────────────────────────────

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        entry = ToolRegistryEntry(
            id="calc",
            name="calculator",
            description="Math",
            tool_type="function",
            prompt_instructions="## Tool: calculator\nDoes math.",
        )
        registry.register(entry)
        assert registry.get("calc") is not None
        assert registry.get("calc").name == "calculator"

    def test_resolve_missing(self):
        registry = ToolRegistry()
        tools, resolved, missing = registry.resolve(["nonexistent"])
        assert len(tools) == 0
        assert len(missing) == 1
        assert "nonexistent" in missing

    def test_get_prompt_instructions(self):
        registry = ToolRegistry()
        entry = ToolRegistryEntry(
            id="calc",
            name="calculator",
            description="Math",
            tool_type="function",
            prompt_instructions="## Tool: calculator\nDoes math.",
        )
        registry.register(entry)
        assert "calculator" in registry.get_prompt_instructions("calc")


# ── PromptComposer Tests ────────────────────────────────────

class TestPromptComposer:
    def test_tool_block_injection(self):
        tool_registry = ToolRegistry()
        tool_registry.register(ToolRegistryEntry(
            id="calculator",
            name="calculator",
            description="Math",
            tool_type="function",
            prompt_instructions="## Tool: calculator\nEvaluate math expressions.",
        ))

        composer = PromptComposer(tool_registry)
        template = _make_template()
        config = SpawnConfig(template_id="test-agent")

        result = composer.compose(template, config)

        # calculator block should be injected
        assert "## Tool: calculator" in result
        assert "Evaluate math expressions" in result
        # web_search is optional and not registered, should be removed
        assert "{{TOOL_BLOCK:web_search}}" not in result

    def test_missing_required_tool_flagged(self):
        tool_registry = ToolRegistry()
        composer = PromptComposer(tool_registry)
        # Use a template where calculator is required but NOT in tool_overrides
        template = _make_template()
        config = SpawnConfig(template_id="test-agent", tool_overrides=[])

        result = composer.compose(template, config)

        # calculator is required but excluded via tool_overrides=[]
        assert "NOT AVAILABLE" in result

    def test_task_context_injected(self):
        tool_registry = ToolRegistry()
        composer = PromptComposer(tool_registry)
        template = _make_template(system_prompt="Base prompt.", required_tools=[], optional_tools=[])
        config = SpawnConfig(template_id="test-agent", task_context="Analyze AAPL")

        result = composer.compose(template, config)
        assert "Analyze AAPL" in result

    def test_sirp_injection(self):
        tool_registry = ToolRegistry()
        composer = PromptComposer(tool_registry)
        template = _make_template(system_prompt="Base.", required_tools=[], optional_tools=[])
        config = SpawnConfig(
            template_id="test-agent",
            reasoning_framework=ReasoningFramework.SIRP,
        )

        result = composer.compose(template, config)
        assert "SIRP" in result
        assert "<thinking>" in result
