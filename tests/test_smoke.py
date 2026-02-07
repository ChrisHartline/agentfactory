"""
Smoke test — verifies the full MCP server + Factory pipeline
WITHOUT requiring an LLM API key.

Tests:
1. MCP tool server starts and responds to tool calls
2. Factory composes prompts with injected tool instructions
3. Knowledge base server reads markdown files
"""

import json
import os
import sys
import subprocess
import tempfile
import pytest
from pathlib import Path

from agentfactory import (
    AgentFactory,
    PromptRegistry,
    ToolRegistry,
    SpawnConfig,
    ToolRegistryEntry,
)
from agentfactory.templates import load_all_templates
from agentfactory.mcp.manager import ToolServerManager
from agentfactory.mcp.bridge import register_mcp_tools


# ── MCP Server Tests ────────────────────────────────────────

class TestCalculatorServer:
    """Test the calculator MCP server via subprocess."""

    def _call_server(self, method: str, params: dict) -> dict:
        """Send a single JSON-RPC request to the calculator server."""
        request = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        }) + "\n"

        result = subprocess.run(
            [sys.executable, "-m", "agentfactory.mcp.servers.calculator"],
            input=request,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
        )

        if result.returncode != 0 and result.stderr:
            # Server may log to stderr, that's fine
            pass

        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        assert len(lines) >= 1, f"No response from server. stderr: {result.stderr[:200]}"

        return json.loads(lines[0])

    def test_tools_list(self):
        response = self._call_server("tools/list", {})
        assert "result" in response
        tools = response["result"]
        tool_names = [t["name"] for t in tools]
        assert "calculate" in tool_names
        assert "convert_units" in tool_names

    def test_calculate(self):
        response = self._call_server("tools/call", {
            "name": "calculate",
            "arguments": {"expression": "2 + 2"},
        })
        assert "result" in response
        assert response["result"]["result"] == 4.0

    def test_calculate_complex(self):
        response = self._call_server("tools/call", {
            "name": "calculate",
            "arguments": {"expression": "sqrt(16) + pi"},
        })
        assert "result" in response
        import math
        expected = 4.0 + math.pi
        assert abs(response["result"]["result"] - expected) < 0.001

    def test_convert_units(self):
        response = self._call_server("tools/call", {
            "name": "convert_units",
            "arguments": {"value": 100, "from_unit": "km", "to_unit": "miles"},
        })
        assert "result" in response
        assert abs(response["result"]["result"] - 62.1371) < 0.01


# ── Knowledge Base Server Tests ──────────────────────────────

class TestKnowledgeBaseServer:
    """Test the knowledge base MCP server with a temp directory."""

    @pytest.fixture
    def kb_dir(self, tmp_path):
        """Create a temporary knowledge base with test documents."""
        doc1 = tmp_path / "aar-vercel-deploy.md"
        doc1.write_text(
            "# AAR: Vercel Deployment\n"
            "Project: AI Doppelganger\n"
            "Date: 2025-02-03\n"
            "Outcome: SUCCESS\n\n"
            "## Root Causes\n"
            "1. Missing serverless function config\n"
            "2. Incorrect build output directory\n"
        )

        doc2 = tmp_path / "sop-git-workflow.md"
        doc2.write_text(
            "# SOP: Git Workflow\n"
            "Always create feature branches from main.\n"
            "Use conventional commits.\n"
        )

        return tmp_path

    def _call_server(self, method: str, params: dict, kb_dir: Path) -> dict:
        request = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        }) + "\n"

        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
            "KNOWLEDGE_BASE_DIR": str(kb_dir),
        }

        result = subprocess.run(
            [sys.executable, "-m", "agentfactory.mcp.servers.knowledge_base"],
            input=request,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        assert len(lines) >= 1, f"No response. stderr: {result.stderr[:200]}"
        return json.loads(lines[0])

    def test_list_documents(self, kb_dir):
        response = self._call_server("tools/call", {
            "name": "list_documents",
            "arguments": {},
        }, kb_dir)
        assert "result" in response
        result = response["result"]
        assert result["document_count"] == 2

    def test_read_document(self, kb_dir):
        response = self._call_server("tools/call", {
            "name": "read_document",
            "arguments": {"filename": "aar-vercel-deploy.md"},
        }, kb_dir)
        assert "result" in response
        assert "Vercel Deployment" in response["result"]["content"]

    def test_search_knowledge(self, kb_dir):
        response = self._call_server("tools/call", {
            "name": "search_knowledge",
            "arguments": {"query": "serverless"},
        }, kb_dir)
        assert "result" in response
        assert response["result"]["match_count"] > 0
        assert "serverless" in response["result"]["matches"][0]["excerpt"].lower()

    def test_path_traversal_blocked(self, kb_dir):
        response = self._call_server("tools/call", {
            "name": "read_document",
            "arguments": {"filename": "../../etc/passwd"},
        }, kb_dir)
        assert "result" in response
        result = response["result"]
        assert "error" in result or "Access denied" in str(result)


# ── Factory Integration Test (no LLM) ───────────────────────

class TestFactoryIntegration:
    """Test the factory pipeline without calling an LLM."""

    def test_compose_prompt_with_tools(self):
        """Verify factory composes a prompt with tool instructions injected."""
        prompt_registry = load_all_templates()
        tool_registry = ToolRegistry()

        # Register a mock tool
        tool_registry.register(ToolRegistryEntry(
            id="knowledge_base",
            name="knowledge_base",
            description="Search knowledge base",
            tool_type="mcp_server",
            prompt_instructions="## Tool: knowledge_base\nSearch the team's knowledge base for lessons learned, SOPs, and AARs.\n\nTools: list_documents, read_document, search_knowledge",
            domain_tags=["general"],
        ))

        factory = AgentFactory(
            prompt_registry=prompt_registry,
            tool_registry=tool_registry,
        )

        # Compose a prompt (no LLM call)
        template = prompt_registry.get("project-manager-v2")
        assert template is not None

        config = SpawnConfig(
            template_id="project-manager-v2",
            task_context="Create a PRD for the Agent Factory project",
        )

        composed = factory.composer.compose(template, config)

        # Should have the knowledge base tool injected
        assert "knowledge_base" in composed
        assert "lessons learned" in composed.lower() or "knowledge base" in composed.lower()
        # Should have the task context
        assert "Agent Factory" in composed

    def test_factory_genealogy(self):
        """Verify spawn tracking works."""
        registry = PromptRegistry()
        registry.load_from_dict({
            "test": {
                "id": "test",
                "name": "Test",
                "system_prompt": "You are a test.",
                "domain_tags": ["testing"],
                "required_tools": [],
                "optional_tools": [],
            }
        })

        factory = AgentFactory(prompt_registry=registry)

        # We can't actually create an agent (no LLM), but we can
        # verify the registry and composition work
        template = registry.get("test")
        assert template is not None
        config = SpawnConfig(template_id="test")
        composed = factory.composer.compose(template, config)
        assert "You are a test" in composed
