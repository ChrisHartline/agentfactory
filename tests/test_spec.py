"""
Tests for the spec-based agent creation system.

Covers:
- AgentSpec loading from YAML and dict
- ServerRegistry loading and lookup
- PromptTemplate.from_markdown() (external template files)
- Knowledge paths filtering in KB server
- Factory.from_spec() pipeline (no LLM)
"""

import json
import os
import sys
import subprocess
import pytest
from pathlib import Path

from agentfactory import (
    AgentFactory,
    AgentSpec,
    PromptTemplate,
    ServerDef,
    ServerRegistry,
    default_server_registry,
)
from agentfactory.spec import default_server_registry
from agentfactory.templates import load_all_templates


# ── AgentSpec Tests ─────────────────────────────────────────

class TestAgentSpec:
    def test_from_dict_minimal(self):
        spec = AgentSpec.from_dict({
            "name": "test-agent",
            "template": "project-manager-v2",
        })
        assert spec.name == "test-agent"
        assert spec.template == "project-manager-v2"
        assert spec.mcp_servers == []
        assert spec.knowledge_paths == []
        assert spec.reasoning == "none"
        assert not spec.template_is_file

    def test_from_dict_full(self):
        spec = AgentSpec.from_dict({
            "name": "pm-agent",
            "template": "project-manager-v2",
            "model": "anthropic:claude-sonnet-4-5-20250929",
            "mcp_servers": ["calculator", "knowledge_base"],
            "knowledge_paths": ["aars/", "sops/"],
            "reasoning": "sirp",
            "task": "Write a PRD",
            "max_iterations": 10,
        })
        assert spec.name == "pm-agent"
        assert spec.mcp_servers == ["calculator", "knowledge_base"]
        assert spec.knowledge_paths == ["aars/", "sops/"]
        assert spec.reasoning == "sirp"
        assert spec.task == "Write a PRD"
        assert spec.max_iterations == 10

    def test_from_dict_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            AgentSpec.from_dict({"template": "test"})

    def test_from_dict_missing_template_raises(self):
        with pytest.raises(ValueError, match="template"):
            AgentSpec.from_dict({"name": "test"})

    def test_from_yaml(self, tmp_path):
        spec_file = tmp_path / "agent.yaml"
        spec_file.write_text(
            "name: test-agent\n"
            "template: project-manager-v2\n"
            "mcp_servers:\n"
            "  - calculator\n"
            "knowledge_paths:\n"
            "  - aars/\n"
            "reasoning: standard\n"
        )
        spec = AgentSpec.from_yaml(spec_file)
        assert spec.name == "test-agent"
        assert spec.template == "project-manager-v2"
        assert spec.mcp_servers == ["calculator"]
        assert spec.knowledge_paths == ["aars/"]
        assert spec.reasoning == "standard"

    def test_from_yaml_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            AgentSpec.from_yaml("/nonexistent/agent.yaml")

    def test_template_is_file(self):
        spec = AgentSpec.from_dict({
            "name": "test",
            "template": "templates/agent-pm.md",
        })
        assert spec.template_is_file

        spec2 = AgentSpec.from_dict({
            "name": "test",
            "template": "project-manager-v2",
        })
        assert not spec2.template_is_file


# ── ServerRegistry Tests ────────────────────────────────────

class TestServerRegistry:
    def test_register_and_get(self):
        reg = ServerRegistry()
        reg.register(ServerDef(
            name="calc",
            command=["python", "-m", "calc"],
            domain_tags=["math"],
        ))
        assert reg.get("calc") is not None
        assert reg.get("calc").domain_tags == ["math"]
        assert reg.get("nonexistent") is None

    def test_load_from_dict(self):
        reg = ServerRegistry()
        count = reg.load_from_dict({
            "calculator": {
                "command": ["python", "-m", "agentfactory.mcp.servers.calculator"],
                "domain_tags": ["math"],
                "id_map": {"calculate": "calculator"},
            },
            "empty": {},  # no command, should be skipped
        })
        assert count == 1
        assert reg.get("calculator") is not None
        assert reg.get("empty") is None

    def test_load_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "servers.yaml"
        yaml_file.write_text(
            "calculator:\n"
            "  command: [python, -m, agentfactory.mcp.servers.calculator]\n"
            "  domain_tags: [math, finance]\n"
            "  id_map:\n"
            "    calculate: calculator\n"
        )
        reg = ServerRegistry()
        count = reg.load_from_yaml(yaml_file)
        assert count == 1
        assert reg.get("calculator").domain_tags == ["math", "finance"]

    def test_to_bridge_configs(self):
        reg = ServerRegistry()
        reg.register(ServerDef(
            name="calc",
            command=["python", "-m", "calc"],
            id_map={"calculate": "calculator"},
            prompt_instructions={"calculator": "## Tool: calculator\nDoes math."},
        ))
        configs = reg.to_bridge_configs(["calc"])
        assert "calc" in configs
        assert configs["calc"]["id_map"] == {"calculate": "calculator"}
        assert "calculator" in configs["calc"]["prompt_instructions"]

    def test_to_bridge_configs_filters(self):
        reg = ServerRegistry()
        reg.register(ServerDef(name="a", command=["a"]))
        reg.register(ServerDef(name="b", command=["b"]))
        configs = reg.to_bridge_configs(["a"])
        assert "a" in configs
        assert "b" not in configs

    def test_default_server_registry(self):
        reg = default_server_registry()
        assert reg.get("calculator") is not None
        assert reg.get("knowledge_base") is not None
        assert reg.count >= 2


# ── PromptTemplate.from_markdown() Tests ────────────────────

class TestMarkdownTemplates:
    def test_plain_markdown(self, tmp_path):
        md_file = tmp_path / "agent-pm.md"
        md_file.write_text(
            "# Project Manager Agent\n\n"
            "You are a senior project manager specialized in PRDs.\n"
        )
        template = PromptTemplate.from_markdown(md_file)
        assert template.id == "agent-pm"
        assert "senior project manager" in template.system_prompt
        assert template.source == str(md_file)

    def test_markdown_with_front_matter(self, tmp_path):
        md_file = tmp_path / "agent-arch.md"
        md_file.write_text(
            "---\n"
            "domain_tags: [architecture, development]\n"
            "required_tools: [knowledge_base]\n"
            "reasoning_style: analytical\n"
            "description: Enterprise architect agent\n"
            "---\n"
            "# System Architect\n\n"
            "You are an enterprise system architect.\n"
        )
        template = PromptTemplate.from_markdown(md_file)
        assert template.id == "agent-arch"
        assert template.domain_tags == ["architecture", "development"]
        assert template.required_tools == ["knowledge_base"]
        assert template.description == "Enterprise architect agent"
        assert "enterprise system architect" in template.system_prompt
        # Front matter should not be in the system prompt
        assert "domain_tags" not in template.system_prompt

    def test_custom_id_and_name(self, tmp_path):
        md_file = tmp_path / "template.md"
        md_file.write_text("You are an agent.")
        template = PromptTemplate.from_markdown(
            md_file, template_id="custom-id", name="Custom Agent"
        )
        assert template.id == "custom-id"
        assert template.name == "Custom Agent"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PromptTemplate.from_markdown("/nonexistent/template.md")


# ── Knowledge Paths Filtering Tests ─────────────────────────

class TestKnowledgePathsFiltering:
    """Test that KNOWLEDGE_PATHS env var filters KB documents."""

    @pytest.fixture
    def kb_dir(self, tmp_path):
        """Create a KB with subdirectories."""
        (tmp_path / "aars").mkdir()
        (tmp_path / "sops").mkdir()
        (tmp_path / "standards").mkdir()

        (tmp_path / "aars" / "deploy-aar.md").write_text(
            "# AAR: Deployment\nDeployed to prod.\n"
        )
        (tmp_path / "sops" / "git-workflow.md").write_text(
            "# SOP: Git\nAlways use feature branches.\n"
        )
        (tmp_path / "standards" / "coding.md").write_text(
            "# Coding Standards\nUse type hints.\n"
        )
        return tmp_path

    def _call_server(self, method, params, kb_dir, knowledge_paths=None):
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
        if knowledge_paths:
            env["KNOWLEDGE_PATHS"] = knowledge_paths

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

    def test_no_filter_sees_all(self, kb_dir):
        response = self._call_server("tools/call", {
            "name": "list_documents",
            "arguments": {},
        }, kb_dir, knowledge_paths=None)
        assert response["result"]["document_count"] == 3

    def test_filter_aars_only(self, kb_dir):
        response = self._call_server("tools/call", {
            "name": "list_documents",
            "arguments": {},
        }, kb_dir, knowledge_paths="aars/")
        assert response["result"]["document_count"] == 1
        assert "aars/" in response["result"]["documents"][0]["filename"]

    def test_filter_multiple_paths(self, kb_dir):
        response = self._call_server("tools/call", {
            "name": "list_documents",
            "arguments": {},
        }, kb_dir, knowledge_paths="aars/,sops/")
        assert response["result"]["document_count"] == 2

    def test_filter_blocks_read(self, kb_dir):
        response = self._call_server("tools/call", {
            "name": "read_document",
            "arguments": {"filename": "standards/coding.md"},
        }, kb_dir, knowledge_paths="aars/,sops/")
        result = response["result"]
        assert "error" in result or "Access denied" in str(result)

    def test_filter_allows_read(self, kb_dir):
        response = self._call_server("tools/call", {
            "name": "read_document",
            "arguments": {"filename": "aars/deploy-aar.md"},
        }, kb_dir, knowledge_paths="aars/")
        assert "content" in response["result"]
        assert "Deployment" in response["result"]["content"]

    def test_filter_search(self, kb_dir):
        response = self._call_server("tools/call", {
            "name": "search_knowledge",
            "arguments": {"query": "type hints"},
        }, kb_dir, knowledge_paths="aars/,sops/")
        # "type hints" is only in standards/coding.md, which is filtered out
        assert response["result"]["match_count"] == 0


# ── Factory.from_spec() Pipeline Tests ──────────────────────

class TestFactoryFromSpec:
    """Test from_spec() pipeline without LLM calls."""

    def test_from_spec_dict_registry_template(self):
        """Spec referencing a registry template composes correctly."""
        prompt_registry = load_all_templates()
        factory = AgentFactory(prompt_registry=prompt_registry)

        spec = AgentSpec.from_dict({
            "name": "pm-test",
            "template": "project-manager-v2",
            "task": "Write a PRD for Agent Factory",
            "reasoning": "standard",
        })

        # We can't build the actual agent (no LLM), but we can verify
        # the template resolution and prompt composition
        template = prompt_registry.get(spec.template)
        assert template is not None

        from agentfactory.models import ReasoningFramework, SpawnConfig
        config = SpawnConfig(
            template_id=template.id,
            reasoning_framework=ReasoningFramework(spec.reasoning),
            task_context=spec.task,
        )
        composed = factory.composer.compose(template, config)
        assert "Agent Factory" in composed
        assert "Self-Reflection" in composed  # standard reasoning

    def test_from_spec_markdown_template(self, tmp_path):
        """Spec referencing an external .md template."""
        md_file = tmp_path / "agent-test.md"
        md_file.write_text(
            "---\n"
            "domain_tags: [testing]\n"
            "required_tools: []\n"
            "---\n"
            "# Test Agent\n\n"
            "You are a test agent for spec validation.\n"
        )

        factory = AgentFactory()
        spec = AgentSpec.from_dict({
            "name": "spec-test",
            "template": str(md_file),
            "task": "Validate the spec system",
        })

        assert spec.template_is_file
        template = PromptTemplate.from_markdown(spec.template, name=spec.name)
        assert "test agent for spec validation" in template.system_prompt

        from agentfactory.models import SpawnConfig
        config = SpawnConfig(
            template_id=template.id,
            task_context=spec.task,
        )
        composed = factory.composer.compose(template, config)
        assert "Validate the spec system" in composed

    def test_from_spec_yaml_file(self, tmp_path):
        """Load spec from a YAML file and verify pipeline."""
        spec_file = tmp_path / "agent.yaml"
        spec_file.write_text(
            "name: yaml-test\n"
            "template: project-manager-v2\n"
            "reasoning: sirp\n"
            "task: Test YAML loading\n"
        )

        prompt_registry = load_all_templates()
        factory = AgentFactory(prompt_registry=prompt_registry)

        spec = AgentSpec.from_yaml(spec_file)
        assert spec.name == "yaml-test"
        assert spec.reasoning == "sirp"

        template = prompt_registry.get(spec.template)
        assert template is not None

        from agentfactory.models import ReasoningFramework, SpawnConfig
        config = SpawnConfig(
            template_id=template.id,
            reasoning_framework=ReasoningFramework(spec.reasoning),
            task_context=spec.task,
        )
        composed = factory.composer.compose(template, config)
        assert "SIRP" in composed
        assert "Test YAML loading" in composed

    def test_server_registry_default_has_builtins(self):
        """Factory gets default server registry with built-in servers."""
        factory = AgentFactory()
        assert factory.server_registry.get("calculator") is not None
        assert factory.server_registry.get("knowledge_base") is not None
