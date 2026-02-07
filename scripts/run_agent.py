#!/usr/bin/env python3
"""
Run Agent — end-to-end: MCP servers -> Factory -> Live Agent.

Usage:
    # List available templates
    python scripts/run_agent.py --list

    # Dry run (compose prompt, no LLM)
    python scripts/run_agent.py --template project-manager-v2 --task "Create a PRD" --dry-run

    # Full run (requires ANTHROPIC_API_KEY)
    python scripts/run_agent.py --template project-manager-v2 --task "Create a PRD"

    # With knowledge base
    KNOWLEDGE_BASE_DIR=./shared_knowledge python scripts/run_agent.py --template project-manager-v2 --task "..." --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import signal
import logging
from pathlib import Path

# Ensure src/ is on path for development
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentfactory import (
    AgentFactory,
    ToolRegistry,
    SpawnConfig,
    ReasoningFramework,
)
from agentfactory.templates import load_all_templates
from agentfactory.mcp.manager import ToolServerManager
from agentfactory.mcp.bridge import register_mcp_tools

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── MCP Server Definitions ──────────────────────────────────
# Add new servers here as you build them.

MCP_SERVERS = {
    "calculator": {
        "command": [sys.executable, "-m", "agentfactory.mcp.servers.calculator"],
        "domain_tags": ["math", "finance", "data-science"],
        "id_map": {
            "calculate": "calculator",
            "convert_units": "convert_units",
        },
        "prompt_instructions": {
            "calculator": """## Tool: calculator
Evaluate a mathematical expression.

Parameters:
  - expression (str): Math expression to evaluate.
    Supports: +, -, *, /, **, sqrt(), log(), sin(), cos(), pi, e

Returns: JSON with 'expression' and 'result' (numeric).""",
            "convert_units": """## Tool: convert_units
Convert between common units (length, weight, temperature).

Parameters:
  - value (number): The value to convert
  - from_unit (str): Source unit (km, miles, kg, lb, celsius, fahrenheit, m, ft)
  - to_unit (str): Target unit

Returns: JSON with 'value', 'from', 'to', 'result'.""",
        },
    },
    "knowledge_base": {
        "command": [sys.executable, "-m", "agentfactory.mcp.servers.knowledge_base"],
        "domain_tags": ["general", "business", "development"],
        "id_map": {
            "list_documents": "kb_list",
            "read_document": "kb_read",
            "search_knowledge": "knowledge_base",
        },
        "prompt_instructions": {
            "knowledge_base": """## Tool: knowledge_base
Search the team's knowledge base for lessons learned, SOPs, AARs, and conventions.

Parameters:
  - query (str): Search term or phrase

Returns: Matching excerpts with filenames and context.

Use this to check if there are existing lessons learned or SOPs
before starting a new task. The knowledge base contains AARs
(After Action Reviews), SOPs, and team conventions.""",
            "kb_list": """## Tool: kb_list
List all available documents in the knowledge base.

Returns: Document filenames with summaries.""",
            "kb_read": """## Tool: kb_read
Read the full contents of a specific knowledge base document.

Parameters:
  - filename (str): Document filename (use kb_list to discover available files)

Returns: Full document content.""",
        },
    },
}


def start_mcp_servers(manager: ToolServerManager, server_ids: list[str] | None = None):
    """Start MCP tool servers and return discovered tools."""
    server_ids = server_ids or list(MCP_SERVERS.keys())
    all_tools = {}

    for sid in server_ids:
        config = MCP_SERVERS.get(sid)
        if not config:
            logger.warning(f"Unknown MCP server: {sid}")
            continue

        # Set PYTHONPATH for subprocess
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        manager.register_server(sid, config["command"], env=env)

        try:
            tools = manager.start(sid)
            tool_names = [t["name"] for t in tools]
            logger.info(f"  [{sid}] started: {tool_names}")
            all_tools[sid] = tools
        except Exception as e:
            logger.error(f"  [{sid}] failed to start: {e}")

    return all_tools


def main():
    parser = argparse.ArgumentParser(
        description="Run an Agent Factory agent with live MCP tools.",
    )
    parser.add_argument("--list", action="store_true", help="List available templates")
    parser.add_argument("--template", "-t", type=str, help="Template ID to spawn")
    parser.add_argument("--task", type=str, help="Task description for the agent")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model override")
    parser.add_argument("--reasoning", "-r", choices=["none", "standard", "sirp"], default="none")
    parser.add_argument("--dry-run", action="store_true", help="Compose prompt only, no LLM")
    parser.add_argument("--servers", type=str, nargs="*", default=None, help="Which MCP servers to start")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load templates
    prompt_registry = load_all_templates()

    if args.list:
        print(f"\nAvailable templates ({prompt_registry.count}):\n")
        templates = prompt_registry.search()
        by_domain: dict[str, list] = {}
        for t in templates:
            for d in t.domain_tags:
                by_domain.setdefault(d, []).append(t)

        seen = set()
        for domain in sorted(by_domain.keys()):
            print(f"  [{domain}]")
            for t in by_domain[domain]:
                if t.id not in seen:
                    tools = ", ".join(t.required_tools) if t.required_tools else "none"
                    print(f"    {t.id:<35} {t.name} (tools: {tools})")
                    seen.add(t.id)
            print()
        return

    if not args.template or not args.task:
        parser.error("--template and --task are required (or use --list)")

    # Start MCP servers
    manager = ToolServerManager()

    def shutdown(sig, frame):
        print("\nShutting down MCP servers...")
        manager.stop_all()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    print("Starting MCP tool servers...")
    discovered = start_mcp_servers(manager, args.servers)

    # Register tools in factory
    tool_registry = ToolRegistry()
    server_configs = {sid: MCP_SERVERS[sid] for sid in discovered}
    registered = register_mcp_tools(manager, tool_registry, server_configs)

    print(f"Registered {len(registered)} live tools: {registered}\n")

    # Create factory
    factory = AgentFactory(
        prompt_registry=prompt_registry,
        tool_registry=tool_registry,
        default_model=args.model or "anthropic:claude-sonnet-4-5-20250929",
    )

    reasoning = {
        "none": ReasoningFramework.NONE,
        "standard": ReasoningFramework.STANDARD,
        "sirp": ReasoningFramework.SIRP,
    }[args.reasoning]

    config = SpawnConfig(
        template_id=args.template,
        reasoning_framework=reasoning,
        task_context=args.task,
        model=args.model,
    )

    # Dry run
    if args.dry_run:
        template = prompt_registry.get(config.template_id)
        if not template:
            print(f"Error: Template '{config.template_id}' not found.")
            manager.stop_all()
            return

        tool_ids = template.required_tools + template.optional_tools
        _, resolved_ids, missing_ids = tool_registry.resolve(tool_ids)
        composed = factory.composer.compose(template, config)

        print(f"Template: {template.id} ({template.name})")
        print(f"  Tools attached: {resolved_ids}")
        print(f"  Tools missing:  {missing_ids}")
        print(f"  Prompt length:  {len(composed)} chars")
        print(f"\n{'='*60}")
        print("  DRY RUN — Composed System Prompt")
        print(f"{'='*60}\n")
        print(composed)
        print(f"\n{'='*60}")
        manager.stop_all()
        return

    # Full run
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nANTHROPIC_API_KEY not set. Use --dry-run to see the composed prompt.")
        manager.stop_all()
        return

    try:
        result = factory.create(config)
        print(f"Agent spawned: {result.agent_id}")
        print(f"  Tools: {result.tools_attached}")
        print(f"\nInvoking with task: {args.task}\n")
        print("=" * 60)

        from langchain_core.messages import HumanMessage
        response = result.agent.invoke({"messages": [HumanMessage(content=args.task)]})
        messages = response.get("messages", [])
        if messages:
            print(messages[-1].content)
        else:
            print("Agent completed but produced no output.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("=" * 60)
        manager.stop_all()


if __name__ == "__main__":
    main()
