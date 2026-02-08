#!/usr/bin/env python3
"""
Run Agent — end-to-end: MCP servers -> Factory -> Live Agent.

Usage:
    # List available templates
    python scripts/run_agent.py --list

    # From a YAML spec (dry run)
    python scripts/run_agent.py --spec specs/pm-agent.yaml --dry-run

    # From a YAML spec (full run, requires ANTHROPIC_API_KEY)
    python scripts/run_agent.py --spec specs/pm-agent.yaml

    # Classic mode: template + task (dry run)
    python scripts/run_agent.py --template project-manager-v2 --task "Create a PRD" --dry-run

    # With knowledge base
    KNOWLEDGE_BASE_DIR=./shared_knowledge python scripts/run_agent.py --spec specs/pm-agent.yaml --dry-run
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
    AgentSpec,
    PromptTemplate,
    ToolRegistry,
    SpawnConfig,
    ReasoningFramework,
    default_server_registry,
)
from agentfactory.templates import load_all_templates
from agentfactory.mcp.manager import ToolServerManager
from agentfactory.mcp.bridge import register_mcp_tools

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def start_mcp_servers(manager, server_registry, server_names=None):
    """Start MCP tool servers and return discovered tools."""
    server_names = server_names or server_registry.list_all()
    all_tools = {}

    for name in server_names:
        server_def = server_registry.get(name)
        if not server_def:
            logger.warning(f"Unknown MCP server: {name}")
            continue

        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        env.update(server_def.env)
        manager.register_server(name, server_def.command, env=env)

        try:
            tools = manager.start(name)
            tool_names = [t["name"] for t in tools]
            logger.info(f"  [{name}] started: {tool_names}")
            all_tools[name] = tools
        except Exception as e:
            logger.error(f"  [{name}] failed to start: {e}")

    return all_tools


def run_spec_mode(args, prompt_registry):
    """Run from a YAML agent spec."""
    spec = AgentSpec.from_yaml(args.spec)

    # Override task from CLI if provided
    task = args.task or spec.task
    if not task and not args.dry_run:
        print("Error: No task specified. Add 'task:' to the spec or use --task.")
        return

    server_registry = default_server_registry()

    # Resolve the template
    if spec.template_is_file:
        template = PromptTemplate.from_markdown(spec.template, name=spec.name)
        prompt_registry._templates[template.id] = template
    else:
        template = prompt_registry.get(spec.template)
        if not template:
            print(f"Error: Template '{spec.template}' not found.")
            return

    # Start MCP servers declared in spec
    manager = ToolServerManager()

    def shutdown(sig, frame):
        print("\nShutting down MCP servers...")
        manager.stop_all()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    if spec.mcp_servers:
        print(f"Starting MCP servers: {spec.mcp_servers}")
        env_extra = {}
        if spec.knowledge_paths:
            env_extra["KNOWLEDGE_PATHS"] = ",".join(spec.knowledge_paths)
        env_extra.update(spec.env)

        # Temporarily apply extra env for server startup
        for k, v in env_extra.items():
            os.environ[k] = v

        discovered = start_mcp_servers(manager, server_registry, spec.mcp_servers)

        # Register tools
        tool_registry = ToolRegistry()
        bridge_configs = server_registry.to_bridge_configs(list(discovered.keys()))
        registered = register_mcp_tools(manager, tool_registry, bridge_configs)
        print(f"Registered {len(registered)} live tools: {registered}\n")
    else:
        tool_registry = ToolRegistry()

    model = args.model or spec.model or "anthropic:claude-sonnet-4-5-20250929"
    factory = AgentFactory(
        prompt_registry=prompt_registry,
        tool_registry=tool_registry,
        server_registry=server_registry,
        default_model=model,
    )

    reasoning = ReasoningFramework(spec.reasoning) if spec.reasoning else ReasoningFramework.NONE

    config = SpawnConfig(
        template_id=template.id,
        reasoning_framework=reasoning,
        task_context=task,
        model=args.model or spec.model,
        tool_overrides=spec.tool_overrides,
        max_iterations=spec.max_iterations,
    )

    if args.dry_run:
        tool_ids = template.required_tools + template.optional_tools
        _, resolved_ids, missing_ids = tool_registry.resolve(tool_ids)
        composed = factory.composer.compose(template, config)

        print(f"Spec: {args.spec}")
        print(f"Template: {template.id} ({template.name})")
        print(f"  MCP servers: {spec.mcp_servers}")
        print(f"  Knowledge paths: {spec.knowledge_paths or '(all)'}")
        print(f"  Tools attached: {resolved_ids}")
        print(f"  Tools missing:  {missing_ids}")
        print(f"  Reasoning: {spec.reasoning}")
        print(f"  Prompt length:  {len(composed)} chars")
        print(f"\n{'='*60}")
        print("  DRY RUN — Composed System Prompt")
        print(f"{'='*60}\n")
        print(composed)
        print(f"\n{'='*60}")
        manager.stop_all()
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nANTHROPIC_API_KEY not set. Use --dry-run to see the composed prompt.")
        manager.stop_all()
        return

    try:
        result = factory.create(config)
        print(f"Agent spawned: {result.agent_id}")
        print(f"  Tools: {result.tools_attached}")
        print(f"\nInvoking with task: {task}\n")
        print("=" * 60)

        from langchain_core.messages import HumanMessage
        response = result.agent.invoke({"messages": [HumanMessage(content=task)]})
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


def run_classic_mode(args, prompt_registry):
    """Run with --template and --task flags (original mode)."""
    if not args.template or not args.task:
        print("Error: --template and --task are required (or use --spec or --list)")
        return

    server_registry = default_server_registry()
    manager = ToolServerManager()

    def shutdown(sig, frame):
        print("\nShutting down MCP servers...")
        manager.stop_all()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    print("Starting MCP tool servers...")
    discovered = start_mcp_servers(manager, server_registry, args.servers)

    tool_registry = ToolRegistry()
    bridge_configs = server_registry.to_bridge_configs(list(discovered.keys()))
    registered = register_mcp_tools(manager, tool_registry, bridge_configs)
    print(f"Registered {len(registered)} live tools: {registered}\n")

    factory = AgentFactory(
        prompt_registry=prompt_registry,
        tool_registry=tool_registry,
        server_registry=server_registry,
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


def main():
    parser = argparse.ArgumentParser(
        description="Run an Agent Factory agent with live MCP tools.",
    )
    parser.add_argument("--list", action="store_true", help="List available templates")
    parser.add_argument("--spec", "-s", type=str, help="Path to a YAML agent spec file")
    parser.add_argument("--template", "-t", type=str, help="Template ID to spawn")
    parser.add_argument("--task", type=str, help="Task description for the agent")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model override")
    parser.add_argument("--reasoning", "-r", choices=["none", "standard", "sirp"], default="none")
    parser.add_argument("--dry-run", action="store_true", help="Compose prompt only, no LLM")
    parser.add_argument("--servers", type=str, nargs="*", default=None, help="Which MCP servers to start (classic mode)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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

        sr = default_server_registry()
        print(f"Available MCP servers ({sr.count}):\n")
        for name in sr.list_all():
            sdef = sr.get(name)
            print(f"  {name:<25} domains: {sdef.domain_tags}")
        print()
        return

    if args.spec:
        run_spec_mode(args, prompt_registry)
    elif args.template or args.task:
        run_classic_mode(args, prompt_registry)
    else:
        parser.error("Use --spec <file>, --template + --task, or --list")


if __name__ == "__main__":
    main()
