"""
Bridge between MCP tool servers and LangChain/Agent Factory.

Converts MCP tool servers into LangChain StructuredTools
that can be registered in the Agent Factory's ToolRegistry.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from .manager import ToolServerManager
from ..models import ToolRegistryEntry


def mcp_to_langchain_tool(
    manager: ToolServerManager,
    server_id: str,
    tool_name: str,
    description_override: str | None = None,
) -> StructuredTool:
    """
    Create a LangChain StructuredTool that wraps an MCP tool server call.

    When invoked by an agent, sends a JSON-RPC request to the
    specified tool server via stdio and returns the result.
    """
    tools = manager.list_tools(server_id)
    tool_schema = next((t for t in tools if t["name"] == tool_name), None)

    description = (
        description_override
        or (tool_schema.get("description", tool_name) if tool_schema else f"MCP tool: {server_id}/{tool_name}")
    )

    def _call_mcp(**kwargs: Any) -> str:
        try:
            result = manager.call(server_id, tool_name, kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error calling {server_id}/{tool_name}: {e}"

    return StructuredTool.from_function(
        func=_call_mcp,
        name=tool_name,
        description=description,
    )


def register_mcp_tools(
    manager: ToolServerManager,
    tool_registry: Any,  # ToolRegistry (avoid circular import)
    server_configs: dict[str, dict] | None = None,
) -> list[str]:
    """
    Discover all tools from running MCP servers and register
    them in the Agent Factory's ToolRegistry.

    Args:
        manager: ToolServerManager with running servers
        tool_registry: An Agent Factory ToolRegistry instance
        server_configs: Optional {server_id: {domain_tags, prompt_instructions, id_map}}

    Returns:
        List of registered tool IDs.
    """
    server_configs = server_configs or {}
    registered = []

    for server_id, running in manager.list_servers().items():
        if not running:
            continue

        config = server_configs.get(server_id, {})
        domain_tags = config.get("domain_tags", [])
        instructions_map = config.get("prompt_instructions", {})
        id_map = config.get("id_map", {})

        for tool_schema in manager.list_tools(server_id):
            mcp_name = tool_schema["name"]
            factory_id = id_map.get(mcp_name, mcp_name)

            lc_tool = mcp_to_langchain_tool(manager, server_id, mcp_name)

            instructions = instructions_map.get(factory_id, "")
            if not instructions:
                instructions = instructions_map.get(mcp_name, "")
            if not instructions:
                instructions = _auto_prompt_instructions(tool_schema)

            entry = ToolRegistryEntry(
                id=factory_id,
                name=lc_tool.name,
                description=lc_tool.description,
                tool_type="mcp_server",
                prompt_instructions=instructions,
                tool_instance=lc_tool,
                domain_tags=domain_tags,
            )
            tool_registry.register(entry)
            registered.append(factory_id)

    return registered


def _auto_prompt_instructions(schema: dict) -> str:
    """Generate prompt instructions from an MCP tool schema."""
    name = schema.get("name", "unknown")
    description = schema.get("description", "")
    params = schema.get("parameters", {}).get("properties", {})

    lines = [f"## Tool: {name}", description, ""]
    if params:
        lines.append("Parameters:")
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            lines.append(f"  - {pname} ({ptype}): {pdesc}")

    return "\n".join(lines)
