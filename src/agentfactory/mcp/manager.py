"""
Tool Server Manager — launches and manages MCP tool server processes.

Usage:
    manager = ToolServerManager()
    manager.register_server("calculator", ["python", "-m", "agentfactory.mcp.servers.calculator"])
    tools = manager.start("calculator")
    result = manager.call("calculator", "calculate", {"expression": "2 + 2"})
    manager.stop_all()
"""

from __future__ import annotations

import logging
from typing import Any

from .transport import StdioTransport, JsonRpcRequest

logger = logging.getLogger(__name__)


class ToolServerManager:
    """
    Manages the lifecycle of MCP tool server processes.

    Responsibilities:
    - Launch tool servers as subprocesses (stdio transport)
    - Route tool calls to the correct server
    - Graceful shutdown
    """

    def __init__(self):
        self._servers: dict[str, dict] = {}

    def register_server(
        self,
        server_id: str,
        command: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        """Register a tool server (does not start it yet)."""
        self._servers[server_id] = {
            "command": command,
            "transport": None,
            "env": env,
            "tools": [],
        }
        logger.info(f"Registered server: {server_id}")

    def start(self, server_id: str) -> list[dict]:
        """Start a tool server and discover its tools."""
        server = self._servers.get(server_id)
        if not server:
            raise ValueError(f"Unknown server: {server_id}")

        transport = StdioTransport(server["command"], server.get("env"))
        transport.start()
        server["transport"] = transport

        # Discover tools
        request = JsonRpcRequest(
            method="tools/list", params={}, id=transport.next_id()
        )
        response = transport.send(request)

        if response.is_error:
            raise RuntimeError(
                f"Failed to discover tools from {server_id}: {response.error}"
            )

        server["tools"] = response.result or []
        tool_names = [t["name"] for t in server["tools"]]
        logger.info(f"Started {server_id}: tools={tool_names}")
        return server["tools"]

    def start_all(self) -> dict[str, list[dict]]:
        """Start all registered servers."""
        results = {}
        for server_id in self._servers:
            try:
                results[server_id] = self.start(server_id)
            except Exception as e:
                logger.error(f"Failed to start {server_id}: {e}")
                results[server_id] = []
        return results

    def stop(self, server_id: str) -> None:
        server = self._servers.get(server_id)
        if server and server["transport"]:
            server["transport"].stop()
            server["transport"] = None
            logger.info(f"Stopped {server_id}")

    def stop_all(self) -> None:
        for server_id in list(self._servers.keys()):
            self.stop(server_id)

    def call(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call a tool on a specific server."""
        server = self._servers.get(server_id)
        if not server:
            raise ValueError(f"Unknown server: {server_id}")

        transport = server.get("transport")
        if not transport or not transport.is_alive():
            raise RuntimeError(f"Server {server_id} is not running.")

        request = JsonRpcRequest(
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
            id=transport.next_id(),
        )
        response = transport.send(request)

        if response.is_error:
            raise RuntimeError(
                f"Tool call failed ({server_id}/{tool_name}): {response.error}"
            )
        return response.result

    def list_tools(self, server_id: str) -> list[dict]:
        server = self._servers.get(server_id)
        return server["tools"] if server else []

    def list_servers(self) -> dict[str, bool]:
        return {
            sid: (s["transport"] is not None and s["transport"].is_alive())
            for sid, s in self._servers.items()
        }

    def is_running(self, server_id: str) -> bool:
        server = self._servers.get(server_id)
        return (
            server is not None
            and server["transport"] is not None
            and server["transport"].is_alive()
        )
