"""
MCP Tool Server base class.

A tool server is a standalone process that:
1. Reads JSON-RPC requests from stdin
2. Dispatches to registered ToolHandlers
3. Writes JSON-RPC responses to stdout

Protocol: JSON-RPC 2.0, one message per line (MCP stdio transport).

Usage:
    from agentfactory.mcp import StdioToolServer, ToolHandler

    class MyTool(ToolHandler):
        name = "my_tool"
        description = "Does something useful"
        parameters = {"input": {"type": "string", "description": "The input"}}

        def handle(self, params: dict) -> dict:
            return {"result": f"processed: {params['input']}"}

    if __name__ == "__main__":
        server = StdioToolServer()
        server.register(MyTool())
        server.run()
"""

from __future__ import annotations

import json
import sys
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class ToolHandler(ABC):
    """Base class for a tool implementation."""

    name: str = ""
    description: str = ""
    parameters: dict[str, dict] = {}

    @abstractmethod
    def handle(self, params: dict[str, Any]) -> Any:
        """Execute the tool with the given parameters."""
        ...

    def get_schema(self) -> dict:
        """Return the tool schema for discovery."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
            },
        }


class StdioToolServer:
    """
    JSON-RPC tool server communicating via stdin/stdout.

    Supported methods:
        - "tools/list"  -> returns registered tool schemas
        - "tools/call"  -> calls a tool by name with params
        - "ping"        -> health check
    """

    def __init__(self):
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, handler: ToolHandler) -> None:
        if not handler.name:
            raise ValueError(f"ToolHandler {handler.__class__.__name__} has no name")
        self._handlers[handler.name] = handler

    def run(self) -> None:
        """Main loop: read stdin, dispatch, write stdout. Blocks until stdin closes."""
        logger.info(
            f"Tool server starting with {len(self._handlers)} tools: "
            f"{list(self._handlers.keys())}"
        )

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                self._write_error(None, -32700, f"Parse error: {e}")
                continue

            request_id = request.get("id")
            method = request.get("method", "")
            params = request.get("params", {})

            try:
                result = self._dispatch(method, params)
                self._write_result(request_id, result)
            except Exception as e:
                self._write_error(request_id, -32603, str(e))

    def _dispatch(self, method: str, params: dict) -> Any:
        if method == "ping":
            return {"status": "ok", "tools": list(self._handlers.keys())}

        if method == "tools/list":
            return [h.get_schema() for h in self._handlers.values()]

        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_params = params.get("arguments", {})

            handler = self._handlers.get(tool_name)
            if not handler:
                raise ValueError(
                    f"Unknown tool: '{tool_name}'. "
                    f"Available: {list(self._handlers.keys())}"
                )
            return handler.handle(tool_params)

        raise ValueError(f"Unknown method: '{method}'")

    def _write_result(self, request_id: Any, result: Any) -> None:
        response = json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "result": result}
        )
        sys.stdout.write(response + "\n")
        sys.stdout.flush()

    def _write_error(self, request_id: Any, code: int, message: str) -> None:
        response = json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
        )
        sys.stdout.write(response + "\n")
        sys.stdout.flush()
