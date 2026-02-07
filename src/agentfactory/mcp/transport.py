"""
Transport layer for MCP tool communication.

Implements StdioTransport: JSON-RPC over stdin/stdout pipes
to a subprocess. This is MCP's native local transport.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 request."""
    method: str
    params: dict[str, Any]
    id: int | str

    def to_json(self) -> str:
        return json.dumps({
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id,
        })


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 response."""
    id: int | str
    result: Any = None
    error: dict | None = None

    @classmethod
    def from_json(cls, data: str) -> JsonRpcResponse:
        parsed = json.loads(data)
        return cls(
            id=parsed.get("id"),
            result=parsed.get("result"),
            error=parsed.get("error"),
        )

    @property
    def is_error(self) -> bool:
        return self.error is not None


class StdioTransport:
    """
    JSON-RPC over stdin/stdout pipes to a subprocess.

    The tool server runs as a child process. We write JSON-RPC
    requests to its stdin and read responses from its stdout.
    One line = one message.
    """

    def __init__(self, command: list[str], env: dict[str, str] | None = None):
        self.command = command
        self.env = env
        self._process: subprocess.Popen | None = None
        self._request_id = 0

    def start(self) -> None:
        """Launch the tool server subprocess."""
        if self._process and self._process.poll() is None:
            logger.warning("Transport already running, stopping first")
            self.stop()

        logger.info(f"Starting stdio transport: {' '.join(self.command)}")
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def stop(self) -> None:
        """Terminate the tool server subprocess."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            logger.info("Stdio transport stopped")

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def send(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Send JSON-RPC request via stdin, read response from stdout."""
        if not self.is_alive():
            raise RuntimeError("Transport not running. Call start() first.")

        line = request.to_json() + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        if not response_line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"Tool server process died. stderr: {stderr[:500]}")

        return JsonRpcResponse.from_json(response_line.strip())

    def next_id(self) -> int:
        self._request_id += 1
        return self._request_id
