"""
MCP tool infrastructure.

Provides:
- StdioToolServer / ToolHandler — framework for building tool servers
- StdioTransport — client-side transport (subprocess + stdio pipes)
- ToolServerManager — lifecycle management for tool server processes
- Bridge utilities — MCP tools -> LangChain StructuredTool wrappers
"""

from .server import StdioToolServer, ToolHandler
from .transport import StdioTransport
from .manager import ToolServerManager
from .bridge import mcp_to_langchain_tool, register_mcp_tools

__all__ = [
    "StdioToolServer",
    "ToolHandler",
    "StdioTransport",
    "ToolServerManager",
    "mcp_to_langchain_tool",
    "register_mcp_tools",
]
