"""
Knowledge Base MCP tool server.

Provides agents access to a local knowledge base of markdown documents
(AARs, SOPs, conventions, lessons learned).

The knowledge base directory is configured via:
  - KNOWLEDGE_BASE_DIR environment variable
  - Or defaults to ./shared_knowledge/ relative to the project root

Run as:
    python -m agentfactory.mcp.servers.knowledge_base

Documents are expected to be .md files. The server provides three tools:
  - list_documents: List all available knowledge base documents
  - read_document: Read the full contents of a specific document
  - search_knowledge: Search across all documents for a keyword/phrase
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

from agentfactory.mcp.server import StdioToolServer, ToolHandler

logger = logging.getLogger(__name__)


def _get_kb_dir() -> Path:
    """Resolve the knowledge base directory."""
    env_dir = os.environ.get("KNOWLEDGE_BASE_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    # Default: shared_knowledge/ in project root (or symlinked)
    project_root = Path(__file__).parent.parent.parent.parent.parent
    candidates = [
        project_root / "shared_knowledge",
        project_root / "knowledge",
        Path.home() / "Desktop" / "Development" / "agent-knowledge",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    # Fallback: create a default location
    default = project_root / "shared_knowledge"
    default.mkdir(exist_ok=True)
    return default


class ListDocumentsTool(ToolHandler):
    name = "list_documents"
    description = (
        "List all available documents in the knowledge base. "
        "Returns filenames and first-line summaries."
    )
    parameters = {}

    def handle(self, params: dict[str, Any]) -> dict:
        kb_dir = _get_kb_dir()
        if not kb_dir.exists():
            return {"error": f"Knowledge base directory not found: {kb_dir}"}

        docs = []
        for md_file in sorted(kb_dir.glob("**/*.md")):
            rel_path = md_file.relative_to(kb_dir)
            # Read first non-empty line as summary
            summary = ""
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            summary = line[:120]
                            break
                        elif line.startswith("#"):
                            summary = line.lstrip("#").strip()[:120]
                            break
            except Exception:
                summary = "(could not read)"

            docs.append({
                "filename": str(rel_path),
                "summary": summary,
                "size_bytes": md_file.stat().st_size,
            })

        return {
            "knowledge_base_dir": str(kb_dir),
            "document_count": len(docs),
            "documents": docs,
        }


class ReadDocumentTool(ToolHandler):
    name = "read_document"
    description = (
        "Read the full contents of a specific knowledge base document. "
        "Use list_documents first to see available files."
    )
    parameters = {
        "filename": {
            "type": "string",
            "description": "Filename or relative path of the document to read",
        },
    }

    def handle(self, params: dict[str, Any]) -> dict:
        filename = params.get("filename", "")
        if not filename:
            return {"error": "No filename provided"}

        kb_dir = _get_kb_dir()
        file_path = (kb_dir / filename).resolve()

        # Security: ensure we stay within the knowledge base directory
        if not str(file_path).startswith(str(kb_dir)):
            return {"error": "Access denied: path is outside the knowledge base directory"}

        if not file_path.exists():
            return {"error": f"Document not found: {filename}"}

        try:
            content = file_path.read_text(encoding="utf-8")
            return {
                "filename": filename,
                "content": content,
                "size_bytes": len(content),
            }
        except Exception as e:
            return {"error": f"Failed to read {filename}: {e}"}


class SearchKnowledgeTool(ToolHandler):
    name = "search_knowledge"
    description = (
        "Search across all knowledge base documents for a keyword or phrase. "
        "Returns matching excerpts with context."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "Search term or phrase to find",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of matches to return (default: 10)",
        },
    }

    def handle(self, params: dict[str, Any]) -> dict:
        query = params.get("query", "").lower()
        if not query:
            return {"error": "No query provided"}

        max_results = int(params.get("max_results", 10))
        kb_dir = _get_kb_dir()

        if not kb_dir.exists():
            return {"error": f"Knowledge base directory not found: {kb_dir}"}

        matches = []
        for md_file in sorted(kb_dir.glob("**/*.md")):
            rel_path = str(md_file.relative_to(kb_dir))
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            lines = content.split("\n")
            for i, line in enumerate(lines):
                if query in line.lower():
                    # Extract context: 2 lines before and after
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    excerpt = "\n".join(lines[start:end])

                    matches.append({
                        "filename": rel_path,
                        "line_number": i + 1,
                        "excerpt": excerpt[:500],
                    })

                    if len(matches) >= max_results:
                        break

            if len(matches) >= max_results:
                break

        return {
            "query": query,
            "match_count": len(matches),
            "matches": matches,
        }


def main():
    server = StdioToolServer()
    server.register(ListDocumentsTool())
    server.register(ReadDocumentTool())
    server.register(SearchKnowledgeTool())
    server.run()


if __name__ == "__main__":
    main()
