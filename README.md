# Agent Factory

A Python package that lets agents create specialized agents with tools.

The core idea: an orchestrator agent (Clara) recognizes that a task needs
a specialist, searches the template registry, spawns an agent with the
right tools via MCP, and gets results back.

```
┌─────────────────────────────────────────────────────────┐
│  Clara (orchestrator)                                    │
│                                                          │
│  "I need a PM agent to write a PRD"                      │
│       │                                                  │
│       ▼                                                  │
│  search_prompt_registry(query="project manager")         │
│       │                                                  │
│       ▼                                                  │
│  spawn_agent(template_id="project-manager-v2",           │
│              task="Write a PRD for Agent Factory v1")    │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────────────────────────────┐                    │
│  │  Agent Factory                    │                   │
│  │  1. Look up template              │                   │
│  │  2. Start MCP tool servers        │                   │
│  │  3. Compose system prompt         │                   │
│  │  4. Build LangGraph agent         │                   │
│  │  5. Invoke with task              │                   │
│  │  6. Return result                 │                   │
│  └──────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

### As a tool for Clara (or any LangChain agent)

```python
from agentfactory import AgentFactory, SpawnConfig
from agentfactory.templates import load_all_templates

# Load all templates
registry = load_all_templates()

# Create factory
factory = AgentFactory(prompt_registry=registry)

# Give these tools to your orchestrator agent
spawn_tool = factory.as_tool()           # spawn_agent(template_id, task)
search_tool = factory.search_registry_tool()  # search_prompt_registry(query)
```

### Standalone usage

```python
from agentfactory import AgentFactory, SpawnConfig
from agentfactory.templates import load_all_templates

registry = load_all_templates()
factory = AgentFactory(prompt_registry=registry)

result = factory.create(SpawnConfig(
    template_id="project-manager-v2",
    task_context="Write a PRD for the Agent Factory v1 release",
))

print(f"Agent: {result.agent_id}")
print(f"Tools: {result.tools_attached}")
# result.agent.invoke({"messages": [HumanMessage(content="...")]})
```

### CLI (dry run)

```bash
# List templates
python scripts/run_agent.py --list

# Compose prompt with live MCP tools (no LLM needed)
python scripts/run_agent.py \
  --template project-manager-v2 \
  --task "Write a PRD for Agent Factory" \
  --dry-run

# Full run (requires ANTHROPIC_API_KEY)
python scripts/run_agent.py \
  --template project-manager-v2 \
  --task "Write a PRD for Agent Factory"
```

## Architecture

### Package Structure

```
src/agentfactory/
├── __init__.py          # Public API: AgentFactory, SpawnConfig, etc.
├── models.py            # Data models and enums
├── registry.py          # PromptRegistry + ToolRegistry
├── composer.py          # Prompt composition (tool block injection)
├── factory.py           # AgentFactory (the core engine)
├── templates/
│   ├── __init__.py      # load_all_templates()
│   ├── registry.yaml    # 18 YAML templates
│   └── expanded.py      # 12 fully-expanded v2 templates
└── mcp/
    ├── __init__.py
    ├── server.py         # StdioToolServer + ToolHandler base
    ├── transport.py      # StdioTransport (JSON-RPC client)
    ├── manager.py        # ToolServerManager (process lifecycle)
    ├── bridge.py         # MCP -> LangChain tool bridge
    └── servers/
        ├── calculator.py      # Math + unit conversion
        └── knowledge_base.py  # Read/search markdown knowledge base
```

### How it works

1. **Templates** define agent personas: identity, reasoning protocol, tool dependencies, guardrails
2. **Tool Registry** catalogs available tools with prompt instructions for injection
3. **Prompt Composer** replaces `{{TOOL_BLOCK:tool_id}}` placeholders with real tool descriptions
4. **MCP Servers** are stdio subprocesses that provide tools via JSON-RPC
5. **Bridge** wraps MCP tools as LangChain StructuredTools for the agent runtime
6. **Factory** assembles everything and builds a LangGraph ReAct agent

### MCP Tool Servers

Tool servers are standalone processes that communicate via JSON-RPC over stdio.
Build new ones by subclassing `ToolHandler`:

```python
from agentfactory.mcp import StdioToolServer, ToolHandler

class DrawDiagram(ToolHandler):
    name = "draw_diagram"
    description = "Generate a draw.io diagram from a description"
    parameters = {
        "description": {"type": "string", "description": "What to draw"},
        "format": {"type": "string", "description": "Output format: drawio|svg|png"},
    }

    def handle(self, params):
        # Your implementation here
        return {"diagram_path": "output.drawio", "format": params["format"]}

if __name__ == "__main__":
    server = StdioToolServer()
    server.register(DrawDiagram())
    server.run()
```

### Knowledge Base

The `knowledge_base` MCP server gives agents access to your local markdown
knowledge base (AARs, SOPs, conventions, lessons learned).

Configure via environment variable:
```bash
export KNOWLEDGE_BASE_DIR=/path/to/your/knowledge/base
```

Or symlink a `shared_knowledge/` directory in the project root.

Tools provided:
- `list_documents` — List all available KB documents
- `read_document` — Read a specific document
- `search_knowledge` — Search across all documents

## Templates

12 fully-expanded templates with 5-layer prompt structure:

| Template | Domain | Required Tools |
|----------|--------|----------------|
| `financial-analyst-v2` | finance | market_data_api |
| `code-review-agent-v2` | development | code_executor |
| `data-scientist-v2` | data-science | code_executor |
| `senior-system-architect-v2` | architecture | (none) |
| `devops-engineer-v2` | devops | code_executor |
| `llm-researcher-v2` | ai-ml | web_search |
| `deep-research-protocol-v2` | research | web_search |
| `image-gen-director-v2` | creative | image_gen |
| `project-manager-v2` | business | (none) |
| `cybersecurity-specialist-v2` | security | web_search |
| `research-paper-evaluator-v2` | research | (none) |
| `stock-market-analyst-v2` | finance | market_data_api |

Plus 6 stub templates ready for expansion.

## Tests

```bash
pytest tests/ -v
```

23 tests covering:
- Registry loading and search
- Prompt composition with tool injection
- MCP server communication (calculator + knowledge base)
- Path traversal security on knowledge base
- Factory integration (prompt composition pipeline)

## Development Roadmap

- [ ] Add `deepagents` runtime support (pip install agentfactory[full])
- [ ] Migrate MCP servers to official `mcp` Python SDK (FastMCP)
- [ ] Build web_search MCP server (Tavily or SerpAPI)
- [ ] Build file_system MCP server (workspace access)
- [ ] Build draw.io MCP server (diagram generation)
- [ ] Add DoDAF/SysML/TOGAF reference tools for arch agents
- [ ] Expand 6 stub templates to full 5-layer structure
- [ ] Deploy MCP servers to Cloud Run via Artifact Registry
- [ ] Add Apigee/Cloud Endpoints gateway for network access

## License

MIT
