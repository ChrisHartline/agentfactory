# Agent Factory

A meta-agent system that enables agents to dynamically create specialized agents.

```
┌──────────────────────────────────────────────────────────┐
│  Agent Factory                                           │
│  ┌────────────────┐  ┌────────────────┐                  │
│  │ Prompt Registry │  │  Tool Registry  │                 │
│  │  (what to be)   │  │  (what to use)  │                │
│  └───────┬────────┘  └───────┬─────────┘                 │
│          │                   │                            │
│          ▼                   ▼                            │
│  ┌─────────────────────────────────────┐                 │
│  │         Prompt Composer              │                 │
│  │  Identity + Tools + Reasoning +      │                 │
│  │  Contract + Guardrails               │                 │
│  └───────────────┬─────────────────────┘                 │
│                  │                                        │
│                  ▼                                        │
│  ┌─────────────────────────────────────┐                 │
│  │     Agent Runtime                    │                 │
│  │  (deepagents or langgraph fallback)  │                 │
│  └─────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────┘
```

## Features

- **Template-based agent creation** — Searchable registry of agent personas and tasks
- **Dynamic tool injection** — `{{TOOL_BLOCK}}` placeholders replaced at spawn time
- **Dual runtime support** — Uses deepagents (full features) or falls back to LangGraph
- **Meta-reasoning injection** — SIRP protocol or standard reflection
- **Hierarchical spawning** — Agents can spawn sub-agents with depth limits
- **Genealogy tracking** — Full spawn chain for observability

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone <repo-url>
cd agentfactory

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

```python
from agent_factory import AgentFactory, PromptRegistry, ToolRegistry
from expanded_system_prompts import EXPANDED_TEMPLATES

# Initialize registries
prompt_registry = PromptRegistry()
prompt_registry.load_from_yaml('prompt_registry.yaml')
prompt_registry.load_from_dict(EXPANDED_TEMPLATES)  # Override with expanded versions

tool_registry = ToolRegistry()
# Register your tools here (see Tool Registry section)

# Create factory
factory = AgentFactory(
    prompt_registry=prompt_registry,
    tool_registry=tool_registry,
)

print(f"Runtime: {factory.runtime}")  # "deepagents" or "langgraph"

# Search for templates
results = prompt_registry.search(domain_tags=["finance"])
for t in results:
    print(f"  {t.id}: {t.name}")

# Spawn an agent
from agent_factory import SpawnConfig

result = factory.create(SpawnConfig(
    template_id="financial-analyst-v2",
    task_context="Analyze NVDA for potential entry points.",
))

print(f"Agent ID: {result.agent_id}")
print(f"Tools attached: {result.tools_attached}")
```

## Template Structure

Templates follow the **5-layer prompt pattern**:

### 1. Identity + Mandate

Who the agent is and what it's responsible for.

```yaml
# Identity + Mandate

You are a senior financial analyst specializing in equity research...

Your mandate: Given a ticker, analysis type, and timeframe, produce 
a structured assessment with a clear recommendation and confidence score.
```

### 2. Tool Instructions

Dynamic tool blocks injected at spawn time.

```yaml
# Available Tools

{{TOOL_BLOCK:market_data_api}}
{{TOOL_BLOCK:web_search}}
{{TOOL_BLOCK:calculator}}
```

The factory replaces `{{TOOL_BLOCK:tool_id}}` with actual tool instructions from the Tool Registry.

### 3. Reasoning Protocol

How the agent should think through problems.

```yaml
# Reasoning Protocol

Follow this analytical sequence:

1. DATA GATHERING — Retrieve current price, volume...
2. CONTEXT ASSESSMENT — Check for recent news...
3. PATTERN RECOGNITION — Identify support/resistance...
4. RISK IDENTIFICATION — List top 3-5 risks...
5. SYNTHESIS — Combine findings into recommendation...
```

### 4. Input/Output Contract

What the agent expects and what it produces.

```yaml
# Output Contract

Return a JSON object:
{
  "recommendation": "buy|hold|sell|...",
  "confidence": 0.0-1.0,
  "analysis": "...",
  "key_metrics": {...},
  "risks": [...],
  "sources": [...]
}
```

### 5. Guardrails

Constraints and termination conditions.

```yaml
# Guardrails

- NEVER recommend specific position sizes or dollar amounts
- NEVER claim certainty — always frame as analysis, not advice
- Maximum 10 tool calls per analysis
- If confidence falls below 0.4, return "insufficient_data"
```

## Tool Registry

Register tools for agents to use:

```python
from agent_factory import ToolRegistry, ToolRegistryEntry

tool_registry = ToolRegistry()

# Option 1: Register a LangChain tool
from langchain_community.tools import TavilySearchResults

search_tool = TavilySearchResults()
tool_registry.register_langchain_tool(
    tool_id="web_search",
    tool=search_tool,
    prompt_instructions="Search the web for current information...",
    domain_tags=["general", "research"],
)

# Option 2: Register a function
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

tool_registry.register_function(
    tool_id="calculator",
    func=calculate,
    name="calculator",
    description="Evaluate mathematical expressions",
    prompt_instructions="Use for precise calculations...",
    domain_tags=["math", "finance"],
)

# Option 3: Register an MCP server (manual entry)
tool_registry.register(ToolRegistryEntry(
    id="market_data_api",
    name="Market Data API",
    description="Retrieve stock market data",
    tool_type="mcp_server",
    prompt_instructions="""## Tool: market_data_api
Retrieve real-time and historical stock market data.

Parameters:
  - ticker (str): Stock ticker symbol
  - timeframe (str): "1d", "5d", "1m", "3m", "1y"
  - indicators (list[str]): Technical indicators to compute
...""",
    config={"server": "alpha_vantage"},
    domain_tags=["finance", "trading"],
))
```

## Factory API

### SpawnConfig

Configuration for creating an agent:

```python
from agent_factory import SpawnConfig, ReasoningFramework

config = SpawnConfig(
    template_id="financial-analyst-v2",      # Required: which template
    tool_overrides=["web_search"],           # Optional: override template's tools
    model="anthropic:claude-sonnet-4-5-20250929",  # Optional: override model
    max_iterations=15,                       # Optional: override max iterations
    reasoning_framework=ReasoningFramework.SIRP,  # Optional: inject meta-reasoning
    budget={"max_tool_calls": 10},           # Optional: resource limits
    parent_agent_id="parent-1",              # Optional: for genealogy tracking
    task_context="Additional context...",    # Optional: injected into prompt
)
```

### SpawnResult

What you get back:

```python
result = factory.create(config)

result.agent          # The compiled LangGraph agent
result.agent_id       # Unique ID (e.g., "financial-analyst-v2-1")
result.template_id    # Template used
result.tools_attached # Tools successfully resolved
result.tools_missing  # Tools that weren't available
result.composed_prompt # Final system prompt (for debugging)
result.genealogy      # Spawn chain info
```

### Registry Search

Find the right template:

```python
# Search by domain
results = prompt_registry.search(domain_tags=["finance", "trading"])

# Search by reasoning style
from agent_factory import ReasoningStyle
results = prompt_registry.search(reasoning_style=ReasoningStyle.ANALYTICAL)

# Search by keyword
results = prompt_registry.search(query="security assessment")

# Combined filters
results = prompt_registry.search(
    domain_tags=["development"],
    composable_only=True,
    min_quality=0.7,
)
```

### Factory as Tool

Expose the factory to an orchestrator agent:

```python
# Get tools for an orchestrator to use
spawn_tool = factory.as_tool()
search_tool = factory.search_registry_tool()

# The orchestrator can now:
# 1. Search for the right template
# 2. Spawn specialized agents to handle tasks
```

## Meta-Reasoning Frameworks

### SIRP (Structured Iterative Reasoning Protocol)

Injected when `reasoning_framework=ReasoningFramework.SIRP`:

```
<thinking> tags for exploratory thoughts
<count> tags for step budget tracking
<reflection> tags every 3-4 steps with quality score
<reward> tag for final self-assessment
```

### Standard Reflection

Injected when `reasoning_framework=ReasoningFramework.STANDARD`:

```
After completing analysis, assess:
- Did I address the core question?
- What is my confidence?
- What would I do differently?
```

## Creating New Templates

### YAML Format

Add to `prompt_registry.yaml`:

```yaml
- id: my-new-agent
  name: My New Agent
  version: "1.0.0"
  description: What this agent does
  prompt_type: persona  # persona | task | composite | tool-wrapper
  domain_tags: [domain1, domain2]
  reasoning_style: analytical  # analytical | creative | adversarial | methodical | exploratory | conversational
  complexity: moderate  # atomic | moderate | complex
  composable: true  # Can be used as sub-agent?
  required_tools: [tool1]
  optional_tools: [tool2, tool3]
  recommended_graph: react  # react | chain | plan-execute
  max_iterations: 12
  author: your-name
  source: curated
  system_prompt: |
    # Identity + Mandate
    ...
    
    # Available Tools
    {{TOOL_BLOCK:tool1}}
    {{TOOL_BLOCK:tool2}}
    
    # Reasoning Protocol
    ...
    
    # Output Contract
    ...
    
    # Guardrails
    ...
```

### Python Dict Format

Add to `expanded_system_prompts.py` for full control:

```python
EXPANDED_TEMPLATES["my-new-agent-v2"] = {
    "id": "my-new-agent-v2",
    "name": "My New Agent",
    "version": "2.0.0",
    # ... all fields ...
    "system_prompt": """...""",
    "input_schema": {...},  # JSON Schema for validation
    "output_schema": {...},
}
```

## Available Templates

### Finance
- `financial-analyst-v2` — Equity research with fundamental + technical analysis
- `stock-market-analyst-v2` — Market timing and technical patterns

### Development
- `code-review-agent-v2` — Code quality, security, performance review
- `data-scientist-v2` — Data analysis, ML pipelines, insights
- `devops-engineer-v2` — Infrastructure, CI/CD, deployment

### Architecture
- `senior-system-architect-v2` — Enterprise system design

### Research
- `llm-researcher-v2` — AI/ML research and paper analysis
- `deep-research-protocol-v2` — Multi-hop research with evidence tracking
- `research-paper-evaluator-v2` — Academic paper review

### Creative
- `image-gen-director-v2` — Diffusion model prompt engineering

### Business
- `project-manager-v2` — PRDs, project plans, agile docs

### Security
- `cybersecurity-specialist-v2` — Threat modeling, security assessment

### Base Templates (stubs for expansion)
- `technical-writer` — Documentation
- `qa-engineer` — Test planning
- `ux-researcher` — Usability analysis
- `data-engineer` — Data pipelines
- `customer-support-agent` — Support interactions
- `content-strategist` — Content planning

## Runtime Support

The factory automatically detects the available runtime:

```python
factory = AgentFactory(...)
print(factory.runtime)  # "deepagents" or "langgraph"
```

### deepagents (Full Features)
- Planning capabilities (write_todos)
- Filesystem context management
- Sub-agent spawning

### langgraph (Fallback)
- Basic ReAct loop
- Works without deepagents installed
- Good for development/testing

## Project Structure

```
agentfactory/
├── agent_factory.py        # Core factory, registries, composer
├── expanded_system_prompts.py  # 12 fully-expanded v2 templates
├── prompt_registry.yaml    # 18 templates in YAML format
├── demo_factory.py         # Demo script
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
└── README.md              # This file
```

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add templates to `prompt_registry.yaml` or `expanded_system_prompts.py`
4. Register tools in your implementation
5. Submit a pull request

When adding templates:
- Follow the 5-layer structure
- Include all required fields
- Add appropriate domain tags
- Test with the demo script
