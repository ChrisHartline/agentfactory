"""
Agent Factory Demo
==================

Demonstrates the full factory lifecycle:
1. Initialize registries
2. Load prompt templates (enriched 155 + expanded 12)
3. Register tools
4. Search the registry
5. Compose a system prompt
6. Spawn an agent
7. View genealogy

This demo uses mock tools since we don't have live API keys.
In production, you'd register real MCP servers and LangChain tools.
"""

import sys
import json
from pathlib import Path

# Get the directory where this script lives
SCRIPT_DIR = Path(__file__).parent.resolve()

from agent_factory import (
    AgentFactory,
    PromptRegistry,
    ToolRegistry,
    ToolRegistryEntry,
    SpawnConfig,
    ReasoningFramework,
    ReasoningStyle,
    PromptType,
)


def create_mock_tools():
    """
    Create mock tool registry entries.

    In production, these would be real MCP servers or LangChain tools.
    The key thing to see is the prompt_instructions field — that's
    what gets injected into {{TOOL_BLOCK:xxx}} placeholders.
    """

    tools = {
        "web_search": ToolRegistryEntry(
            id="web_search",
            name="Web Search",
            description="Search the web for current information",
            tool_type="mcp_server",
            prompt_instructions="""## Tool: web_search
Search the web for current information, news, and data.

Parameters:
  - query (str): The search query. Be specific and concise.
  - max_results (int, optional): Number of results to return (default: 5, max: 10)

Returns: JSON array of results, each with 'title', 'url', 'snippet'.

Usage notes:
  - Prefer specific queries over broad ones
  - Check multiple sources for important claims
  - Results are ranked by relevance""",
            config={"server": "tavily", "api_key_env": "TAVILY_API_KEY"},
            rate_limit="10 calls/minute",
            required_env_vars=["TAVILY_API_KEY"],
            domain_tags=["general", "research", "news"],
        ),

        "market_data_api": ToolRegistryEntry(
            id="market_data_api",
            name="Market Data API",
            description="Retrieve stock market data, prices, and technical indicators",
            tool_type="mcp_server",
            prompt_instructions="""## Tool: market_data_api
Retrieve real-time and historical stock market data.

Parameters:
  - ticker (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
  - timeframe (str): "1d", "5d", "1m", "3m", "6m", "1y", "5y"
  - indicators (list[str], optional): Technical indicators to compute.
    Available: ["rsi", "ema_20", "ema_50", "ema_200", "macd", "bollinger",
                "volume_avg", "atr", "stochastic"]
  - data_type (str, optional): "price" (default), "fundamentals", "options"

Returns: JSON with 'ticker', 'current_price', 'history' (OHLCV array),
         and 'indicators' (computed values).

Usage notes:
  - Rate limit: 5 calls per minute
  - Fundamental data updates quarterly
  - Intraday data available for last 30 days only
  - If the API returns an error, retry once before reporting failure""",
            config={"server": "alpha_vantage", "api_key_env": "ALPHA_VANTAGE_KEY"},
            rate_limit="5 calls/minute",
            required_env_vars=["ALPHA_VANTAGE_KEY"],
            domain_tags=["finance", "trading", "data-science"],
        ),

        "code_executor": ToolRegistryEntry(
            id="code_executor",
            name="Code Executor",
            description="Execute Python code in a sandboxed environment",
            tool_type="langchain_tool",
            prompt_instructions="""## Tool: code_executor
Execute Python code in an isolated sandbox.

Parameters:
  - code (str): Python code to execute
  - timeout (int, optional): Max execution time in seconds (default: 30)

Returns: JSON with 'stdout', 'stderr', 'return_value', 'execution_time_ms'.

Pre-installed packages: pandas, numpy, scikit-learn, matplotlib, seaborn,
                        requests, beautifulsoup4, pyyaml, json

Usage notes:
  - Sandbox has no network access
  - File writes go to /tmp/ (cleared between executions)
  - matplotlib plots are returned as base64 PNG
  - Maximum 30 second execution time
  - Do NOT use for untrusted code or code that modifies system state""",
            config={"sandbox": "docker", "image": "python:3.11-slim"},
            domain_tags=["development", "data-science", "testing"],
        ),

        "calculator": ToolRegistryEntry(
            id="calculator",
            name="Calculator",
            description="Perform mathematical calculations",
            tool_type="function",
            prompt_instructions="""## Tool: calculator
Evaluate mathematical expressions.

Parameters:
  - expression (str): Mathematical expression to evaluate.
    Supports: +, -, *, /, **, sqrt(), log(), sin(), cos(), pi, e

Returns: Numeric result as a string.

Usage notes:
  - Use for precise calculations rather than mental math
  - Supports complex expressions: "sqrt(2) * pi / 3"
  - Returns error message for invalid expressions""",
            domain_tags=["math", "finance", "data-science"],
        ),

        "arxiv_mcp": ToolRegistryEntry(
            id="arxiv_mcp",
            name="ArXiv Search",
            description="Search and retrieve academic papers from ArXiv",
            tool_type="mcp_server",
            prompt_instructions="""## Tool: arxiv_mcp
Search the ArXiv preprint repository for academic papers.

Parameters:
  - query (str): Search query (supports ArXiv search syntax)
  - max_results (int, optional): Number of papers to return (default: 5)
  - sort_by (str, optional): "relevance" (default), "lastUpdatedDate", "submittedDate"
  - categories (list[str], optional): ArXiv categories (e.g., ["cs.AI", "cs.LG", "quant-ph"])

Returns: JSON array of papers with 'title', 'authors', 'abstract',
         'arxiv_id', 'pdf_url', 'published_date', 'categories'.

Usage notes:
  - Abstracts are included; full paper text is NOT
  - Use pdf_url to reference papers for the user
  - Category filters significantly improve result quality
  - For quantum computing papers, use categories: ["quant-ph", "cs.ET"]""",
            config={"server": "arxiv-mcp", "endpoint": "https://export.arxiv.org/api/query"},
            domain_tags=["research", "ai-ml", "math"],
        ),

        "file_system": ToolRegistryEntry(
            id="file_system",
            name="File System",
            description="Read and write files in the agent's workspace",
            tool_type="langchain_tool",
            prompt_instructions="""## Tool: file_system
Read, write, and list files in the agent's workspace.

Operations:
  - read(path): Read file contents
  - write(path, content): Write content to file
  - list(directory): List files in directory
  - append(path, content): Append to existing file

Usage notes:
  - Workspace is isolated per agent session
  - Use for storing intermediate results, notes, and artifacts
  - Maximum file size: 5MB
  - Supported formats: .txt, .json, .yaml, .py, .md, .csv""",
            domain_tags=["general", "development"],
        ),

        "chart_generator": ToolRegistryEntry(
            id="chart_generator",
            name="Chart Generator",
            description="Generate charts and visualizations",
            tool_type="function",
            prompt_instructions="""## Tool: chart_generator
Generate charts and data visualizations.

Parameters:
  - chart_type (str): "line", "bar", "scatter", "pie", "heatmap", "candlestick"
  - data (dict): Data to plot (format depends on chart_type)
  - title (str): Chart title
  - x_label (str, optional): X-axis label
  - y_label (str, optional): Y-axis label
  - style (str, optional): "default", "dark", "publication", "minimal"

Returns: Path to generated PNG image.

Usage notes:
  - Candlestick charts expect OHLCV data format
  - Maximum 10,000 data points per chart
  - Output resolution: 1200x800 pixels""",
            domain_tags=["finance", "data-science", "research"],
        ),

        "image_gen": ToolRegistryEntry(
            id="image_gen",
            name="Image Generator",
            description="Generate images from text descriptions using diffusion models",
            tool_type="mcp_server",
            prompt_instructions="""## Tool: image_gen
Generate images from text prompts using a diffusion model.

Parameters:
  - prompt (str): Detailed description of the desired image
  - negative_prompt (str, optional): What to avoid in the image
  - width (int, optional): Image width (default: 1024, options: 512, 768, 1024, 1792)
  - height (int, optional): Image height (default: 1024, options: 512, 768, 1024, 1792)
  - steps (int, optional): Generation quality steps (default: 30, range: 20-50)
  - seed (int, optional): For reproducible results

Returns: Path to generated image file.

Usage notes:
  - More detailed prompts produce better results
  - Include style, lighting, composition, and mood in prompts
  - Typical generation time: 10-30 seconds
  - Never include real person names in prompts""",
            config={"server": "comfyui", "endpoint": "http://localhost:8188"},
            domain_tags=["image-generation", "creative"],
        ),

        "network_tools": ToolRegistryEntry(
            id="network_tools",
            name="Network Analysis Tools",
            description="Analyze network packets and configurations",
            tool_type="function",
            prompt_instructions="""## Tool: network_tools
Analyze network packets, configurations, and topology.

Operations:
  - analyze_pcap(file_path): Analyze a packet capture file
  - parse_config(config_text, vendor): Parse router/switch configs (Cisco, Juniper)
  - dns_lookup(domain): Perform DNS resolution
  - port_scan_results(results_json): Analyze port scan output

Returns: Structured analysis as JSON.

Usage notes:
  - PCAP analysis returns protocol distribution, top talkers, and anomalies
  - Config parsing supports Cisco IOS, IOS-XE, NX-OS, and Juniper JunOS
  - Does NOT perform active scanning — only analyzes provided data""",
            domain_tags=["networking", "security"],
        ),

        "git_tools": ToolRegistryEntry(
            id="git_tools",
            name="Git Tools",
            description="Interact with Git repositories",
            tool_type="mcp_server",
            prompt_instructions="""## Tool: git_tools
Interact with Git repositories.

Operations:
  - clone(repo_url): Clone a repository
  - diff(path, branch): Show changes
  - log(path, n): Show recent commits
  - blame(path, file): Show line-by-line authorship
  - create_branch(path, name): Create a new branch
  - commit(path, message, files): Stage and commit files

Returns: Command output as text.

Usage notes:
  - Clone to the agent's workspace directory
  - Commits are local only (no push without explicit approval)
  - Maximum repo size: 500MB""",
            config={"server": "github-mcp"},
            domain_tags=["development", "devops"],
        ),

        "database": ToolRegistryEntry(
            id="database",
            name="Database Query",
            description="Execute SQL queries against connected databases",
            tool_type="mcp_server",
            prompt_instructions="""## Tool: database
Execute read-only SQL queries against connected databases.

Parameters:
  - query (str): SQL query to execute (SELECT only)
  - database (str, optional): Database name if multiple connected
  - limit (int, optional): Maximum rows to return (default: 100)

Returns: JSON with 'columns', 'rows', 'row_count', 'execution_time_ms'.

Usage notes:
  - READ-ONLY: INSERT, UPDATE, DELETE, DROP are blocked
  - Maximum 100 rows per query (use LIMIT)
  - Query timeout: 30 seconds
  - Available databases are listed on connection""",
            domain_tags=["data-science", "development"],
        ),
    }

    return tools


# ============================================================
# DEMO EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("  AGENT FACTORY DEMO")
    print("=" * 60)

    # -- 1. Initialize registries -----------------------------
    print("\n[1] Initializing registries...")
    prompt_registry = PromptRegistry()
    tool_registry = ToolRegistry()

    # -- 2. Load prompt templates -----------------------------
    print("\n[2] Loading prompt templates...")

    # Load base templates from YAML
    yaml_path = SCRIPT_DIR / 'prompt_registry.yaml'
    count_yaml = prompt_registry.load_from_yaml(yaml_path)
    print(f"    Loaded {count_yaml} templates from YAML registry")

    # Load the 12 expanded templates (these override the v1 entries)
    from expanded_system_prompts import EXPANDED_TEMPLATES
    count_expanded = prompt_registry.load_from_dict(EXPANDED_TEMPLATES)
    print(f"    Loaded {count_expanded} expanded templates (v2, with full system prompts)")

    print(f"    Total templates: {prompt_registry.count}")

    # -- 3. Register tools ------------------------------------
    print("\n[3] Registering tools...")
    mock_tools = create_mock_tools()
    for tool_id, entry in mock_tools.items():
        tool_registry.register(entry)
    print(f"    Registered {tool_registry.count} tools")

    # -- 4. Initialize the factory ----------------------------
    print("\n[4] Initializing Agent Factory...")
    factory = AgentFactory(
        prompt_registry=prompt_registry,
        tool_registry=tool_registry,
        default_model="anthropic:claude-sonnet-4-5-20250929",
        max_recursion_depth=3,
    )
    print("    Factory ready!")

    # -- 5. Demo: Search the registry -------------------------
    print("\n" + "=" * 60)
    print("  DEMO: Registry Search")
    print("=" * 60)

    # Search for finance-related templates
    print("\n[Search] domain_tags=['finance'], reasoning_style='analytical':")
    results = prompt_registry.search(
        domain_tags=["finance"],
        reasoning_style=ReasoningStyle.ANALYTICAL,
    )
    for t in results[:5]:
        print(f"    [{t.id}] {t.name} (v{t.version})")
        print(f"      Tools: {t.required_tools} | Graph: {t.recommended_graph.value}")

    # Search for composable research agents
    print("\n[Search] query='research', composable_only=True:")
    results = prompt_registry.search(
        query="research",
        composable_only=True,
    )
    for t in results[:5]:
        print(f"    [{t.id}] {t.name}")
        print(f"      Domains: {', '.join(t.domain_tags)}")

    # -- 6. Demo: Compose a system prompt ---------------------
    print("\n" + "=" * 60)
    print("  DEMO: Prompt Composition")
    print("=" * 60)

    template = prompt_registry.get("financial-analyst-v2")
    if template:
        config = SpawnConfig(
            template_id="financial-analyst-v2",
            reasoning_framework=ReasoningFramework.SIRP,
            task_context="Analyze NVDA for potential entry points given the recent AI infrastructure buildout.",
            budget={"max_tool_calls": 10, "timeout": 120},
        )
        composed = factory.composer.compose(template, config)

        print(f"\n[Composed prompt for: {template.name}]")
        print(f"  Length: {len(composed)} chars")
        print(f"\n--- First 2000 chars ---")
        print(composed[:2000])
        print(f"\n--- Last 800 chars ---")
        print(composed[-800:])

        # Verify tool blocks were replaced
        import re
        remaining_blocks = re.findall(r'\{\{TOOL_BLOCK:\w+\}\}', composed)
        print(f"\n[Verification] Unreplaced tool blocks: {len(remaining_blocks)}")
        if remaining_blocks:
            print(f"  Remaining: {remaining_blocks}")

    # -- 7. Demo: Show what tools got resolved ----------------
    print("\n" + "=" * 60)
    print("  DEMO: Tool Resolution")
    print("=" * 60)

    tool_ids = ["market_data_api", "web_search", "calculator", "chart_generator", "nonexistent_tool"]
    _, resolved_ids, missing_ids = tool_registry.resolve(tool_ids)
    print(f"\n  Requested: {tool_ids}")
    print(f"  Resolved:  {resolved_ids}")
    print(f"  Missing:   {missing_ids}")

    # -- 8. Demo: Spawn an agent (dry run — no API key) -------
    print("\n" + "=" * 60)
    print("  DEMO: Agent Spawn (dry run)")
    print("=" * 60)

    # We can't actually invoke without an API key, but we can
    # show the full creation flow
    try:
        result = factory.create(SpawnConfig(
            template_id="code-review-agent-v2",
            reasoning_framework=ReasoningFramework.STANDARD,
            task_context="Review the agent_factory.py module for code quality.",
        ))
        print(f"\n  [OK] Agent spawned: {result.agent_id}")
        print(f"    Template: {result.template_id}")
        print(f"    Tools attached: {result.tools_attached}")
        print(f"    Tools missing: {result.tools_missing}")
        print(f"    Prompt length: {len(result.composed_prompt)} chars")
    except Exception as e:
        print(f"\n  Agent creation result: {e}")

    # Spawn another to show genealogy
    try:
        result2 = factory.create(SpawnConfig(
            template_id="financial-analyst-v2",
            parent_agent_id=result.agent_id if 'result' in dir() else None,
        ))
        print(f"\n  [OK] Agent spawned: {result2.agent_id}")
        print(f"    Parent: {result2.genealogy.get('parent_agent_id', 'ROOT')}")
        print(f"    Depth: {result2.genealogy.get('depth', 0)}")
    except Exception as e:
        print(f"\n  Second agent: {e}")

    # -- 9. Demo: Genealogy -----------------------------------
    print("\n" + "=" * 60)
    print("  DEMO: Spawn Genealogy")
    print("=" * 60)
    print(f"\n{factory.print_genealogy()}")

    # -- 10. Demo: Factory as Tool ----------------------------
    print("\n" + "=" * 60)
    print("  DEMO: Factory as LangChain Tool")
    print("=" * 60)

    spawn_tool = factory.as_tool()
    search_tool = factory.search_registry_tool()

    print(f"\n  spawn_agent tool:")
    print(f"    Name: {spawn_tool.name}")
    print(f"    Description: {spawn_tool.description[:150]}...")

    print(f"\n  search_prompt_registry tool:")
    print(f"    Name: {search_tool.name}")
    print(f"    Description: {search_tool.description[:150]}...")

    # Actually call the search tool (doesn't need API key)
    print(f"\n  [Calling search_prompt_registry('security')]")
    search_result = search_tool.invoke({
        "query": "security",
        "domain_tags": ["security"],
    })
    print(f"  {search_result[:500]}")

    # -- Summary ----------------------------------------------
    print("\n" + "=" * 60)
    print("  FACTORY STATUS")
    print("=" * 60)
    print(f"""
  Prompt Registry:  {prompt_registry.count} templates loaded
  Tool Registry:    {tool_registry.count} tools registered
  Agents Spawned:   {factory._spawn_counter}
  Max Recursion:    {factory.max_recursion_depth} levels

  Clara can use these tools to create agents:
    1. search_prompt_registry - find the right template
    2. spawn_agent - create and invoke a specialized agent

  The factory handles:
    [x] Prompt composition (5 layers + SIRP injection)
    [x] Tool resolution (required vs optional, missing warnings)
    [x] Agent building (via deepagents runtime)
    [x] Genealogy tracking (who spawned whom)
    [x] Recursion depth limits (prevent infinite spawning)
    """)


if __name__ == "__main__":
    main()
