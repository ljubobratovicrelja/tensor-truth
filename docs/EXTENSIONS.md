# Custom Extensions: Commands, Agents & Tools

TensorTruth supports user-defined commands and agents via simple YAML config files (or Python for advanced cases). Drop files into `~/.tensortruth/commands/` or `~/.tensortruth/agents/`, restart the app, and your extensions are available immediately.

Extensions build on [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) tools. If you've already added MCP servers to `~/.tensortruth/mcp_servers.json`, extensions let you wire those tools into slash commands and autonomous agents without writing code.

## Quick Start: arXiv Search in 3 Steps

This example adds an `/arxiv` command that searches academic papers. It assumes you have `tensor-truth` installed via pip.

**Step 1: Add the MCP server**

Create or edit `~/.tensortruth/mcp_servers.json`:

```json
{
  "servers": [
    {
      "name": "simple-arxiv",
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-simple-arxiv"],
      "enabled": true
    }
  ]
}
```

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) for `uvx`. Alternatively, `pip install mcp-simple-arxiv` and use `"command": "python", "args": ["-m", "mcp_simple_arxiv"]`.

**Step 2: Create the command YAML**

```bash
mkdir -p ~/.tensortruth/commands
```

Create `~/.tensortruth/commands/arxiv.yaml`:

```yaml
name: arxiv
description: "Search arXiv for academic papers"
usage: "/arxiv <query>"
aliases: [ax]

steps:
  - tool: search_papers
    params:
      query: "{{args}}"
      max_results: 5
      sort_by: relevance

requires_mcp: simple-arxiv
```

**Step 3: Restart and use**

```bash
tensor-truth
```

Check startup logs for `Registered user command: /arxiv`. Then in the chat:

```
/arxiv attention is all you need
/ax transformer architecture          # alias works too
```

That's it. No code changes, no rebuilds.

## Ready-Made Extensions

The repository includes an [`extension_library/`](../extension_library/) directory with tested extensions you can copy directly:

```bash
# Copy a command
cp extension_library/commands/arxiv.yaml ~/.tensortruth/commands/

# Copy an agent
cp extension_library/agents/doc_researcher.yaml ~/.tensortruth/agents/
```

See the [extension library README](../extension_library/README.md) for the full catalog and MCP server prerequisites.

## How It Works

On startup, TensorTruth scans two directories:

```
~/.tensortruth/
├── commands/          # YAML or Python → slash commands
│   ├── arxiv.yaml
│   ├── context7.yaml
│   └── my_custom.py
├── agents/            # YAML → autonomous agents
│   └── doc_researcher.yaml
└── mcp_servers.json   # MCP tool sources (already exists)
```

- `.yaml` / `.yml` files are validated against Pydantic schemas
- `.py` files are dynamically imported and must expose a `register()` function
- Extensions that fail to load are skipped with a warning (never crash the app)
- Startup logs show exactly what loaded: `Loaded user extensions: 2 commands, 1 agents`

## YAML Command Reference

A YAML command defines a **sequential tool pipeline** — each step calls one MCP tool with templated parameters.

### Minimal Example

```yaml
name: mycommand
description: "What it does"
steps:
  - tool: some-tool
    params:
      query: "{{args}}"
```

### Full Schema

```yaml
name: mycommand                    # Required. Becomes /mycommand
description: "What it does"       # Required. Shown in /help
usage: "/mycommand <args>"        # Optional. Auto-generated if omitted
aliases: [mc, my]                 # Optional. Alternative names

steps:                             # Sequential tool pipeline
  - tool: tool-name               # MCP tool to call
    params:                        # Parameters (supports templates)
      param1: "{{args}}"
      param2: "{{args.0}}"
    result_var: step1              # Store output as template variable
    result_extract: "ID: (?P<id>\\S+)"  # Optional regex to extract fields

  - tool: another-tool
    params:
      input: "{{step1.id}}"       # Use extracted field from step 1

response: "{{_last_result}}"      # Optional. Template for final output
requires_mcp: server-name         # Optional. Hint for error messages
```

### Template Variables

Templates use `{{variable}}` syntax. No loops or conditionals — just variable substitution.

| Pattern | Resolves to |
|---------|-------------|
| `{{args}}` | Full text after command name |
| `{{args.0}}`, `{{args.1}}` | Positional args (whitespace-split) |
| `{{args.rest}}` | Everything after the first arg |
| `{{step_var}}` | Full string output from a named step |
| `{{step_var.field}}` | Dot-path into JSON-parsed step output |
| `{{_last_result}}` | Output of the most recent step |

### Extracting Data Between Steps

When a tool returns human-readable text (not JSON), use `result_extract` to pull out specific values with a regex. Named capture groups become template variables.

```yaml
steps:
  - tool: resolve-library-id
    params:
      libraryName: "{{args.0}}"
    result_var: resolved
    result_extract: "library ID: (?P<libraryId>\\S+)"

  - tool: query-docs
    params:
      libraryId: "{{resolved.libraryId}}"   # From named group
```

If the tool returns JSON, fields are flattened automatically — no regex needed:

```yaml
steps:
  - tool: search-api
    params:
      q: "{{args}}"
    result_var: result
    # If tool returns {"id": "abc", "title": "..."}, then:
    # {{result.id}} and {{result.title}} are available automatically

  - tool: get-details
    params:
      id: "{{result.id}}"
```

### Agent-Delegating Commands

Instead of a tool pipeline, a command can delegate to a registered agent:

```yaml
name: research_docs
description: "Research documentation using AI agent"
usage: "/research_docs <query>"
agent: doc_researcher    # Name of agent from ~/.tensortruth/agents/
```

`steps` and `agent` are mutually exclusive.

## YAML Agent Reference

Agent YAML files define autonomous agents that use LLM reasoning with tool access.

```yaml
name: doc_researcher
description: "Research using Context7 + web search"
tools:                            # List of MCP/built-in tool names
  - search_web
  - fetch_page
  - get-library-docs
agent_type: function              # "function" (recommended) or "router"
system_prompt: |                  # Instructions for the LLM
  You are a documentation researcher. Use library docs and web search
  to find accurate, up-to-date information. Always cite your sources.
max_iterations: 10                # Max tool-call rounds (default: 10)
model: null                       # null = use session model
factory_params: {}                # Advanced: passed to agent factory
```

Place in `~/.tensortruth/agents/` and reference from agent-delegating commands or use directly via the agent system.

**Agent types:**
- `function` — Generic tool-calling agent. Recommended for most use cases. Uses the system prompt and tools you specify.
- `router` — Uses the built-in browse agent's search/fetch/synthesize routing logic. Less flexible for custom agents.

**Model behavior:**
- `model: null` — Uses whatever model the user has selected in the session (recommended)
- `model: "deepseek-r1:8b"` — Pins to a specific Ollama model

## Python Extensions

For logic that YAML can't express, use Python files with a `register()` entry point:

```python
# ~/.tensortruth/commands/my_custom.py
from tensortruth.api.routes.commands import ToolCommand

class MyCommand(ToolCommand):
    name = "mycommand"
    aliases = ["mc"]
    description = "My custom command"
    usage = "/mycommand <args>"

    async def execute(self, args, session, websocket):
        from tensortruth.api.deps import get_tool_service
        tool_service = get_tool_service()
        result = await tool_service.execute_tool("some-tool", {"param": args})
        await websocket.send_json({"type": "done", "content": str(result["data"])})

def register(command_registry, agent_service, tool_service):
    """Required entry point. Receives all three registries."""
    command_registry.register(MyCommand())
```

The `register()` function can register commands, agents, and tools from a single file.

## MCP Server Setup

Extensions call MCP tools, so the relevant MCP server must be configured. Add servers to `~/.tensortruth/mcp_servers.json`:

```json
{
  "servers": [
    {
      "name": "context7",
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"],
      "enabled": true
    },
    {
      "name": "simple-arxiv",
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-simple-arxiv"],
      "enabled": true
    }
  ]
}
```

**Common MCP server types:**

| Runtime | Config |
|---------|--------|
| npm package | `"command": "npx", "args": ["-y", "package-name"]` |
| Python package (uvx) | `"command": "uvx", "args": ["package-name"]` |
| Python package (installed) | `"command": "python", "args": ["-m", "module_name"]` |
| Local script | `"command": "node", "args": ["/path/to/server.js"]` |

**Tip:** Run the MCP server command manually first to make sure it starts. For Python packages, pre-install with `pip install` or `uv tool install` to avoid first-run download delays.

## Troubleshooting

**Command doesn't appear after restart**

Check the startup logs for warnings. Common causes:
- Invalid YAML syntax (indentation, missing quotes)
- Missing required fields (`name`, `description`, `steps` or `agent`)
- File not in `~/.tensortruth/commands/` (check the path)

**"Tool X not found" at runtime**

The MCP server isn't configured or failed to start:
- Verify the server is in `mcp_servers.json` with `"enabled": true`
- Check startup logs for `Failed to connect to MCP server`
- Test the server command manually: `npx -y @upstash/context7-mcp@latest`

**"Template variable not resolved"**

A `{{variable}}` in your YAML couldn't be found in the context:
- Check the variable name matches `result_var` from a previous step
- For dot-path access (`{{step.field}}`), verify the tool returns JSON or use `result_extract`
- Use `{{_last_result}}` as a fallback for the raw output of the previous step

**Python extension not loading**

- Must have a `register(command_registry, agent_service, tool_service)` function
- Import errors are caught and logged — check logs for the specific error
- Use absolute imports (`from tensortruth.api.deps import ...`)

**Extension works locally but not in Docker**

Mount the extensions directory in your Docker run command:
```bash
docker run -d \
  -v ~/.tensortruth:/root/.tensortruth \
  ...
```

The `-v ~/.tensortruth:/root/.tensortruth` mapping already covers commands, agents, and MCP config.
