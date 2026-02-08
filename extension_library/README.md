# Extension Library

Ready-to-use extensions for TensorTruth. Copy what you need to `~/.tensortruth/`.

For the full guide on writing your own extensions, see [docs/EXTENSIONS.md](../docs/EXTENSIONS.md).

## Quick Start

```bash
# Create directories (first time only)
mkdir -p ~/.tensortruth/commands ~/.tensortruth/agents

# Install a command
cp extension_library/commands/arxiv.yaml ~/.tensortruth/commands/

# Install an agent (needs both the agent + a command to trigger it)
cp extension_library/agents/doc_researcher.yaml ~/.tensortruth/agents/
cp extension_library/commands/research_docs.yaml ~/.tensortruth/commands/
```

Then restart `tensor-truth`. Check startup logs for confirmation.

## Available Extensions

### Commands

| File | Command | Description | MCP Server |
|------|---------|-------------|------------|
| `commands/arxiv.yaml` | `/arxiv <query>` (alias: `/ax`) | Search arXiv for academic papers | simple-arxiv |
| `commands/arxiv_paper.yaml` | `/arxiv_paper <id>` (alias: `/axp`) | Get detailed info about a paper by arXiv ID | simple-arxiv |
| `commands/context7.yaml` | `/context7 <library> <topic>` (alias: `/c7`) | Look up library docs via Context7 | context7 |
| `commands/context7.py` | `/context7 <library> <topic>` | Same as above, Python reference implementation | context7 |
| `commands/research_docs.yaml` | `/research_docs <query>` (alias: `/rd`) | Research docs using AI agent (Context7 + web) | context7 |

### Agents

Agents are autonomous LLM-powered tools that reason over multiple steps. They must be paired with an **agent-delegating command** to be usable — the agent YAML alone just registers the agent, the command YAML gives it a `/slash` trigger.

| File | Agent | Description | Tools Used | Command |
|------|-------|-------------|------------|---------|
| `agents/doc_researcher.yaml` | `doc_researcher` | Multi-tool documentation research agent | Context7 MCP + built-in web tools | `commands/research_docs.yaml` |

**Model requirements for `doc_researcher`:** This agent relies on native tool calling, so the model must support it and correctly pass parameters to MCP tools. `gpt-oss:20b` produces the best results (structured tables, code, math) if your system has enough VRAM. `qwen3:8b` and `llama3.1:8b` are the minimum for reliable tool calling. Smaller models like `llama3.2:3b` tend to pass broken parameters, and reasoning models like `deepseek-r1` don't support tool calling at all.

To install the doc_researcher agent, copy **both** files:

```bash
cp extension_library/agents/doc_researcher.yaml ~/.tensortruth/agents/
cp extension_library/commands/research_docs.yaml ~/.tensortruth/commands/
```

## MCP Server Prerequisites

Extensions call MCP tools, so you need the relevant MCP server configured in `~/.tensortruth/mcp_servers.json`. Below are the server configs for each extension set.

### arXiv (for `/arxiv`, `/arxiv_paper`)

Requires [mcp-simple-arxiv](https://github.com/andybrandt/mcp-simple-arxiv). Pre-install for faster startup:

```bash
uv tool install mcp-simple-arxiv   # or: pip install mcp-simple-arxiv
```

Add to `~/.tensortruth/mcp_servers.json`:

```json
{
  "name": "simple-arxiv",
  "type": "stdio",
  "command": "uvx",
  "args": ["mcp-simple-arxiv"],
  "enabled": true
}
```

If you don't have `uvx`, use `"command": "python", "args": ["-m", "mcp_simple_arxiv"]` instead (requires `pip install mcp-simple-arxiv`).

> **Note:** `mcp-simple-arxiv` v0.6.0 prints a `ValueError` traceback to stderr on startup. This is an [upstream bug](https://github.com/andybrandt/mcp-simple-arxiv) — tools load and work correctly despite the error.

### Context7 (for `/context7`, `doc_researcher`)

Requires [Context7 MCP](https://github.com/upstash/context7). No pre-install needed (`npx` handles it):

```json
{
  "name": "context7",
  "type": "stdio",
  "command": "npx",
  "args": ["-y", "@upstash/context7-mcp@latest"],
  "enabled": true
}
```

### Full config example

A `mcp_servers.json` with both servers:

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

## Writing Your Own

See [docs/EXTENSIONS.md](../docs/EXTENSIONS.md) for the full reference, including:

- YAML command schema (steps, templates, `result_extract`)
- YAML agent schema
- Python extension convention
- Template variable reference
- Troubleshooting
