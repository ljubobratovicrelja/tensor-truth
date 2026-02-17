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

| File | Command | Description | Requires |
|------|---------|-------------|----------|
| `commands/arxiv.yaml` | `/arxiv <query>` (alias: `/ax`) | Search arXiv for academic papers | built-in |
| `commands/arxiv_paper.yaml` | `/arxiv_paper <id>` (alias: `/axp`) | Get detailed info about a paper by arXiv ID | built-in |
| `commands/context7.yaml` | `/context7 <library> <topic>` (alias: `/c7`) | Look up library docs via Context7 | context7 MCP |
| `commands/context7.py` | `/context7 <library> <topic>` | Same as above, Python reference implementation | context7 MCP |
| `commands/research_docs.yaml` | `/research_docs <query>` (alias: `/rd`) | Research docs using AI agent (Context7 + web) | context7 MCP |

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

Some extensions call MCP tools, so you need the relevant MCP server configured in `~/.tensortruth/mcp_servers.json`. The arXiv commands (`/arxiv`, `/arxiv_paper`) use built-in tools and need no MCP server.

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

A `mcp_servers.json` with Context7:

```json
{
  "servers": [
    {
      "name": "context7",
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"],
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
