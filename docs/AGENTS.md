# Autonomous Agents

Tensor-Truth supports autonomous agents via [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) integration. This document covers usage, configuration, and connecting external MCP servers.

## Table of Contents

- [Using Agents](#using-agents)
  - [Command-Based](#command-based)
  - [Natural Language Triggers](#natural-language-triggers)
- [Configuration](#configuration)
  - [Agent Settings](#agent-settings)
  - [Preset Configuration](#preset-configuration)
- [Connecting MCP Servers](#connecting-mcp-servers)
  - [Configuration File](#configuration-file)
  - [Examples](#examples)
- [Architecture & Roadmap](#architecture--roadmap)
- [Troubleshooting](#troubleshooting)

---

## Using Agents

### Command-Based

Use the `/browse` command followed by your research query:

```
/browse What are the key differences between React and Vue.js?
/browse Latest developments in quantum computing 2024
/browse How does the Python GIL work?
```

### Natural Language Triggers

You can also invoke agents naturally in conversation. The system detects trigger words and routes your message to the appropriate agent:

**Browse/Research triggers:**
```
Research the latest transformer architectures
Browse recent papers on diffusion models
Find out how attention mechanisms work
Look up the PyTorch autograd implementation
```

**Web Search triggers:**
```
Search the web for Python 3.13 release notes
Google the differences between pip and uv
```

---

## Configuration

### Agent Settings

Agent behavior is configured in `~/.tensortruth/config.yaml` under the `agent` section:

```yaml
agent:
  # Maximum reasoning iterations before stopping
  max_iterations: 10

  # Model for agent reasoning (should be fast)
  reasoning_model: "llama3.1:8b"

  # Enable natural language agent triggers
  enable_natural_language_agents: true

  # Model for intent classification (should be very fast)
  intent_classifier_model: "llama3.2:3b"
```

| Setting | Default | Description |
|---------|---------|-------------|
| `max_iterations` | 10 | Maximum tool calls before the agent stops. Increase for more thorough research. |
| `reasoning_model` | `llama3.1:8b` | Fast model for deciding what tools to use. |
| `enable_natural_language_agents` | `true` | Whether to detect agent triggers in natural language. |
| `intent_classifier_model` | `llama3.2:3b` | Model for classifying user intent. |

### Preset Configuration

You can override agent settings per-preset in `~/.tensortruth/presets.json`:

```json
{
  "DL Research Assistant": {
    "description": "Deep learning research with thorough web searches",
    "modules": ["book_deep_learning_goodfellow", "dl_foundations"],
    "model": "deepseek-r1:14b",
    "agent_max_iterations": 15,
    "agent_reasoning_model": "llama3.1:8b"
  },
  "Fast Researcher": {
    "description": "Quick answers with fewer iterations",
    "agent_max_iterations": 6
  }
}
```

---

## Connecting MCP Servers

You can extend Tensor-Truth with additional MCP servers to give agents access to more tools (GitHub, ArXiv, filesystem, databases, etc.).

### Configuration File

Create `~/.tensortruth/mcp_servers.json` to register external MCP servers:

```json
{
  "servers": [
    {
      "name": "server-name",
      "type": "stdio",
      "command": "command-to-run",
      "args": ["arg1", "arg2"],
      "description": "What this server provides",
      "enabled": true
    }
  ]
}
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier for the server |
| `type` | Yes | Transport type: `stdio` or `sse` |
| `command` | For stdio | Command to execute |
| `args` | For stdio | Command arguments |
| `url` | For sse | Server URL for SSE transport |
| `description` | No | Human-readable description |
| `enabled` | No | Set to `false` to disable (default: `true`) |

### Examples

**GitHub MCP Server:**

If you have the [GitHub MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/github) installed:

```json
{
  "servers": [
    {
      "name": "github",
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "description": "GitHub repository access",
      "enabled": true
    }
  ]
}
```

**Filesystem MCP Server:**

For local file access with the [filesystem MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem):

```json
{
  "servers": [
    {
      "name": "filesystem",
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"],
      "description": "Local filesystem access",
      "enabled": true
    }
  ]
}
```

**Multiple Servers:**

```json
{
  "servers": [
    {
      "name": "github",
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "enabled": true
    },
    {
      "name": "arxiv",
      "type": "stdio",
      "command": "python",
      "args": ["-m", "arxiv_mcp_server"],
      "enabled": true
    },
    {
      "name": "postgres",
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
      "enabled": false
    }
  ]
}
```

Tools from all enabled servers are automatically available to agents.

---

## Architecture & Roadmap

The agent system is designed around two core principles:

1. **Configuration-driven MCP integration** - Connect to any MCP-compatible server via JSON configuration, no code changes required
2. **Extensible tooling framework** - Define custom tools and agents through user configuration

### Current Implementation

The current release provides:

- **Web Research Agent** (`/browse`) - Autonomous web search and synthesis
- **MCP Server Registry** - Load tools from multiple MCP servers via `~/.tensortruth/mcp_servers.json`
- **Natural Language Routing** - Trigger agents through conversational patterns

### Planned Features

**Custom Agents (Planned)**

Define new agents in `~/.tensortruth/agents.json`:

```json
{
  "agents": [
    {
      "name": "github",
      "command": "/github",
      "description": "GitHub repository operations",
      "mcp_servers": ["github"],
      "system_prompt": "You are a GitHub assistant..."
    }
  ]
}
```

**Custom Tools (Planned)**

Define tools via configuration or Python scripts in `~/.tensortruth/tools/`:

```
~/.tensortruth/
  tools/
    my_tool.json      # Config-based tool definition
    my_tool.py        # Python-based tool implementation
```

**Tool Routing (Planned)**

Automatic routing based on available MCP servers - the agent will be aware of all connected tools and use them appropriately based on the user's request.

### Design Goals

- **No code changes for extensibility** - Users add capabilities through config files
- **MCP as the standard** - Any MCP-compatible server works out of the box
- **Composable agents** - Mix and match tools from different MCP servers
- **Local-first** - All configuration in `~/.tensortruth/`, no external dependencies

---

## Troubleshooting

### Agent returns "No tools available"

**Cause:** MCP server failed to start or load tools.

**Solution:**
1. Verify the MCP server command works standalone
2. Check server logs for errors
3. Ensure required dependencies are installed (e.g., `npx` for Node-based servers)

### Agent hits iteration limit

**Cause:** Research required more steps than `max_iterations` allows.

**Solution:** Increase `agent_max_iterations` in your preset or config. The agent returns partial results with a warning when this happens.

### Slow agent responses

**Solution:**
- Use a smaller reasoning model (e.g., `llama3.2:3b`)
- Reduce `max_iterations` for faster but less thorough research

### Natural language triggers not working

**Solution:**
1. Verify `enable_natural_language_agents: true` in config
2. Ensure classifier model is available: `ollama pull llama3.2:3b`
3. Use explicit `/browse` command as fallback

### External MCP server not loading

**Solution:**
1. Test the server command directly in terminal
2. Check `~/.tensortruth/mcp_servers.json` syntax
3. Verify `enabled: true` for the server
