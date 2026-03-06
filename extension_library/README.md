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
| `commands/gh_repos.yaml` | `/gh_repos <query>` (alias: `/ghr`) | Search GitHub repositories | github MCP |
| `commands/gh_code.yaml` | `/gh_code <query>` (alias: `/ghc`) | Search code across GitHub | github MCP |
| `commands/gh_issues.yaml` | `/gh_issues <owner> <repo>` (alias: `/ghi`) | List issues for a repository | github MCP |
| `commands/gh_pr.yaml` | `/gh_pr <owner> <repo> <number>` (alias: `/ghp`) | Get details of a specific PR | github MCP |
| `commands/gh_prs.yaml` | `/gh_prs <owner> <repo>` | List pull requests for a repository | github MCP |
| `commands/gh_file.yaml` | `/gh_file <owner> <repo> <path>` (alias: `/ghf`) | Get contents of a file from a repo | github MCP |
| `commands/gh_commits.yaml` | `/gh_commits <owner> <repo>` | List recent commits | github MCP |
| `commands/gh_runs.yaml` | `/gh_runs <owner> <repo>` | List CI workflow runs | github MCP |
| `commands/gh_search_issues.yaml` | `/gh_search_issues <query>` (alias: `/ghsi`) | Search issues/PRs across GitHub | github MCP |
| `commands/github.yaml` | `/github <query>` (alias: `/gh`) | Research GitHub using AI agent | github MCP |
| `commands/hf_models.yaml` | `/hf_models <query>` (alias: `/hfm`) | Search HuggingFace models | huggingface MCP |
| `commands/hf_datasets.yaml` | `/hf_datasets <query>` (alias: `/hfd`) | Search HuggingFace datasets | huggingface MCP |
| `commands/hf_papers.yaml` | `/hf_papers <query>` (alias: `/hfp`) | Search HuggingFace papers | huggingface MCP |
| `commands/hf_spaces.yaml` | `/hf_spaces <query>` (alias: `/hfs`) | Search HuggingFace Spaces | huggingface MCP |
| `commands/hf_docs.yaml` | `/hf_docs <query>` | Search HuggingFace documentation | huggingface MCP |
| `commands/hf_repo.yaml` | `/hf_repo <repo_id>` (alias: `/hfr`) | Get detailed info about a HF repository | huggingface MCP |
| `commands/huggingface.yaml` | `/huggingface <query>` (alias: `/hf`) | Research HuggingFace using AI agent | huggingface MCP |

### Agents

Agents are autonomous LLM-powered tools that reason over multiple steps. They must be paired with an **agent-delegating command** to be usable — the agent YAML alone just registers the agent, the command YAML gives it a `/slash` trigger.

| File | Agent | Description | Tools Used | Command |
|------|-------|-------------|------------|---------|
| `agents/doc_researcher.yaml` | `doc_researcher` | Multi-tool documentation research agent | Context7 MCP + built-in web tools | `commands/research_docs.yaml` |
| `agents/github_researcher.yaml` | `github_researcher` | GitHub research agent for repos, issues, PRs, code | GitHub MCP + built-in web tools | `commands/github.yaml` |
| `agents/hf_researcher.yaml` | `hf_researcher` | HuggingFace Hub research agent for models, datasets, papers | HuggingFace MCP + built-in web/arXiv tools | `commands/huggingface.yaml` |

**Model requirements:** These agents rely on native tool calling, so the model must support it and correctly pass parameters to MCP tools. `gpt-oss:20b` produces the best results (structured tables, code, math) if your system has enough VRAM. `qwen3:8b` and `llama3.1:8b` are the minimum for reliable tool calling. Smaller models like `llama3.2:3b` tend to pass broken parameters, and reasoning models like `deepseek-r1` don't support tool calling at all.

### Installing extension sets

**Doc researcher** (Context7 + web):
```bash
cp extension_library/agents/doc_researcher.yaml ~/.tensortruth/agents/
cp extension_library/commands/research_docs.yaml ~/.tensortruth/commands/
```

**GitHub** (all GitHub commands + agent):
```bash
cp extension_library/agents/github_researcher.yaml ~/.tensortruth/agents/
cp extension_library/commands/gh_*.yaml ~/.tensortruth/commands/
cp extension_library/commands/github.yaml ~/.tensortruth/commands/
```

**HuggingFace** (all HF commands + agent):
```bash
cp extension_library/agents/hf_researcher.yaml ~/.tensortruth/agents/
cp extension_library/commands/hf_*.yaml ~/.tensortruth/commands/
cp extension_library/commands/huggingface.yaml ~/.tensortruth/commands/
```

## Extensions & Agentic Mode

When **agentic mode** is enabled (the default), the orchestrator can use MCP extension tools alongside built-in tools (RAG, web search, page fetch) in a single query. This means the agent can combine multiple sources autonomously — for example, searching arXiv for a paper, looking up implementation details via Context7, and fetching supplementary web pages — all in one go, without the user needing to run separate slash commands.

This is especially useful for complex research queries where a single tool isn't enough:

```
How does PyTorch implement flash attention? Check the docs and recent papers.
```

With Context7 installed, the orchestrator may call `resolve-library-id` + `query-docs` for PyTorch API details, `search_arxiv` for the original Flash Attention paper (built-in, no extension needed), and `search_web` + `fetch_page` for blog posts — then synthesize everything into a single cited response.

**Note:** Agentic mode requires models with reliable tool-calling (`qwen3:8b`+, `llama3.1:8b`+). If you're running smaller models, disable agentic mode in session settings and use the explicit slash commands instead (e.g., `/arxiv`, `/c7`, `/web`).

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

### GitHub (for `/gh_*`, `github_researcher`)

Requires the [GitHub MCP Server](https://github.com/github/github-mcp-server) via Docker. You must set a `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable with a [personal access token](https://github.com/settings/tokens) that has repo access.

```json
{
  "name": "github",
  "type": "stdio",
  "command": "docker",
  "args": [
    "run", "-i", "--rm",
    "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
    "-e", "GITHUB_TOOLSETS=repos,issues,pull_requests,actions",
    "ghcr.io/github/github-mcp-server"
  ],
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "$GITHUB_PERSONAL_ACCESS_TOKEN"
  },
  "enabled": true
}
```

The `env` field resolves `$VAR` references from your shell environment and passes them to the MCP server process. This is necessary because the MCP SDK only inherits a minimal set of default environment variables (HOME, PATH, etc.).

### HuggingFace (for `/hf_*`, `hf_researcher`)

Requires the [HuggingFace MCP Server](https://github.com/llmindset/hf-mcp-server) via npx. Set a `HF_TOKEN` environment variable with a [HuggingFace token](https://huggingface.co/settings/tokens).

```json
{
  "name": "huggingface",
  "type": "stdio",
  "command": "npx",
  "args": ["-y", "@llmindset/hf-mcp-server"],
  "env": {
    "HF_TOKEN": "$HF_TOKEN"
  },
  "enabled": true
}
```

### Full config example

A `mcp_servers.json` with all three servers:

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
      "name": "github",
      "type": "stdio",
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "-e", "GITHUB_TOOLSETS=repos,issues,pull_requests,actions",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "$GITHUB_PERSONAL_ACCESS_TOKEN"
      },
      "enabled": true
    },
    {
      "name": "huggingface",
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@llmindset/hf-mcp-server"],
      "env": {
        "HF_TOKEN": "$HF_TOKEN"
      },
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
