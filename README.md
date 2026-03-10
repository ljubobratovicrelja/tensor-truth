![logo](https://raw.githubusercontent.com/ljubobratovicrelja/tensor-truth/main/media/tensor_truth_banner.png)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/tensor-truth.svg)](https://pypi.org/project/tensor-truth/)
[![Docker Hub](https://img.shields.io/docker/v/ljubobratovicrelja/tensor-truth?label=docker)](https://hub.docker.com/r/ljubobratovicrelja/tensor-truth)
[![Tests](https://github.com/ljubobratovicrelja/tensor-truth/actions/workflows/tests.yml/badge.svg)](https://github.com/ljubobratovicrelja/tensor-truth/actions/workflows/tests.yml)


A local-first RAG application purpose-built for technical documentation and research papers. While many tools let you "chat with your PDFs," TensorTruth focuses on retrieval quality — using hierarchical chunking with auto-merging retrieval and cross-encoder reranking to get better answers from small local models.

It includes specialized ingestion for Sphinx/Doxygen API docs, arXiv papers (with metadata extraction), and textbooks (with TOC-based chapter splitting), alongside configurable chunking strategies tuned per content type. The result is a RAG pipeline that meaningfully reduces hallucinations on technical queries, even with 7-8B parameter models.

TensorTruth is designed as a single-user, local-first application — no authentication, no server deployment. This is intentional: the target audience is people running small local models on their own hardware (8-24GB VRAM), where multi-user serving isn't practical anyway. That said, if someone wants to propose multi-user support, the door is open — it's just not in the project scope at the moment.

## What to Expect

The core focus is **high-quality retrieval from large, complex technical knowledge bases** — getting better answers out of small local models by investing in the retrieval pipeline rather than the model size.

In practice, this means:
- **Hierarchical chunking with auto-merging retrieval** — documents are parsed into a multi-level node hierarchy. At query time, if enough child nodes from the same parent match, the system merges up to the parent for richer context. This avoids the "lost in the middle" problem of flat chunking while keeping retrieval precise.
- **Cross-encoder reranking** — retrieved chunks are reranked by a cross-encoder model (BGE-reranker-v2-m3 by default) before being sent to the LLM, significantly improving relevance over embedding similarity alone.
- **Format-aware document ingestion** — Sphinx and Doxygen API docs are crawled via their inventory files, preserving module structure. arXiv papers are fetched with full metadata. Textbooks are split by table-of-contents into chapters. This structure carries through to the index.
- **Configurable chunking strategies** — choose between hierarchical (fast, good for uniform docs), semantic (embedding-aware splits at natural boundaries), or semantic-hierarchical (two-pass: semantic split then hierarchical) per index. Tune chunk sizes, overlap, and breakpoint thresholds for your content type.

Beyond retrieval, TensorTruth also supports agentic orchestration, multiple LLM providers, project-based knowledge management, and a YAML extension system — but the retrieval pipeline is the core of what it does.

## Quick Start

Prepare a Python 3.11+ environment (recommended due to the large dependency tree):

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate(.ps1) on Windows CMD/PowerShell
```

Or via conda:

```bash
conda create -n tensor-truth python=3.11
conda activate tensor-truth
```

If using CUDA, install the appropriate PyTorch version first from [pytorch.org](https://pytorch.org/get-started/locally/) (tested with torch 2.9 + CUDA 12.8). Otherwise, pip will pull CPU-only PyTorch automatically.

```bash
pip install tensor-truth
```

You need at least one LLM provider. The simplest option is [Ollama](https://ollama.com/):
```bash
ollama serve
```

Run the app:
```bash
tensor-truth
```

On first launch, the web UI will offer to download pre-built indexes from HuggingFace Hub and guide you through connecting to your LLM provider.

## LLM Providers

TensorTruth supports three provider types. Configure them in `~/.tensortruth/config.yaml` under the `providers` section, or manage them through the web UI's settings panel (which includes auto-discovery for local servers).

### Ollama (default)

Works out of the box when Ollama is running locally. Models are discovered automatically.

```yaml
providers:
  - id: ollama
    type: ollama
    base_url: http://localhost:11434
```

### OpenAI-Compatible APIs

Connects to vLLM, Groq, Together AI, LocalAI, or any service that implements the OpenAI API. Requires a static model list since these endpoints don't always publish available models.

```yaml
providers:
  - id: groq
    type: openai_compatible
    base_url: https://api.groq.com/openai/v1
    api_key: ${GROQ_API_KEY}    # environment variable expansion
    models:
      - name: llama-3.3-70b-versatile
        display_name: Llama 3.3 70B
        capabilities: ["tools"]
        context_window: 131072
```

### llama.cpp

Connects to a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server, including router mode for serving multiple models. Models and capabilities are discovered automatically.

```yaml
providers:
  - id: llama-cpp
    type: llama_cpp
    base_url: http://localhost:8080
```

The web UI groups models by provider in the model selector, with separate sections for models that support agentic tool-calling and those that don't.

## Docker Deployment

A pre-built Docker image avoids manual CUDA/PyTorch setup (NVIDIA GPU and drivers still required):

```bash
docker pull ljubobratovicrelja/tensor-truth:latest

docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8000:8000 \
  -v ~/.tensortruth:/root/.tensortruth \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  ljubobratovicrelja/tensor-truth:latest
```

Access the app at **http://localhost:8000**

**See [docs/DOCKER.md](docs/DOCKER.md) for complete Docker documentation, troubleshooting, and advanced usage.**

## Projects

Projects let you organize documents and conversations into topic-specific knowledge bases. Create a project, upload documents, and start chatting with context scoped to that project's knowledge.

**Supported document types:**
- **PDF files** — uploaded and converted to markdown for indexing
- **Web URLs** — fetched and converted to markdown automatically
- **arXiv papers** — looked up by arXiv ID with automatic metadata extraction and PDF download
- **Text/Markdown files** — indexed directly

Each project maintains its own vector index, built incrementally as documents are added. Projects also support attaching modules from the global catalog (pre-built indexes) alongside project-specific documents.

Project configuration (model, parameters, active modules) is inherited by all chat sessions within that project, with per-session overrides available.

## Agentic Mode & Web Search

When **agentic mode** is enabled (the default for models that support tool-calling), every message is routed through an orchestrator agent that autonomously decides what to do. The agent loops internally — calling a tool, inspecting the result, deciding whether more information is needed, and repeating until it has a complete answer (up to 10 iterations).

The orchestrator has access to these tools:
- **`rag_query`** — search the indexed knowledge base with retrieval, reranking, and confidence scoring
- **`web_search`** — search the web via DuckDuckGo
- **`fetch_page`** — fetch a single web page and convert to markdown (supports Wikipedia, GitHub, arXiv, YouTube)
- **`fetch_pages_batch`** — fetch multiple URLs in parallel with content-stage reranking
- **MCP tools** — any tools from configured MCP servers or extensions

Just ask a question and the orchestrator figures out how to answer it:

```
What is flash attention and how does PyTorch implement it?
Compare Adam vs AdamW optimizer convergence properties
```

The orchestrator streams real-time progress to the UI — you see which tools are being called, what's being searched, and when synthesis begins. Sources from both RAG and web are combined and cited in the response.

Agentic mode relies on the model's native tool-calling ability. Larger models (`qwen3:8b`, `llama3.1:8b` and above) handle it well; smaller models tend to produce broken tool calls or loop unproductively. The toggle is per-session in the session settings panel. It is automatically disabled for models that don't support tool-calling (e.g., `deepseek-r1`), falling back to the direct RAG pipeline.

The `/web` slash command is still available for explicit web-only searches:

- **`/web <query>`** — Search the web (via DuckDuckGo), fetch top results, and generate an AI summary with sources. Supports optional instructions: `/web python 3.13;focus on performance improvements`.

## Custom Extensions

Add your own slash commands and agents by dropping YAML files into `~/.tensortruth/commands/` and `~/.tensortruth/agents/`. No code changes needed — just define a tool pipeline or agent config and restart.

```bash
mkdir -p ~/.tensortruth/commands
cp extension_library/commands/arxiv.yaml ~/.tensortruth/commands/
```

The repository includes ready-to-use extensions in the [`extension_library/`](extension_library/) directory — for example, [Context7](https://github.com/upstash/context7) integration for live library docs. arXiv search (`search_arxiv`, `get_arxiv_paper`) is built-in and needs no extension or MCP server; the extension library just provides optional `/arxiv` and `/arxiv_paper` slash command wrappers. For the full guide (YAML schema, template variables, Python extensions, MCP setup), see **[docs/EXTENSIONS.md](docs/EXTENSIONS.md)**.

## Data Storage

All user data (chat history, projects, indexes) is stored in `~/.tensortruth` on macOS/Linux or `%USERPROFILE%\.tensortruth` on Windows. This keeps your working directory clean while maintaining persistent state across sessions.

Pre-built indexes are hosted on [HuggingFace Hub](https://huggingface.co/datasets/ljubobratovicrelja/tensor-truth-indexes) and can be downloaded through the web UI on first launch.

For index contents, see [config/sources.json](config/sources.json). This is a curated list of useful libraries and research papers. Fork and customize as needed.

## Requirements

Tested on:
- MacBook M1 Max (32GB unified memory)
- Desktop with RTX 3090 Ti (24GB VRAM)
- ASUS Ascent DX10

If you encounter memory issues, consider running smaller models. Keep track of what models are loaded in Ollama, as they consume GPU VRAM and tend to stay in memory until Ollama is restarted.


### Recommended Models

Any model from a configured provider works. For Ollama, these offer a good balance of performance and capability with RAG:

**General Purpose (with tool-calling for agentic mode):**
```bash
ollama pull qwen3:8b           # Good tool-calling support
ollama pull llama3.1:8b        # Solid all-around
```

**Reasoning (no tool-calling — uses direct RAG pipeline):**
```bash
ollama pull deepseek-r1:8b     # Balanced
ollama pull deepseek-r1:14b    # More capable
```

Note that with RAG context in the prompt, VRAM requirements increase compared to plain chat. A 24GB GPU may struggle with 32B+ parameter models in RAG mode.

**Code/Technical Docs:**

```bash
ollama pull deepseek-coder-v2:16b   # Strong for code
ollama pull qwen2.5-coder:7b        # Smaller, holds up well with API docs
```

## Building Your Own Indexes

Pre-built indexes cover common libraries, but you can create custom knowledge bases for your specific needs.

### Quick Start

**Interactive Mode (Recommended):**
```bash
tensor-truth-docs --add    # Guided wizard for adding libraries, papers, or books
```

**Command-Line Mode:**
```bash
tensor-truth-docs --list                              # Show all available sources
tensor-truth-docs pytorch_2.9 numpy_2.3               # Fetch library documentation
tensor-truth-docs --type papers --category foundation_models --arxiv-ids 1706.03762  # Add specific papers
tensor-truth-build --modules foundation_models        # Build vector index
```

**In-App Document Upload:**

Upload documents directly in the web UI — either in a project or in a chat session. Supports PDF files, web URLs, arXiv papers (by ID), and plain text/markdown. Documents are converted and indexed automatically.

### Detailed Documentation

For comprehensive guides on building custom indexes, see [docs/INDEXES.md](docs/INDEXES.md), which covers:
- Interactive source addition workflow
- Adding libraries, papers, and books
- Chunk size optimization strategies
- Advanced workflows and troubleshooting

## Development

For frontend development, the React UI runs as a separate Vite dev server with hot-reload, proxying API calls to the backend:

```bash
# Terminal 1: Start the API server with auto-reload
tensor-truth --reload

# Terminal 2: Start the React dev server (port 5173)
tensor-truth-ui
```

The production `tensor-truth` command serves the bundled React frontend directly from port 8000 — no separate frontend process needed.

## Roadmap

The focus going forward is on retrieval quality while staying within consumer hardware budgets (8-24GB VRAM, running the LLM + embeddings + reranker on the same GPU):

- **Graph-enhanced retrieval** — integrating lightweight graph RAG approaches (LazyGraphRAG, SpaCy-based entity extraction) that don't require large models for indexing
- **Retrieval evaluation** — automated metrics (faithfulness, relevancy, hit rate) to guide chunking and pipeline tuning instead of relying on manual testing
- **Selective multimodal retrieval** — visual document retrieval for tables, diagrams, and equations where text-based chunking falls short, using models that fit consumer VRAM (e.g., ColPali at ~6GB)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer & Content Ownership

**1. Software License:**
The source code of `tensor-truth` is licensed under the MIT License. This covers the logic, UI, and retrieval pipelines created for this project.

**2. Third-Party Content:**
This tool is designed to fetch and index publicly available technical documentation, research papers (via ArXiv), and educational textbooks.
- **I do not own the rights to the indexed content.** All PDF files, textbooks, and research papers fetched by this tool remain the intellectual property of their respective authors and publishers.
- **Source Links:** The configuration files (`config/sources.json`, etc.) point exclusively to official sources, author-hosted pages, or open-access repositories (like ArXiv).
- **Usage:** This tool is intended for **personal, non-commercial research and educational use**.

**3. Takedown Request:**
If you are an author or copyright holder of any material referenced in the default configurations or included in the pre-built indexes and wish for it to be removed, please open an issue or contact me at ljubobratovic.relja@gmail.com, and the specific references/data will be removed immediately.
