![logo](https://raw.githubusercontent.com/ljubobratovicrelja/tensor-truth/main/media/tensor_truth_banner.png)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/tensor-truth.svg)](https://pypi.org/project/tensor-truth/)
[![Docker Hub](https://img.shields.io/docker/v/ljubobratovicrelja/tensor-truth?label=docker)](https://hub.docker.com/r/ljubobratovicrelja/tensor-truth)
[![Tests](https://github.com/ljubobratovicrelja/tensor-truth/actions/workflows/tests.yml/badge.svg)](https://github.com/ljubobratovicrelja/tensor-truth/actions/workflows/tests.yml)


A local RAG pipeline for reducing hallucinations in LLMs by indexing technical documentation and research papers. Built for personal use on local hardware, shared here in case others find it useful. Web UI is built with Streamlit, with high level of configurability for the pipeline.

> **Note:** For the moment, this is very much a hobby project. The app has no authentication or multi-user support and is designed to run locally on your own machine. If there's interest in production-ready deployment features, I can add them (feel free to make a request via issues).

## What It Does

Indexes technical documentation and research papers into vector databases, then uses retrieval-augmented generation to ground LLM responses in source material. Uses hierarchical node parsing with auto-merging retrieval and cross-encoder reranking to balance accuracy and context window constraints.

## Quick Start

Install the tool via PyPI. But before you do, I advise you prep the environment because of large volume of dependencies (use Python 3.11+):

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate(.ps1) on Windows CMD/PowerShell
```

Or via conda:

```bash
conda create -n tensor-truth python=3.11
conda activate tensor-truth
```

If using CUDA, make sure to first install the appropriate PyTorch version from [pytorch.org](https://pytorch.org/get-started/locally/). I used torch 2.9 and CUDA 12.8 in environments with CUDA.

If not, just install tensor-truth via pip, which includes CPU-only PyTorch.

```bash
pip install tensor-truth
```

Make sure [ollama](https://ollama.com/) is installed and set up. Start the server:
```bash
ollama serve
```

Run the app:
```bash
tensor-truth
```

On first launch, pre-built indexes will auto-download from Google Drive (takes a few minutes). Also a small qwen2.5:0.5b will be pulled automatically for assigning automatic titles to chats.

## Docker Deployment

For easier deployment without managing virtual environments or CUDA installations, a pre-built Docker image is available on Docker Hub. This approach is useful if you want to avoid setting up PyTorch with CUDA manually, though you still need a machine with NVIDIA GPU and drivers installed.

**Pull the image:**
```bash
docker pull ljubobratovicrelja/tensor-truth:latest
```

**Run the container:**
```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8501:8501 \
  -v ~/.tensortruth:/root/.tensortruth \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  ljubobratovicrelja/tensor-truth:latest
```

Access the app at **http://localhost:8501**

**See [DOCKER.md](docs/DOCKER.md) for complete Docker documentation, troubleshooting, and advanced usage.**


## Data Storage

All user data (chat history, presets, indexes) is stored in `~/.tensortruth` on macOS/Linux or `%USERPROFILE%\.tensortruth` on Windows. This keeps your working directory clean while maintaining persistent state across sessions.

Pre-built indexes download automatically to this directory on startup. If Google Drive rate limits prevent auto-download, manually fetch [indexes.tar](https://drive.google.com/file/d/12wZsBwrywl9nXOCLr50lpWB2SiFdu1XB/view?usp=sharing) and extract to `~/.tensortruth/indexes`.

For index contents, see [config/sources.json](config/sources.json). This is a curated list of useful libraries and research papers. Fork and customize as needed.

## Requirements

Tested on:
- MacBook M1 Max (32GB unified memory)
- Desktop with RTX 3090 Ti (24GB VRAM)

If you encounter memory issues, consider running smaller models. Also keep track of what models are loaded in Ollama, as they consume GPU VRAM, and tend to stuck in memory until Ollama is restarted.


### Recommended Models

Any Ollama model works, but I recommend these for best balance of performance and capability with RAG:

**General Purpose:**
```bash
ollama pull deepseek-r1:8b     # Balanced
ollama pull deepseek-r1:14b    # More capable
```
Note that, even though pure Ollama can run deepseek-r1:32b, with RAG workflow it is likely to struggle on 24GB 3090 for e.g.

**Code/Technical Docs:**

For coding, deepseek-coder-v2 is a strong choice:
```bash
ollama pull deepseek-coder-v2:16b 
```
Or, the smaller qwen2.5-coder, holds up well with API docs on coding aid.
```bash
ollama pull qwen2.5-coder:7b 
````

## Building Your Own Indexes

Pre-built indexes cover common libraries, but you can create custom knowledge bases.

**Scrape Documentation:**
```bash
tensor-truth-docs --list                              # Show all available sources
tensor-truth-docs pytorch_2.9 numpy_2.3               # Scrape library docs
tensor-truth-docs --type papers --category foundation_models   # Fetch paper category
tensor-truth-docs --type papers --category foundation_models --ids 1706.03762 1810.04805  # Add specific papers
```

**Build Vector Index:**
```bash
tensor-truth-build --modules foundation_models
```

See detailed documentation and examples in [INDEXES.md](docs/INDEXES.md).

**Session PDFs:**

Upload PDFs directly in the web UI to create per-session indexes. For now only standard PDF files are supported, but more formats may be added later.

## License

MIT License - see [LICENSE](LICENSE) for details.

Built for personal use but released publicly. Provided as-is with no warranty.

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