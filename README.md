# Tensor Truth

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/tensor-truth.svg)](https://pypi.org/project/tensor-truth/)
[![Tests](https://github.com/ljubobratovicrelja/tensor-truth/actions/workflows/tests.yml/badge.svg)](https://github.com/ljubobratovicrelja/tensor-truth/actions/workflows/tests.yml)

A local RAG pipeline for reducing hallucinations in LLMs by indexing technical documentation and research papers. Built for personal use on local hardware, shared here in case others find it useful. Web UI is built with Streamlit, with high level of configurability for the pipeline.

## What It Does

Indexes technical documentation and research papers into vector databases, then uses retrieval-augmented generation to ground LLM responses in source material. Uses hierarchical node parsing with auto-merging retrieval and cross-encoder reranking to balance accuracy and context window constraints.

## Quick Start

Install the tool via PyPI. But before you do, I advise you prep the environment because of large volume of dependencies (use Python 3.13+):

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate(.ps1) on Windows CMD/PowerShell
```

Or via conda:

```bash
conda create -n tensor-truth python=3.13
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

## Index Downloads

Pre-built indexes download automatically on startup. Note that Google Drive has rate limits, so if it refuses to download, try manually from [indexes.tar](https://drive.google.com/file/d/1jILgN1ADgDgUt5EzkUnFMI8xwY2M_XTu/view?usp=sharing).

Extract to `./indexes` in the project root.

For details on the contents of this archive, see [config/api.json](config/api.json) and [config/papers.json](config/papers.json). These are my curated lists of useful libraries and research papers. Feel free to fork and set up your own indexes. See below instructions on how to build the indexes.

## Requirements

Tested on:
- MacBook M1 Max (32GB unified memory)
- Desktop with RTX 3090 Ti (24GB VRAM)

Minimum recommended: 16GB RAM, Python 3.13+. GPU optional but significantly faster.

### Recommended Models

Any Ollama model works, but these are tested:

**General Purpose:**
```bash
ollama pull deepseek-r1:8b     # Balanced
ollama pull deepseek-r1:14b    # High quality
ollama pull deepseek-r1:32b    # Best quality (24GB+)
```

**Code/Technical Docs:**
```bash
ollama pull deepseek-coder-v2:16b
ollama pull deepseek-coder-v2
```

DeepSeek-R1 models include chain-of-thought reasoning. Coder-V2 variants are optimized for technical content and work particularly well with programming documentation.

## Building Your Own Indexes

Pre-built indexes cover common libraries, but you can create custom knowledge bases:

**Scrape Documentation:**
```bash
tensor-truth-docs --list          # Show available libraries
tensor-truth-docs pytorch         # Scrape PyTorch docs
```

**Fetch Research Papers:**
```bash
tensor-truth-papers --config ./config/papers.json --category your_category --ids 2301.12345
tensor-truth-papers --rebuild your_category
```

**Build Vector Index:**
```bash
tensor-truth-build --modules module_name
```

## Configuration

This system is configured for personal research workflows with these assumptions:

- ChromaDB for vector storage (persistent, single-process)
- HuggingFace sentence-transformers for embeddings
- BGE cross-encoder models for reranking
- Ollama for local LLM inference
- All processing runs locally

If you need different chunking strategies or retrieval parameters, you'll need to modify the source files. The current setup is tuned for technical documentation and research papers.

## License

MIT License - see [LICENSE](LICENSE) for details.

Built for personal use but released publicly. Provided as-is with no warranty.
