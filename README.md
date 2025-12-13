# Tensor Truth

A local RAG pipeline for reducing hallucinations in LLMs by indexing technical documentation and research papers. Built for personal use on local hardware, shared here in case others find it useful.

## What It Does

Indexes technical documentation and research papers into vector databases, then uses retrieval-augmented generation to ground LLM responses in source material. Uses hierarchical node parsing with auto-merging retrieval and cross-encoder reranking to balance accuracy and context window constraints.

## Quick Start

Install dependencies:
```bash
pip install -e .
```

Start Ollama:
```bash
ollama serve
```

Run the app:
```bash
tensor-truth
```

On first launch, pre-built indexes will auto-download from Google Drive (takes a few minutes).

## Index Downloads

Pre-built indexes download automatically on startup. Note that Google Drive has rate limits - if you hit the limit, wait a bit before retrying.

Manual download if needed: [indexes.tar](https://drive.google.com/file/d/1jILgN1ADgDgUt5EzkUnFMI8xwY2M_XTu/view?usp=sharing)

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
