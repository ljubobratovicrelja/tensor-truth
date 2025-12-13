# Tensor Truth

A local RAG (Retrieval-Augmented Generation) pipeline for reducing hallucinations in LLMs by indexing technical documentation and research papers. Built for personal use on local hardware, but shared in case others find it useful.

## What It Does

Tensor Truth indexes technical documentation and research papers into a vector database, then uses retrieval-augmented generation to ground LLM responses in accurate source material. This significantly reduces hallucination when working with technical topics.

The system uses hierarchical node parsing with auto-merging retrieval and cross-encoder reranking to maximize accuracy while staying within context windows.

## Quick Start

1. Install dependencies:
```bash
pip install -e .
```

2. Ensure Ollama is running:
```bash
ollama serve
```

3. Start the web interface:
```bash
streamlit run app.py
```

The application will automatically download pre-built indexes from Google Drive on first run. This may take a few minutes.

## Index Downloads

Pre-built indexes are hosted on Google Drive and auto-download on first launch. Note that Google Drive has rate limits to prevent abuse - if you hit the limit, you'll need to wait before retrying.

Manual download link: [Download Indexes](https://drive.google.com/file/d/1jILgN1ADgDgUt5EzkUnFMI8xwY2M_XTu/view?usp=sharing)

Extract to the `./indexes` directory in the project root.

## Building Your Own Databases

You can create custom knowledge bases using the included tools:

### Scrape Documentation

Index official library documentation (PyTorch, NumPy, OpenCV, etc.):

```bash
python -m tensortruth.scrape_docs --config library_docs.json pytorch
```

See available libraries:
```bash
python -m tensortruth.scrape_docs --list
```

### Fetch Research Papers

Add arXiv papers to your knowledge base:

```bash
python -m tensortruth.fetch_paper --config ./config/papers.json --category your_category --ids 2301.12345
```

Rebuild all papers in a category:
```bash
python -m tensortruth.fetch_paper --rebuild your_category
```

### Build Vector Indexes

After adding documents, build the vector index:

```bash
python -m tensortruth.build_db module_name
```

## Requirements

- Python 3.13+
- Ollama running locally
- 16GB+ RAM recommended (8GB minimum)
- GPU optional but recommended for faster inference

### Recommended Ollama Models

The system works with any Ollama model, but these are tested and recommended:

**General Purpose (with reasoning):**
```bash
# Lightweight (8GB RAM)
ollama pull deepseek-r1:1.5b

# Balanced (16GB RAM)
ollama pull deepseek-r1:8b

# High quality (24GB+ RAM/VRAM)
ollama pull deepseek-r1:14b
ollama pull deepseek-r1:32b
```

**Code-Focused (technical documentation):**
```bash
# Balanced (16GB RAM)
ollama pull deepseek-coder-v2:16b

# High quality (24GB+ RAM/VRAM)
ollama pull deepseek-coder-v2
```

The DeepSeek-R1 models include chain-of-thought reasoning. The DeepSeek-Coder-V2 variants are optimized for code and technical content, which works particularly well when querying programming documentation.

## Configuration Notes

This RAG system is configured for personal research workflows. Key assumptions:

- Indexes are pre-built for speed
- Models run via Ollama (local inference)
- Vector store uses ChromaDB (persistent, single-process)
- Embeddings via HuggingFace sentence-transformers
- Reranking with BGE cross-encoder models

If you want different behavior, you'll need to modify the configuration in the source files.

## License

MIT License - see [LICENSE](LICENSE) file for details.

This project is provided as-is with no warranty. Built for personal use but released publicly in case others find it useful.
