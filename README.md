# Tensor Truth

## Overview

This project is a modular framework for building Retrieval-Augmented Generation (RAG) pipelines running entirely on local hardware (RTX 3090Ti).

**Primary Goal:** Reduce hallucination in local LLMs (specifically DeepSeek-R1-Distill versions) by indexing technical documentation (PyTorch, NumPy, etc.) with high precision.

**Core Mechanics:**

  * **Orchestration:** LlamaIndex.
  * **Inference:** Ollama (serving GGUF/ExLlamaV2 models).
  * **Vector Store:** ChromaDB (Persistent).
  * **Retrieval Strategy:** Hierarchical Node Parsing + Auto-Merging Retriever + Cross-Encoder Reranking.

## Architecture

The pipeline uses a "Small-to-Big" retrieval strategy to maximize context window efficiency while maintaining retrieval accuracy.

1.  **Ingestion:** Documents are parsed into parent nodes (2048 tokens) and child nodes (128 tokens).
2.  **Indexing:** Only child nodes are embedded and stored in ChromaDB. Parent nodes are stored in a DocStore key-value store.
3.  **Retrieval:** Query matches child nodes. If enough children of a specific parent are found, they are merged into the parent node.
4.  **Reranking:** Top-k retrieved contexts are re-scored by a Cross-Encoder (BGE-Reranker) before LLM generation.

## Prerequisites

  * **Hardware:** NVIDIA GPU with 24GB+ VRAM recommended (RTX 3090/4090).
  * **System:** Linux/WSL2 or Windows with CUDA toolkit installed.
  * **Ollama:** Must be running as a background service (`ollama serve`).

## Installation

1. **Environment (venv)**
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/WSL)
source venv/bin/activate

# Activate (Windows PowerShell)
.\venv\Scripts\activate
```

2.  **Dependencies**

```bash
pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-vector-stores-chroma llama-index-readers-file chromadb torch transformers sentence-transformers
```

3.  **Model Pull (Ollama)**

```bash
ollama pull deepseek-r1:32b
```

## Usage

### 1\. Directory Structure

The script expects a flat directory or subdirectory structure for documents.

```text
.
├── chroma_db/              # Generated vector store persistence
├── library_docs/           # DROP RAW FILES HERE (.md, .txt)
├── src/
│   └── pipeline.py         # Main entry point
└── README.md
```

### 2\. Configuration

Modify `src/pipeline.py` globals to switch contexts or models.

```python
# Configuration Constants
DOCS_DIR = "./library_docs/pytorch_v2"  # Source for specific library
COLLECTION_NAME = "pytorch_index"       # ChromaDB collection namespace
LLM_MODEL = "deepseek-r1:32b"           # Ollama model tag
```

### 3\. Execution

Run the pipeline. It handles both indexing (if new data) and querying.

```bash
python src/pipeline.py
```

## Strategy & Customization

### Adding New Contexts (e.g., Internal Codebase)

To index a different dataset without overwriting the previous one:

1.  Change `DOCS_DIR` to point to the new files.
2.  Change `COLLECTION_NAME` to a unique string (e.g., `internal_legacy_code`).
3.  Run the script. ChromaDB will create a new collection alongside the existing ones.

### Modifying Chunking

Adjust `HierarchicalNodeParser` in `build_or_load_index` for different data types.

  * **Code/API Docs:** Use smaller leaf nodes (128 tokens) to isolate function signatures.
  * **Prose/Wiki:** Increase leaf nodes (256-512 tokens) to capture semantic meaning.

### GPU VRAM Management

If OOM errors occur during ingestion or heavy query loads:

1.  **Offload Embeddings:** Set `device="cpu"` in `HuggingFaceEmbedding`.
2.  **Limit Ollama:** Reduce `num_gpu` layers in Ollama or use a smaller quant (Q4\_K\_M).

## Technical Notes

  * **Persistency:** The `chroma_db` folder contains the vector embeddings. The `storage_context` (DocStore) is saved alongside it. **Do not delete this folder** unless you want to re-index everything.
  * **Reranker:** Currently uses `BAAI/bge-reranker-v2-m3`. It is computationally expensive but necessary for code disambiguation.
  * **Locking:** ChromaDB is single-threaded/process locked. Ensure only one script instance accesses the DB at a time.