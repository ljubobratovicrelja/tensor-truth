"""
Tensor-Truth: Local RAG Pipeline for Technical Documentation

A modular framework for building Retrieval-Augmented Generation (RAG) pipelines
running entirely on local hardware.
"""

from tensortruth.build_db import build_module
from tensortruth.core import get_max_memory_gb, get_running_models, stop_model
from tensortruth.fetch_paper import (
    book_already_processed,
    fetch_and_convert_book,
    fetch_and_convert_paper,
    paper_already_processed,
)
from tensortruth.rag_engine import (
    MultiIndexRetriever,
    get_embed_model,
    get_llm,
    get_reranker,
    load_engine_for_modules,
)
from tensortruth.utils import (
    convert_chat_to_markdown,
    parse_thinking_response,
    run_ingestion,
)

__version__ = "0.1.0"

__all__ = [
    # RAG Engine
    "load_engine_for_modules",
    "get_embed_model",
    "get_llm",
    "get_reranker",
    "MultiIndexRetriever",
    # Utils (Core)
    "parse_thinking_response",
    "run_ingestion",
    "convert_chat_to_markdown",
    # Core (System & Ollama)
    "get_running_models",
    "get_max_memory_gb",
    "stop_model",
    # Database Building
    "build_module",
    # Paper Fetching
    "fetch_and_convert_paper",
    "paper_already_processed",
    "fetch_and_convert_book",
    "book_already_processed",
]
