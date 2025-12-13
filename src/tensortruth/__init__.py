"""
Tensor-Truth: Local RAG Pipeline for Technical Documentation

A modular framework for building Retrieval-Augmented Generation (RAG) pipelines
running entirely on local hardware.
"""

from tensortruth.rag_engine import (
    load_engine_for_modules,
    get_embed_model,
    get_llm,
    get_reranker,
    MultiIndexRetriever,
)

from tensortruth.utils import (
    parse_thinking_response,
    run_ingestion,
    convert_chat_to_markdown,
    get_running_models,
    get_max_memory_gb,
    download_and_extract_indexes,
    stop_model,
)

from tensortruth.build_db import build_module

from tensortruth.fetch_paper import (
    fetch_and_convert_paper,
    paper_already_processed,
    fetch_and_convert_book,
    book_already_processed,
)

__version__ = "0.1.0"

__all__ = [
    # RAG Engine
    "load_engine_for_modules",
    "get_embed_model",
    "get_llm",
    "get_reranker",
    "MultiIndexRetriever",
    # Utils
    "parse_thinking_response",
    "run_ingestion",
    "convert_chat_to_markdown",
    "get_running_models",
    "get_max_memory_gb",
    "download_and_extract_indexes",
    "stop_model",
    # Database Building
    "build_module",
    # Paper Fetching
    "fetch_and_convert_paper",
    "paper_already_processed",
    "fetch_and_convert_book",
    "book_already_processed",
]
