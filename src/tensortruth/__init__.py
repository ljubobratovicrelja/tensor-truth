"""
Tensor-Truth: Local RAG Pipeline for Technical Documentation

A modular framework for building Retrieval-Augmented Generation (RAG) pipelines
running entirely on local hardware.
"""

__version__ = "0.1.0"

# Lazy imports to avoid pulling in heavy dependencies unless actually needed
def __getattr__(name):
    """Lazy import implementation for better testing and startup time."""

    # RAG Engine exports
    if name in ("load_engine_for_modules", "get_embed_model", "get_llm", "get_reranker", "MultiIndexRetriever"):
        from tensortruth.rag_engine import (
            load_engine_for_modules,
            get_embed_model,
            get_llm,
            get_reranker,
            MultiIndexRetriever,
        )
        return locals()[name]

    # Utils exports
    if name in ("parse_thinking_response", "run_ingestion", "convert_chat_to_markdown",
                "get_running_models", "get_max_memory_gb", "download_and_extract_indexes", "stop_model"):
        from tensortruth.utils import (
            parse_thinking_response,
            run_ingestion,
            convert_chat_to_markdown,
            get_running_models,
            get_max_memory_gb,
            download_and_extract_indexes,
            stop_model,
        )
        return locals()[name]

    # Database Building exports
    if name == "build_module":
        from tensortruth.build_db import build_module
        return build_module

    # Paper Fetching exports
    if name in ("fetch_and_convert_paper", "paper_already_processed",
                "fetch_and_convert_book", "book_already_processed"):
        from tensortruth.fetch_paper import (
            fetch_and_convert_paper,
            paper_already_processed,
            fetch_and_convert_book,
            book_already_processed,
        )
        return locals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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
