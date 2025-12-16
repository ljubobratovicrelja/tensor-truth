"""
Tensor-Truth: Local RAG Pipeline for Technical Documentation

A modular framework for building Retrieval-Augmented Generation (RAG) pipelines
running entirely on local hardware.
"""

from tensortruth.core import get_max_memory_gb, get_running_models, stop_model
from tensortruth.rag_engine import (
    CUSTOM_CONTEXT_PROMPT_NO_SOURCES,
    NO_CONTEXT_FALLBACK_CONTEXT,
    MultiIndexRetriever,
    get_base_index_dir,
    get_embed_model,
    get_llm,
    get_reranker,
    load_engine_for_modules,
)
from tensortruth.utils import (
    convert_chat_to_markdown,
    convert_latex_delimiters,
    parse_thinking_response,
)

__version__ = "0.1.0"

__all__ = [
    # RAG Engine
    "load_engine_for_modules",
    "get_base_index_dir",
    "get_embed_model",
    "get_llm",
    "get_reranker",
    "MultiIndexRetriever",
    "NO_CONTEXT_FALLBACK_CONTEXT",
    "CUSTOM_CONTEXT_PROMPT_NO_SOURCES",
    # Utils (Core)
    "parse_thinking_response",
    "convert_chat_to_markdown",
    "convert_latex_delimiters",
    # Core (System & Ollama)
    "get_running_models",
    "get_max_memory_gb",
    "stop_model",
]
