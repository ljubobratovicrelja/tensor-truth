"""Core utilities for Tensor-Truth."""

from .ollama import (
    get_available_models,
    get_ollama_url,
    get_running_models,
    get_running_models_detailed,
    stop_model,
)
from .source import SourceNode, SourceStatus, SourceType
from .source_pipeline import SourceFetchPipeline
from .synthesis import (
    CitationStyle,
    QueryType,
    SynthesisConfig,
    detect_query_type,
    get_model_prompt_config,
    synthesize_with_llm_stream,
)
from .system import get_max_memory_gb
from .types import DocType, DocumentType
from .types import SourceType as ConfigSourceType

__all__ = [
    "get_available_models",
    "get_ollama_url",
    "get_running_models",
    "get_running_models_detailed",
    "stop_model",
    "get_max_memory_gb",
    "DocType",
    "DocumentType",
    "ConfigSourceType",
    "SourceNode",
    "SourceStatus",
    "SourceType",
    "SourceFetchPipeline",
    "CitationStyle",
    "QueryType",
    "SynthesisConfig",
    "detect_query_type",
    "get_model_prompt_config",
    "synthesize_with_llm_stream",
]
