"""RAG (Retrieval-Augmented Generation) service.

This service wraps the existing rag_engine module with lifecycle management
and a cleaner interface for the UI layer.
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.ollama import Ollama

from tensortruth.app_utils.config_schema import TensorTruthConfig
from tensortruth.rag_engine import (
    MultiIndexRetriever,
    get_base_index_dir,
    get_llm,
    load_engine_for_modules,
)

from .models import RAGChunk, RAGResponse


class RAGService:
    """Service for RAG query operations with lifecycle management.

    Manages the chat engine instance, handling loading, reloading,
    and cleanup of GPU resources.
    """

    def __init__(
        self,
        config: Optional[TensorTruthConfig] = None,
        indexes_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize RAG service.

        Args:
            config: TensorTruth configuration. If None, loads from default.
            indexes_dir: Base directory for vector indexes.
        """
        if config is None:
            from tensortruth.app_utils.config import load_config

            config = load_config()

        self.config = config
        self.indexes_dir = Path(indexes_dir) if indexes_dir else Path(get_base_index_dir())

        self._engine: Optional[CondensePlusContextChatEngine] = None
        self._current_config_hash: Optional[Tuple] = None
        self._current_modules: Optional[List[str]] = None
        self._current_params: Optional[Dict[str, Any]] = None

    def _compute_config_hash(
        self,
        modules: Optional[List[str]],
        params: Dict[str, Any],
        session_index_path: Optional[str] = None,
    ) -> Optional[Tuple]:
        """Compute configuration hash for cache invalidation.

        Args:
            modules: List of module names.
            params: Engine parameters.
            session_index_path: Optional session index path.

        Returns:
            Hashable tuple for comparison.
        """
        modules_tuple = tuple(sorted(modules)) if modules else None
        param_items = sorted([(k, v) for k, v in params.items()])
        param_hash = frozenset(param_items)
        has_session_index = bool(session_index_path)

        if modules_tuple or has_session_index:
            return (modules_tuple, param_hash, has_session_index)
        return None

    def load_engine(
        self,
        modules: List[str],
        params: Dict[str, Any],
        session_index_path: Optional[str] = None,
        chat_history: Optional[List] = None,
    ) -> None:
        """Load or reload the RAG engine with specified configuration.

        Args:
            modules: List of module names to load.
            params: Engine parameters (model, temperature, etc).
            session_index_path: Optional path to session-specific PDF index.
            chat_history: Optional chat history to restore.
        """
        # Clear existing engine first to free GPU memory
        self.clear()

        self._engine = load_engine_for_modules(
            selected_modules=modules,
            engine_params=params,
            preserved_chat_history=chat_history,
            session_index_path=session_index_path,
        )

        self._current_modules = modules
        self._current_params = params
        self._current_config_hash = self._compute_config_hash(
            modules, params, session_index_path
        )

    def needs_reload(
        self,
        modules: List[str],
        params: Dict[str, Any],
        session_index_path: Optional[str] = None,
    ) -> bool:
        """Check if engine needs to be reloaded due to config changes.

        Args:
            modules: List of module names.
            params: Engine parameters.
            session_index_path: Optional session index path.

        Returns:
            True if engine should be reloaded.
        """
        if self._engine is None:
            return True

        new_hash = self._compute_config_hash(modules, params, session_index_path)
        return new_hash != self._current_config_hash

    def is_loaded(self) -> bool:
        """Check if an engine is currently loaded.

        Returns:
            True if engine is loaded.
        """
        return self._engine is not None

    def query(self, prompt: str) -> Generator[RAGChunk, None, RAGResponse]:
        """Execute a streaming RAG query.

        Args:
            prompt: User's query prompt.

        Yields:
            RAGChunk with partial response text.

        Returns:
            Final RAGResponse with complete text and sources.

        Raises:
            RuntimeError: If engine is not loaded.
        """
        if self._engine is None:
            raise RuntimeError("RAG engine not loaded. Call load_engine() first.")

        response_stream = self._engine.stream_chat(prompt)

        full_response = ""
        source_nodes = []

        for token in response_stream.response_gen:
            full_response += token
            yield RAGChunk(text=token, is_complete=False)

        # Get source nodes from the response
        if hasattr(response_stream, "source_nodes"):
            source_nodes = response_stream.source_nodes

        # Yield final complete chunk
        yield RAGChunk(text="", source_nodes=source_nodes, is_complete=True)

        return RAGResponse(text=full_response, source_nodes=source_nodes)

    def query_simple(
        self,
        prompt: str,
        chat_history: Optional[List] = None,
    ) -> Generator[str, None, str]:
        """Execute a streaming query without source tracking.

        Simpler interface for basic chat without RAG metadata.

        Args:
            prompt: User's query prompt.
            chat_history: Optional chat history context.

        Yields:
            Response tokens.

        Returns:
            Complete response text.

        Raises:
            RuntimeError: If engine is not loaded.
        """
        if self._engine is None:
            raise RuntimeError("RAG engine not loaded. Call load_engine() first.")

        response_stream = self._engine.stream_chat(prompt)

        full_response = ""
        for token in response_stream.response_gen:
            full_response += token
            yield token

        return full_response

    def get_llm(self) -> Optional[Ollama]:
        """Get the underlying LLM instance from the engine.

        Useful for intent classification reuse.

        Returns:
            Ollama LLM instance or None if engine not loaded.
        """
        if self._engine is None:
            return None
        return self._engine._llm

    def get_llm_from_params(self, params: Dict[str, Any]) -> Ollama:
        """Create an LLM instance from parameters without loading full engine.

        Useful for operations that only need LLM (like intent classification).

        Args:
            params: Engine parameters.

        Returns:
            Ollama LLM instance.
        """
        return get_llm(params)

    def get_chat_history(self) -> List:
        """Get the current chat history from the engine memory.

        Returns:
            List of chat messages or empty list if engine not loaded.
        """
        if self._engine is None:
            return []

        try:
            return self._engine.memory.get_all()
        except (AttributeError, TypeError):
            return []

    def clear(self) -> None:
        """Clear the engine and free GPU memory.

        Should be called before loading a new engine or when done.
        """
        if self._engine is not None:
            # Clear retriever cache to release GPU tensor references
            if hasattr(self._engine, "_retriever"):
                retriever = self._engine._retriever
                if isinstance(retriever, MultiIndexRetriever):
                    retriever.clear_cache()

            # Clear chat memory
            if hasattr(self._engine, "memory"):
                self._engine.memory.reset()

            self._engine = None
            self._current_config_hash = None
            self._current_modules = None
            self._current_params = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.clear()
        return False
