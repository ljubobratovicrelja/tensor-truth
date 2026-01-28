"""Chat service for unified query routing.

Encapsulates mode detection and engine management, providing
a unified interface for chat endpoints.
"""

from typing import Any, Dict, Generator, List, Optional

from tensortruth.services.models import RAGChunk, RAGResponse
from tensortruth.services.rag_service import RAGService


class ChatService:
    """Routes chat queries to appropriate backend (LLM-only or RAG).

    Encapsulates mode detection and engine management, providing
    a unified interface for chat endpoints.
    """

    def __init__(self, rag_service: RAGService):
        self._rag_service = rag_service

    def query(
        self,
        prompt: str,
        modules: List[str],
        params: Dict[str, Any],
        session_messages: Optional[List[Dict[str, Any]]] = None,
        session_index_path: Optional[str] = None,
    ) -> Generator[RAGChunk, None, RAGResponse]:
        """Execute chat query with automatic mode routing.

        Routes to LLM-only when no modules/PDFs, otherwise RAG mode.
        Handles engine reload internally, yielding "loading_models" status
        if reload is needed.

        Args:
            prompt: User's query prompt.
            modules: List of module names for RAG retrieval.
            params: Engine parameters (model, temperature, etc).
            session_messages: Chat history from session storage.
            session_index_path: Optional path to session-specific PDF index.

        Yields:
            RAGChunk with status, thinking, or text content.

        Returns:
            Final RAGResponse with complete text and sources.
        """
        is_llm_only = self.is_llm_only_mode(modules, session_index_path)

        if is_llm_only:
            return (
                yield from self._rag_service.query_llm_only(
                    prompt, params, session_messages=session_messages
                )
            )

        # RAG mode: ensure engine is loaded
        if self._rag_service.needs_reload(modules, params, session_index_path):
            yield RAGChunk(status="loading_models")
            self._rag_service.load_engine(
                modules=modules,
                params=params,
                session_index_path=session_index_path,
            )

        return (
            yield from self._rag_service.query(
                prompt, session_messages=session_messages
            )
        )

    def is_llm_only_mode(
        self,
        modules: List[str],
        session_index_path: Optional[str],
    ) -> bool:
        """Check if configuration results in LLM-only mode.

        Args:
            modules: List of module names.
            session_index_path: Optional path to session-specific PDF index.

        Returns:
            True if no modules and no PDF index (LLM-only mode).
        """
        return not modules and not session_index_path
