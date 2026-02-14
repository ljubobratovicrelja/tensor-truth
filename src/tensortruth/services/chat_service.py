"""Chat service for unified query routing.

Encapsulates mode detection and engine management, providing
a unified interface for chat endpoints.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from tensortruth.core.source_converter import SourceConverter
from tensortruth.services.models import RAGChunk, RAGResponse
from tensortruth.services.rag_service import RAGService


@dataclass
class ChatResult:
    """Result of a non-streaming chat query.

    Contains the complete response with sources converted to API-compatible format.
    Foreign types (LlamaIndex NodeWithScore) are contained within ChatService.
    """

    response: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None


class ChatService:
    """Unified chat query interface with engine management and source conversion.

    Foreign types (LlamaIndex NodeWithScore) are contained within this service.
    External callers receive only our own API-compatible types.
    """

    def __init__(self, rag_service: RAGService):
        self._rag_service = rag_service

    def execute(
        self,
        prompt: str,
        modules: List[str],
        params: Dict[str, Any],
        session_messages: Optional[List[Dict[str, Any]]] = None,
        additional_index_paths: Optional[List[str]] = None,
    ) -> ChatResult:
        """Execute chat query and return complete result.

        Non-streaming interface for REST endpoints. Consumes the streaming
        generator internally and returns clean API types.

        Args:
            prompt: User's query prompt.
            modules: List of module names for RAG retrieval.
            params: Engine parameters (model, temperature, etc).
            session_messages: Chat history from session storage.
            additional_index_paths: Optional list of additional index paths
                (session PDFs, project indexes).

        Returns:
            ChatResult with response text, sources (as API dicts), and metrics.
        """
        full_response = ""
        sources: List[Dict[str, Any]] = []
        metrics: Optional[Dict[str, Any]] = None

        for chunk in self.query(
            prompt=prompt,
            modules=modules,
            params=params,
            session_messages=session_messages,
            additional_index_paths=additional_index_paths,
        ):
            if chunk.is_complete:
                sources = self._extract_sources(chunk.source_nodes)
                metrics = chunk.metrics
            elif chunk.text:
                full_response += chunk.text

        return ChatResult(
            response=full_response,
            sources=sources,
            metrics=metrics,
        )

    def query(
        self,
        prompt: str,
        modules: List[str],
        params: Dict[str, Any],
        session_messages: Optional[List[Dict[str, Any]]] = None,
        additional_index_paths: Optional[List[str]] = None,
    ) -> Generator[RAGChunk, None, RAGResponse]:
        """Execute chat query through the unified pipeline (streaming).

        Handles engine reload internally, yielding "loading_models" status
        if reload is needed. When no modules/PDFs exist, the RAG service
        skips retrieval and uses LLM-only prompting.

        Note: The final RAGChunk contains source_nodes as LlamaIndex types.
        Use execute() for a non-streaming interface with clean API types,
        or call extract_sources() on the final chunk's source_nodes.

        Args:
            prompt: User's query prompt.
            modules: List of module names for RAG retrieval.
            params: Engine parameters (model, temperature, etc).
            session_messages: Chat history from session storage.
            additional_index_paths: Optional list of additional index paths
                (session PDFs, project indexes).

        Yields:
            RAGChunk with status, thinking, or text content.

        Returns:
            Final RAGResponse with complete text and sources.
        """
        if self._rag_service.needs_reload(modules, params, additional_index_paths):
            yield RAGChunk(status="loading_models")
            self._rag_service.load_engine(
                modules=modules,
                params=params,
                additional_index_paths=additional_index_paths,
            )

        return (
            yield from self._rag_service.query(
                prompt, params, session_messages=session_messages
            )
        )

    def extract_sources(self, source_nodes: List[Any]) -> List[Dict[str, Any]]:
        """Convert RAG source nodes to API-compatible format.

        Public method for streaming callers (WebSocket) who need to convert
        source nodes from the final RAGChunk.

        Args:
            source_nodes: LlamaIndex NodeWithScore objects from retriever.

        Returns:
            List of dicts matching SourceNode API schema.
        """
        return self._extract_sources(source_nodes)

    def _extract_sources(self, source_nodes: List[Any]) -> List[Dict[str, Any]]:
        """Internal source extraction using SourceConverter.

        Contains foreign type handling (LlamaIndex NodeWithScore).

        Args:
            source_nodes: LlamaIndex NodeWithScore objects from retriever.

        Returns:
            List of dicts matching SourceNode API schema.
        """
        sources = []
        for idx, node in enumerate(source_nodes):
            unified = SourceConverter.from_rag_node(node, idx)
            api_dict = SourceConverter.to_api_schema(unified)
            sources.append(api_dict)
        return sources
