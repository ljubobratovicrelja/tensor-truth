"""Converters between different source representations.

This module provides conversion utilities between:
- Web search sources (WebSearchSource from utils/web_search.py)
- RAG nodes (NodeWithScore from llama_index)
- Unified sources (SourceNode)
- API schemas (SourceNode from api/schemas/chat.py)
"""

import hashlib
from typing import TYPE_CHECKING, Any, Dict, List

from tensortruth.core.source import SourceNode, SourceStatus, SourceType

if TYPE_CHECKING:
    from tensortruth.utils.web_search import WebSearchSource


def _to_native(value: Any) -> Any:
    """Convert a value to a native Python type for JSON serialization.

    Handles numpy scalars (float32, int64, etc.) that are not
    JSON-serializable by default.
    """
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class SourceConverter:
    """Convert between different source representations."""

    @staticmethod
    def from_web_search_source(source: "WebSearchSource") -> SourceNode:
        """Convert WebSearchSource to SourceNode.

        Args:
            source: WebSearchSource from utils/web_search.py

        Returns:
            SourceNode with mapped fields
        """
        # Map status string to enum
        status_map = {
            "success": SourceStatus.SUCCESS,
            "failed": SourceStatus.FAILED,
            "skipped": SourceStatus.SKIPPED,
        }
        status = status_map.get(source.status, SourceStatus.FAILED)

        # Generate stable ID from URL
        source_id = SourceConverter._generate_id(source.url)

        return SourceNode(
            id=source_id,
            url=source.url,
            title=source.title,
            content=source.content,
            snippet=source.snippet,
            score=source.relevance_score,
            status=status,
            error=source.error,
            source_type=SourceType.WEB,
            content_chars=source.content_chars,
            metadata={
                "original_status": source.status,
            },
        )

    @staticmethod
    def from_rag_node(node: Any, node_index: int = 0) -> SourceNode:
        """Convert LlamaIndex NodeWithScore to SourceNode.

        Args:
            node: NodeWithScore or TextNode from llama_index
            node_index: Index for ID generation if no unique identifier

        Returns:
            SourceNode with mapped fields
        """
        # Handle NodeWithScore wrapper
        inner_node = getattr(node, "node", node)

        # Extract text content
        if hasattr(inner_node, "get_content"):
            text = inner_node.get_content()
        elif hasattr(node, "text"):
            text = node.text
        else:
            text = str(node)

        # Extract score
        score = node.score if hasattr(node, "score") else None

        # Extract metadata
        metadata = {}
        if hasattr(node, "metadata"):
            metadata = dict(node.metadata) if node.metadata else {}
        elif hasattr(inner_node, "metadata"):
            metadata = dict(inner_node.metadata) if inner_node.metadata else {}

        # Determine source type from metadata
        doc_type = metadata.get("doc_type", "")
        source_type = SourceConverter._map_doc_type_to_source_type(doc_type)

        # Generate ID
        node_id = getattr(inner_node, "id_", None) or getattr(node, "id_", None)
        if node_id:
            source_id = str(node_id)
        else:
            # Generate from content hash and index
            source_id = SourceConverter._generate_id(f"{text[:100]}_{node_index}")

        # Extract URL/source from metadata
        url = (
            metadata.get("source_url") or metadata.get("source") or metadata.get("url")
        )
        title = (
            metadata.get("display_name")
            or metadata.get("title")
            or metadata.get("file_name")
            or "Untitled"
        )

        return SourceNode(
            id=source_id,
            url=url,
            title=title,
            content=text,
            snippet=text[:500] if text else None,
            score=score,
            status=SourceStatus.SUCCESS,
            source_type=source_type,
            content_chars=len(text) if text else 0,
            metadata=metadata,
        )

    @staticmethod
    def to_api_schema(source: SourceNode) -> Dict[str, Any]:
        """Convert SourceNode to API response schema.

        Returns a dict matching the SourceNode Pydantic model structure
        in api/schemas/chat.py.

        Args:
            source: SourceNode to convert

        Returns:
            Dict with 'text', 'score', and 'metadata' keys
        """
        # Build metadata dict - merge source metadata first, then override with unified fields
        metadata = {
            **source.metadata,
            "source_url": source.url,
            "display_name": source.title,
            "doc_type": source.source_type.value,
            "fetch_status": source.status.value,
            "content_chars": source.content_chars,
        }

        if source.error:
            metadata["fetch_error"] = source.error

        # Sanitize metadata values for JSON serialization (numpy float32 etc.)
        metadata = {k: _to_native(v) for k, v in metadata.items()}

        # Text priority: content > snippet > empty
        text = source.content or source.snippet or ""

        return {
            "text": text,
            "score": _to_native(source.effective_score),
            "metadata": metadata,
        }

    @staticmethod
    def to_web_search_schema(source: SourceNode) -> Dict[str, Any]:
        """Convert SourceNode to WebSearchSource API schema.

        Returns a dict matching the WebSearchSource Pydantic model structure
        in api/schemas/chat.py (used for streaming web search updates).

        Args:
            source: SourceNode to convert

        Returns:
            Dict with url, title, status, error, snippet keys
        """
        return {
            "url": source.url or "",
            "title": source.title,
            "status": (
                source.status.value
                if source.status != SourceStatus.FILTERED
                else "skipped"
            ),
            "error": source.error,
            "snippet": source.snippet,
        }

    @staticmethod
    def batch_from_rag_nodes(nodes: List[Any]) -> List[SourceNode]:
        """Convert a list of RAG nodes to SourceNodes.

        Args:
            nodes: List of NodeWithScore or TextNode objects

        Returns:
            List of SourceNode objects
        """
        return [
            SourceConverter.from_rag_node(node, idx) for idx, node in enumerate(nodes)
        ]

    @staticmethod
    def batch_to_api_schema(sources: List[SourceNode]) -> List[Dict[str, Any]]:
        """Convert multiple SourceNodes to API schema dicts.

        Args:
            sources: List of SourceNode objects

        Returns:
            List of dicts matching SourceNode API schema
        """
        return [SourceConverter.to_api_schema(source) for source in sources]

    @staticmethod
    def _generate_id(content: str) -> str:
        """Generate a stable ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _map_doc_type_to_source_type(doc_type: str) -> SourceType:
        """Map document type string to SourceType enum."""
        doc_type_lower = doc_type.lower()

        if doc_type_lower in ("arxiv", "paper", "papers"):
            return SourceType.PAPER
        elif doc_type_lower in ("sphinx", "doxygen", "library", "library_doc"):
            return SourceType.LIBRARY_DOC
        elif doc_type_lower in ("pdf", "uploaded_pdf"):
            return SourceType.UPLOADED_PDF
        elif doc_type_lower in ("book", "pdf_book"):
            return SourceType.BOOK
        elif doc_type_lower == "web":
            return SourceType.WEB
        else:
            return SourceType.WEB  # Default to web
