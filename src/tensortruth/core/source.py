"""Source model for both web search and RAG pipelines.

This module provides a single source of truth for source data representation,
eliminating duplication between web_search.py and rag_engine.py.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SourceStatus(str, Enum):
    """Status of a source during pipeline processing."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    FILTERED = "filtered"  # Below relevance threshold


class SourceType(str, Enum):
    """Type of source document."""

    WEB = "web"
    PAPER = "paper"
    LIBRARY_DOC = "library_doc"
    UPLOADED_PDF = "uploaded_pdf"
    BOOK = "book"


@dataclass
class SourceNode:
    """Source model for both web search and RAG.

    This structure ensures consistent source metadata across all
    pipelines, enabling:
    - Proper frontend display of sources
    - Unified reranking interface
    - Consistent metrics computation
    - Single conversion path to API schemas
    """

    id: str
    title: str
    source_type: SourceType
    status: SourceStatus = SourceStatus.SUCCESS

    # Location
    url: Optional[str] = None

    # Content
    content: Optional[str] = None
    snippet: Optional[str] = None

    # Scoring
    score: Optional[float] = None

    # Error handling
    error: Optional[str] = None

    # Metrics
    content_chars: int = 0

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "snippet": self.snippet,
            "score": self.score,
            "status": self.status.value,
            "error": self.error,
            "source_type": self.source_type.value,
            "metadata": self.metadata,
            "content_chars": self.content_chars,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceNode":
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            url=data.get("url"),
            title=data["title"],
            content=data.get("content"),
            snippet=data.get("snippet"),
            score=data.get("score"),
            status=SourceStatus(data.get("status", "success")),
            error=data.get("error"),
            source_type=SourceType(data.get("source_type", "web")),
            metadata=data.get("metadata", {}),
            content_chars=data.get("content_chars", 0),
        )

    def get_display_text(self) -> str:
        """Get text for display/ranking purposes."""
        if self.content:
            return self.content
        if self.snippet:
            return self.snippet
        return ""

    def is_successful(self) -> bool:
        """Check if source was successfully processed."""
        return self.status == SourceStatus.SUCCESS

    def is_usable(self) -> bool:
        """Check if source can be used for synthesis."""
        return self.status in (SourceStatus.SUCCESS, SourceStatus.FILTERED)

    @property
    def effective_score(self) -> float:
        """Get score with status-based fallback for display purposes.

        Returns the explicit score if set, otherwise defaults based on status:
        - SUCCESS: 1.0
        - All other statuses: 0.0
        """
        if self.score is not None:
            return self.score
        return 1.0 if self.status == SourceStatus.SUCCESS else 0.0
