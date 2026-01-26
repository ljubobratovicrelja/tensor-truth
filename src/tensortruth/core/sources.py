"""Unified source data structures for web research commands."""

from dataclasses import dataclass
from typing import ClassVar, Literal, Optional


@dataclass
class SourceNode:
    """A single source from web research (used by both /web and /browse).

    DEPRECATED: Use UnifiedSource from core/unified_sources.py instead.
    This class is maintained for backward compatibility and will be
    removed in a future version.

    This unified structure ensures consistent source metadata across all
    web research commands, enabling proper frontend display of:
    - Source titles and URLs
    - Fetch status and errors
    - Content previews
    - Relevance scores from reranking
    """

    # Deprecation marker for easy detection (grep for __deprecated__)
    __deprecated__: ClassVar[bool] = True
    __deprecation_reason__: ClassVar[str] = (
        "Use UnifiedSource from core/unified_sources.py instead."
    )

    url: str
    title: str
    status: Literal["success", "failed", "skipped"]
    error: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None  # Full fetched content
    content_chars: int = 0  # Character count of content passed to LLM
    relevance_score: Optional[float] = None  # Reranker score (0.0-1.0)
