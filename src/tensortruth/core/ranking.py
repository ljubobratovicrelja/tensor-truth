"""Unified reranking interface for both web search and RAG pipelines.

This module provides a single reranking interface that works with SourceNode,
replacing the separate reranking logic in web_search.py and rag_engine.py.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol

from tensortruth.core.source import SourceNode, SourceStatus

if TYPE_CHECKING:
    pass


class Reranker(Protocol):
    """Protocol for reranker implementations."""

    def rerank(
        self, query: str, documents: List[str], top_n: int = 10
    ) -> List[Dict[str, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rank
            top_n: Maximum number of results to return

        Returns:
            List of dicts with 'index' and 'relevance_score' keys
        """
        ...


@dataclass
class RankingResult:
    """Result of a ranking operation."""

    passed: List[SourceNode] = field(default_factory=list)
    filtered: List[SourceNode] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def all_sources(self) -> List[SourceNode]:
        """Get all sources (passed + filtered), sorted by score descending."""
        all_items = self.passed + self.filtered
        return sorted(
            all_items,
            key=lambda s: s.score if s.score is not None else 0.0,
            reverse=True,
        )


class RankingStage:
    """Unified reranking interface for both pipelines.

    Provides a consistent way to rank sources using any reranker implementation,
    with configurable thresholds and custom instruction support.
    """

    def __init__(
        self,
        reranker: Optional[Reranker] = None,
        threshold: float = 0.0,
        text_extractor: Optional[Callable[[SourceNode], str]] = None,
    ):
        """Initialize ranking stage.

        Args:
            reranker: Reranker implementation (if None, returns sources unranked)
            threshold: Minimum score threshold (sources below are filtered)
            text_extractor: Custom function to extract text from source for ranking
        """
        self.reranker = reranker
        self.threshold = threshold
        self.text_extractor = text_extractor or (lambda s: s.get_display_text())

    def rank(
        self,
        items: List[SourceNode],
        query: str,
        custom_instructions: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> RankingResult:
        """Rank items by relevance to query.

        Args:
            items: Sources to rank
            query: Search query for relevance scoring
            custom_instructions: Optional instructions to append to query
            top_n: Maximum number of top results to return (None = all)

        Returns:
            RankingResult with passed/filtered sources and score mapping
        """
        if not items:
            return RankingResult()

        # If no reranker, return sources with existing scores
        if self.reranker is None:
            return self._passthrough_result(items)

        # Build effective query with custom instructions
        effective_query = query
        if custom_instructions:
            effective_query = f"{query}\n\nAdditional context: {custom_instructions}"

        # Extract texts for ranking
        texts = [self.text_extractor(item) for item in items]

        # Skip ranking if all texts are empty
        if all(not t for t in texts):
            return self._passthrough_result(items)

        # Perform reranking
        limit = top_n if top_n is not None else len(items)
        results = self.reranker.rerank(
            query=effective_query, documents=texts, top_n=limit
        )

        # Map scores back to sources
        scores: Dict[str, float] = {}
        for result in results:
            idx = int(result.get("index", 0))
            score = float(result.get("relevance_score", 0.0))
            if 0 <= idx < len(items):
                source = items[idx]
                scores[source.id] = score
                source.score = score

        # Separate passed and filtered based on threshold
        passed: List[SourceNode] = []
        filtered: List[SourceNode] = []

        for source in items:
            if source.id in scores:
                if scores[source.id] >= self.threshold:
                    passed.append(source)
                else:
                    source.status = SourceStatus.FILTERED
                    filtered.append(source)
            else:
                # Source wasn't in top_n results, consider it filtered
                source.status = SourceStatus.FILTERED
                source.score = 0.0
                scores[source.id] = 0.0
                filtered.append(source)

        # Sort passed by score descending
        passed.sort(key=lambda s: s.score if s.score is not None else 0.0, reverse=True)
        filtered.sort(
            key=lambda s: s.score if s.score is not None else 0.0, reverse=True
        )

        return RankingResult(passed=passed, filtered=filtered, scores=scores)

    def _passthrough_result(self, items: List[SourceNode]) -> RankingResult:
        """Return items without reranking."""
        scores = {item.id: item.score or 0.0 for item in items}

        passed = []
        filtered = []
        for item in items:
            score = item.score if item.score is not None else 0.0
            if score >= self.threshold:
                passed.append(item)
            else:
                item.status = SourceStatus.FILTERED
                filtered.append(item)

        return RankingResult(passed=passed, filtered=filtered, scores=scores)
