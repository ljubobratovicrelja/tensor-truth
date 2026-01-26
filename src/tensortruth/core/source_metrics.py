"""Unified metrics computation for source quality analysis.

This module provides consistent metrics computation across both
web search and RAG pipelines.
"""

from dataclasses import dataclass
from statistics import mean, median
from typing import List, Optional

from tensortruth.core.unified_sources import SourceStatus, UnifiedSource

# Rough estimate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4


@dataclass
class SourceMetrics:
    """Quality metrics for a set of sources."""

    # Counts
    total_sources: int = 0
    successful_sources: int = 0
    filtered_sources: int = 0
    failed_sources: int = 0
    skipped_sources: int = 0

    # Score distribution
    score_mean: Optional[float] = None
    score_median: Optional[float] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None

    # Confidence tiers
    high_confidence_count: int = 0  # score >= 0.7
    medium_confidence_count: int = 0  # 0.4 <= score < 0.7
    low_confidence_count: int = 0  # score < 0.4

    # Coverage
    total_content_chars: int = 0
    estimated_tokens: int = 0

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for API response."""
        return {
            "total_sources": self.total_sources,
            "successful_sources": self.successful_sources,
            "filtered_sources": self.filtered_sources,
            "failed_sources": self.failed_sources,
            "skipped_sources": self.skipped_sources,
            "score_mean": self.score_mean,
            "score_median": self.score_median,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "high_confidence_count": self.high_confidence_count,
            "medium_confidence_count": self.medium_confidence_count,
            "low_confidence_count": self.low_confidence_count,
            "total_content_chars": self.total_content_chars,
            "estimated_tokens": self.estimated_tokens,
        }


def compute_metrics(sources: List[UnifiedSource]) -> SourceMetrics:
    """Compute quality metrics for a list of sources.

    Args:
        sources: List of UnifiedSource objects to analyze

    Returns:
        SourceMetrics with computed values
    """
    if not sources:
        return SourceMetrics()

    # Count by status
    successful = 0
    filtered = 0
    failed = 0
    skipped = 0

    for source in sources:
        if source.status == SourceStatus.SUCCESS:
            successful += 1
        elif source.status == SourceStatus.FILTERED:
            filtered += 1
        elif source.status == SourceStatus.FAILED:
            failed += 1
        elif source.status == SourceStatus.SKIPPED:
            skipped += 1

    # Collect scores (only from sources with scores)
    scores = [s.score for s in sources if s.score is not None]

    # Score statistics
    score_mean = mean(scores) if scores else None
    score_median = median(scores) if scores else None
    score_min = min(scores) if scores else None
    score_max = max(scores) if scores else None

    # Confidence tiers
    high = sum(1 for s in scores if s >= 0.7)
    medium = sum(1 for s in scores if 0.4 <= s < 0.7)
    low = sum(1 for s in scores if s < 0.4)

    # Content coverage
    total_chars = sum(s.content_chars for s in sources)
    estimated_tokens = total_chars // CHARS_PER_TOKEN

    return SourceMetrics(
        total_sources=len(sources),
        successful_sources=successful,
        filtered_sources=filtered,
        failed_sources=failed,
        skipped_sources=skipped,
        score_mean=score_mean,
        score_median=score_median,
        score_min=score_min,
        score_max=score_max,
        high_confidence_count=high,
        medium_confidence_count=medium,
        low_confidence_count=low,
        total_content_chars=total_chars,
        estimated_tokens=estimated_tokens,
    )


def compute_coverage_score(metrics: SourceMetrics) -> float:
    """Compute an overall coverage quality score (0.0-1.0).

    This combines success rate, score distribution, and content volume
    into a single quality indicator.

    Args:
        metrics: Pre-computed SourceMetrics

    Returns:
        Float between 0.0 and 1.0 indicating overall quality
    """
    if metrics.total_sources == 0:
        return 0.0

    # Success rate (weight: 0.4)
    success_rate = metrics.successful_sources / metrics.total_sources
    success_score = success_rate * 0.4

    # Score quality (weight: 0.4)
    if metrics.score_mean is not None:
        score_quality = metrics.score_mean * 0.4
    else:
        score_quality = 0.0

    # High confidence ratio (weight: 0.2)
    scored_count = (
        metrics.high_confidence_count
        + metrics.medium_confidence_count
        + metrics.low_confidence_count
    )
    if scored_count > 0:
        high_ratio = metrics.high_confidence_count / scored_count
        confidence_score = high_ratio * 0.2
    else:
        confidence_score = 0.0

    return success_score + score_quality + confidence_score
