"""Unit tests for source metrics computation."""

import pytest

from tensortruth.core.source_metrics import (
    CHARS_PER_TOKEN,
    SourceMetrics,
    compute_coverage_score,
    compute_metrics,
)
from tensortruth.core.unified_sources import SourceStatus, SourceType, UnifiedSource


def make_source(
    id: str,
    status: SourceStatus = SourceStatus.SUCCESS,
    score: float = None,
    content_chars: int = 0,
) -> UnifiedSource:
    """Helper to create test sources."""
    return UnifiedSource(
        id=id,
        title=f"Source {id}",
        source_type=SourceType.WEB,
        status=status,
        score=score,
        content_chars=content_chars,
    )


@pytest.mark.unit
class TestSourceMetrics:
    """Tests for SourceMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = SourceMetrics()

        assert metrics.total_sources == 0
        assert metrics.successful_sources == 0
        assert metrics.filtered_sources == 0
        assert metrics.failed_sources == 0
        assert metrics.skipped_sources == 0
        assert metrics.score_mean is None
        assert metrics.score_median is None
        assert metrics.score_min is None
        assert metrics.score_max is None
        assert metrics.high_confidence_count == 0
        assert metrics.medium_confidence_count == 0
        assert metrics.low_confidence_count == 0
        assert metrics.total_content_chars == 0
        assert metrics.estimated_tokens == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = SourceMetrics(
            total_sources=10,
            successful_sources=8,
            score_mean=0.75,
            high_confidence_count=5,
            total_content_chars=5000,
            estimated_tokens=1250,
        )

        result = metrics.to_dict()

        assert result["total_sources"] == 10
        assert result["successful_sources"] == 8
        assert result["score_mean"] == 0.75
        assert result["high_confidence_count"] == 5
        assert result["total_content_chars"] == 5000
        assert result["estimated_tokens"] == 1250


@pytest.mark.unit
class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_empty_sources(self):
        """Test metrics for empty source list."""
        metrics = compute_metrics([])

        assert metrics.total_sources == 0
        assert metrics.successful_sources == 0
        assert metrics.score_mean is None

    def test_single_source(self):
        """Test metrics for single source."""
        sources = [
            make_source("1", status=SourceStatus.SUCCESS, score=0.8, content_chars=100)
        ]

        metrics = compute_metrics(sources)

        assert metrics.total_sources == 1
        assert metrics.successful_sources == 1
        assert metrics.score_mean == 0.8
        assert metrics.score_median == 0.8
        assert metrics.score_min == 0.8
        assert metrics.score_max == 0.8
        assert metrics.high_confidence_count == 1
        assert metrics.total_content_chars == 100
        assert metrics.estimated_tokens == 100 // CHARS_PER_TOKEN

    def test_status_counting(self):
        """Test counting by status."""
        sources = [
            make_source("1", status=SourceStatus.SUCCESS),
            make_source("2", status=SourceStatus.SUCCESS),
            make_source("3", status=SourceStatus.FAILED),
            make_source("4", status=SourceStatus.SKIPPED),
            make_source("5", status=SourceStatus.FILTERED),
            make_source("6", status=SourceStatus.FILTERED),
        ]

        metrics = compute_metrics(sources)

        assert metrics.total_sources == 6
        assert metrics.successful_sources == 2
        assert metrics.failed_sources == 1
        assert metrics.skipped_sources == 1
        assert metrics.filtered_sources == 2

    def test_score_statistics(self):
        """Test score statistical calculations."""
        sources = [
            make_source("1", score=0.9),
            make_source("2", score=0.7),
            make_source("3", score=0.5),
            make_source("4", score=0.3),
            make_source("5", score=0.1),
        ]

        metrics = compute_metrics(sources)

        assert metrics.score_mean == 0.5
        assert metrics.score_median == 0.5
        assert metrics.score_min == 0.1
        assert metrics.score_max == 0.9

    def test_confidence_tiers(self):
        """Test confidence tier counting."""
        sources = [
            make_source("1", score=0.9),  # high
            make_source("2", score=0.75),  # high
            make_source("3", score=0.7),  # high
            make_source("4", score=0.5),  # medium
            make_source("5", score=0.4),  # medium
            make_source("6", score=0.3),  # low
            make_source("7", score=0.1),  # low
        ]

        metrics = compute_metrics(sources)

        assert metrics.high_confidence_count == 3
        assert metrics.medium_confidence_count == 2
        assert metrics.low_confidence_count == 2

    def test_content_chars_aggregation(self):
        """Test total content chars and token estimation."""
        sources = [
            make_source("1", content_chars=1000),
            make_source("2", content_chars=2000),
            make_source("3", content_chars=500),
        ]

        metrics = compute_metrics(sources)

        assert metrics.total_content_chars == 3500
        assert metrics.estimated_tokens == 3500 // CHARS_PER_TOKEN

    def test_sources_without_scores(self):
        """Test metrics when some sources have no scores."""
        sources = [
            make_source("1", score=0.8),
            make_source("2", score=None),
            make_source("3", score=0.6),
        ]

        metrics = compute_metrics(sources)

        assert metrics.total_sources == 3
        # Only 2 sources have scores
        assert metrics.score_mean == 0.7  # (0.8 + 0.6) / 2
        assert metrics.score_min == 0.6
        assert metrics.score_max == 0.8

    def test_all_sources_without_scores(self):
        """Test metrics when no sources have scores."""
        sources = [
            make_source("1", score=None),
            make_source("2", score=None),
        ]

        metrics = compute_metrics(sources)

        assert metrics.total_sources == 2
        assert metrics.score_mean is None
        assert metrics.score_median is None
        assert metrics.high_confidence_count == 0

    def test_mixed_scores_distribution(self):
        """Test realistic mixed score distribution."""
        sources = [
            make_source("1", score=0.95, content_chars=500),  # high
            make_source("2", score=0.82, content_chars=800),  # high
            make_source("3", score=0.65, content_chars=300),  # medium
            make_source("4", score=0.55, content_chars=600),  # medium
            make_source("5", score=0.45, content_chars=200),  # medium
            make_source("6", score=0.25, content_chars=100),  # low
            make_source("7", score=None, content_chars=400),  # no score
        ]

        metrics = compute_metrics(sources)

        assert metrics.total_sources == 7
        assert metrics.high_confidence_count == 2
        assert metrics.medium_confidence_count == 3
        assert metrics.low_confidence_count == 1
        assert metrics.total_content_chars == 2900


@pytest.mark.unit
class TestComputeCoverageScore:
    """Tests for compute_coverage_score function."""

    def test_empty_metrics(self):
        """Test coverage score for empty metrics."""
        metrics = SourceMetrics()

        score = compute_coverage_score(metrics)

        assert score == 0.0

    def test_perfect_metrics(self):
        """Test coverage score for ideal metrics."""
        # All sources successful, high scores
        sources = [
            make_source("1", status=SourceStatus.SUCCESS, score=0.9),
            make_source("2", status=SourceStatus.SUCCESS, score=0.85),
            make_source("3", status=SourceStatus.SUCCESS, score=0.8),
        ]
        metrics = compute_metrics(sources)

        score = compute_coverage_score(metrics)

        # Should be close to 1.0
        assert score > 0.8

    def test_poor_metrics(self):
        """Test coverage score for poor metrics."""
        # Many failures, low scores
        sources = [
            make_source("1", status=SourceStatus.SUCCESS, score=0.2),
            make_source("2", status=SourceStatus.FAILED, score=None),
            make_source("3", status=SourceStatus.FAILED, score=None),
            make_source("4", status=SourceStatus.SKIPPED, score=None),
        ]
        metrics = compute_metrics(sources)

        score = compute_coverage_score(metrics)

        # Should be low
        assert score < 0.3

    def test_mixed_metrics(self):
        """Test coverage score for mixed results."""
        sources = [
            make_source("1", status=SourceStatus.SUCCESS, score=0.75),
            make_source("2", status=SourceStatus.SUCCESS, score=0.55),
            make_source("3", status=SourceStatus.FILTERED, score=0.35),
            make_source("4", status=SourceStatus.FAILED, score=None),
        ]
        metrics = compute_metrics(sources)

        score = compute_coverage_score(metrics)

        # Should be moderate
        assert 0.3 < score < 0.7

    def test_no_scored_sources(self):
        """Test coverage score when no sources have scores."""
        sources = [
            make_source("1", status=SourceStatus.SUCCESS, score=None),
            make_source("2", status=SourceStatus.SUCCESS, score=None),
        ]
        metrics = compute_metrics(sources)

        score = compute_coverage_score(metrics)

        # Only success rate contributes
        assert score == pytest.approx(0.4)  # 1.0 * 0.4 (success weight)
