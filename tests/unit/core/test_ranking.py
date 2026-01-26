"""Unit tests for unified ranking interface."""

from typing import Dict, List
from unittest.mock import Mock

import pytest

from tensortruth.core.ranking import RankingResult, RankingStage
from tensortruth.core.source import SourceNode, SourceStatus, SourceType


def make_source(
    id: str,
    content: str = "test content",
    score: float = None,
    status: SourceStatus = SourceStatus.SUCCESS,
) -> SourceNode:
    """Helper to create test sources."""
    return SourceNode(
        id=id,
        title=f"Source {id}",
        source_type=SourceType.WEB,
        content=content,
        score=score,
        status=status,
    )


class MockReranker:
    """Mock reranker for testing."""

    def __init__(self, scores: Dict[int, float]):
        """Initialize with index -> score mapping."""
        self.scores = scores
        self.last_query = None
        self.last_documents = None

    def rerank(
        self, query: str, documents: List[str], top_n: int = 10
    ) -> List[Dict[str, float]]:
        """Return mocked reranking results."""
        self.last_query = query
        self.last_documents = documents

        results = []
        for idx, score in self.scores.items():
            if idx < len(documents) and len(results) < top_n:
                results.append({"index": idx, "relevance_score": score})

        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_n]


@pytest.mark.unit
class TestRankingResult:
    """Tests for RankingResult dataclass."""

    def test_empty_result(self):
        """Test empty RankingResult."""
        result = RankingResult()

        assert result.passed == []
        assert result.filtered == []
        assert result.scores == {}
        assert result.all_sources == []

    def test_all_sources_sorted(self):
        """Test that all_sources returns sorted by score descending."""
        s1 = make_source("1", score=0.3)
        s2 = make_source("2", score=0.9)
        s3 = make_source("3", score=0.6)

        result = RankingResult(
            passed=[s2],
            filtered=[s1, s3],
            scores={"1": 0.3, "2": 0.9, "3": 0.6},
        )

        all_sources = result.all_sources
        assert len(all_sources) == 3
        assert all_sources[0].id == "2"
        assert all_sources[1].id == "3"
        assert all_sources[2].id == "1"

    def test_all_sources_handles_none_scores(self):
        """Test that all_sources handles None scores."""
        s1 = make_source("1", score=None)
        s2 = make_source("2", score=0.5)

        result = RankingResult(passed=[s2], filtered=[s1], scores={"2": 0.5})

        all_sources = result.all_sources
        assert len(all_sources) == 2
        assert all_sources[0].id == "2"
        assert all_sources[1].id == "1"


@pytest.mark.unit
class TestRankingStage:
    """Tests for RankingStage."""

    def test_rank_empty_input(self):
        """Test ranking empty list returns empty result."""
        stage = RankingStage(reranker=Mock())

        result = stage.rank([], query="test query")

        assert result.passed == []
        assert result.filtered == []
        assert result.scores == {}

    def test_rank_without_reranker(self):
        """Test ranking without reranker passes through sources."""
        stage = RankingStage(reranker=None)
        sources = [
            make_source("1", score=0.8),
            make_source("2", score=0.3),
        ]

        result = stage.rank(sources, query="test")

        assert len(result.passed) == 2
        assert result.scores == {"1": 0.8, "2": 0.3}

    def test_rank_with_threshold_filtering(self):
        """Test that threshold filtering works correctly."""
        reranker = MockReranker({0: 0.8, 1: 0.3, 2: 0.6})
        stage = RankingStage(reranker=reranker, threshold=0.5)

        sources = [
            make_source("1", content="first"),
            make_source("2", content="second"),
            make_source("3", content="third"),
        ]

        result = stage.rank(sources, query="test")

        assert len(result.passed) == 2
        assert len(result.filtered) == 1
        assert result.filtered[0].id == "2"
        assert result.filtered[0].status == SourceStatus.FILTERED

    def test_rank_sorted_by_score(self):
        """Test that passed results are sorted by score descending."""
        reranker = MockReranker({0: 0.5, 1: 0.9, 2: 0.7})
        stage = RankingStage(reranker=reranker)

        sources = [
            make_source("1", content="first"),
            make_source("2", content="second"),
            make_source("3", content="third"),
        ]

        result = stage.rank(sources, query="test")

        assert result.passed[0].id == "2"  # score 0.9
        assert result.passed[1].id == "3"  # score 0.7
        assert result.passed[2].id == "1"  # score 0.5

    def test_rank_with_custom_instructions(self):
        """Test that custom instructions are appended to query."""
        reranker = MockReranker({0: 0.8})
        stage = RankingStage(reranker=reranker)

        sources = [make_source("1", content="content")]

        stage.rank(sources, query="main query", custom_instructions="focus on examples")

        assert "main query" in reranker.last_query
        assert "focus on examples" in reranker.last_query
        assert "Additional context" in reranker.last_query

    def test_rank_uses_text_extractor(self):
        """Test that custom text extractor is used."""

        def custom_extractor(s: SourceNode) -> str:
            return f"EXTRACTED: {s.title}"

        reranker = MockReranker({0: 0.8})
        stage = RankingStage(reranker=reranker, text_extractor=custom_extractor)

        sources = [make_source("1", content="ignored content")]

        stage.rank(sources, query="test")

        assert reranker.last_documents == ["EXTRACTED: Source 1"]

    def test_rank_assigns_scores_to_sources(self):
        """Test that ranking assigns scores back to source objects."""
        reranker = MockReranker({0: 0.85, 1: 0.45})
        stage = RankingStage(reranker=reranker)

        sources = [
            make_source("1", content="first"),
            make_source("2", content="second"),
        ]

        stage.rank(sources, query="test")

        assert sources[0].score == 0.85
        assert sources[1].score == 0.45

    def test_rank_all_below_threshold(self):
        """Test ranking when all sources are below threshold."""
        reranker = MockReranker({0: 0.2, 1: 0.3})
        stage = RankingStage(reranker=reranker, threshold=0.5)

        sources = [
            make_source("1", content="first"),
            make_source("2", content="second"),
        ]

        result = stage.rank(sources, query="test")

        assert len(result.passed) == 0
        assert len(result.filtered) == 2
        assert all(s.status == SourceStatus.FILTERED for s in result.filtered)

    def test_rank_with_top_n(self):
        """Test that top_n limits results from reranker."""
        reranker = MockReranker({0: 0.8, 1: 0.7, 2: 0.6})
        stage = RankingStage(reranker=reranker)

        sources = [
            make_source("1", content="first"),
            make_source("2", content="second"),
            make_source("3", content="third"),
        ]

        result = stage.rank(sources, query="test", top_n=2)

        # Source 3 (index 2, score 0.6) should be filtered because it's outside top_n
        passed_ids = [s.id for s in result.passed]
        filtered_ids = [s.id for s in result.filtered]

        assert len(passed_ids) == 2
        assert "3" in filtered_ids

    def test_rank_handles_empty_texts(self):
        """Test ranking handles sources with empty content."""
        stage = RankingStage(reranker=Mock())

        sources = [
            make_source("1", content=""),
            make_source("2", content=""),
        ]

        # Should not call reranker when all texts are empty
        result = stage.rank(sources, query="test")

        assert len(result.passed) + len(result.filtered) == 2

    def test_passthrough_with_threshold(self):
        """Test passthrough mode applies threshold."""
        stage = RankingStage(reranker=None, threshold=0.5)

        sources = [
            make_source("1", score=0.8),
            make_source("2", score=0.3),
            make_source("3", score=0.6),
        ]

        result = stage.rank(sources, query="test")

        assert len(result.passed) == 2
        assert len(result.filtered) == 1
        assert result.filtered[0].id == "2"

    def test_score_mapping_in_result(self):
        """Test that scores dict maps source IDs to scores."""
        reranker = MockReranker({0: 0.75, 1: 0.55})
        stage = RankingStage(reranker=reranker)

        sources = [
            make_source("source-a", content="first"),
            make_source("source-b", content="second"),
        ]

        result = stage.rank(sources, query="test")

        assert result.scores["source-a"] == 0.75
        assert result.scores["source-b"] == 0.55
