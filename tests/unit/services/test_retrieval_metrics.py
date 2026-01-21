"""Tests for retrieval metrics calculation."""

from unittest.mock import MagicMock

import pytest

from tensortruth.services.retrieval_metrics import (
    RetrievalMetrics,
    calculate_entropy,
    compute_retrieval_metrics,
)


class TestCalculateEntropy:
    """Tests for Shannon entropy calculation."""

    def test_uniform_distribution(self):
        """Equal counts should give max entropy."""
        counts = [10, 10, 10, 10]
        entropy = calculate_entropy(counts)
        assert entropy == pytest.approx(2.0, rel=0.01)  # log2(4) = 2

    def test_single_source(self):
        """Single source should give zero entropy."""
        counts = [40]
        entropy = calculate_entropy(counts)
        assert entropy == 0.0

    def test_empty_counts(self):
        """Empty list should return 0."""
        assert calculate_entropy([]) == 0.0

    def test_skewed_distribution(self):
        """Skewed distribution should give lower entropy."""
        counts = [30, 5, 3, 2]
        entropy = calculate_entropy(counts)
        assert 0 < entropy < 2.0  # Between min and max possible

    def test_zero_counts(self):
        """Zero counts should be handled gracefully."""
        counts = [10, 0, 5]
        entropy = calculate_entropy(counts)
        assert entropy > 0


class TestComputeRetrievalMetrics:
    """Tests for comprehensive metrics computation."""

    @pytest.fixture
    def mock_nodes_varied_scores(self):
        """Create mock nodes with varied scores."""
        nodes = []
        scores = [0.95, 0.87, 0.76, 0.65, 0.52]
        for i, score in enumerate(scores):
            node = MagicMock()
            node.score = score
            node.node.get_content.return_value = "Test content " * 100  # ~1300 chars
            node.node.metadata = {
                "filename": f"doc{i}.pdf",
                "doc_type": "paper" if i % 2 == 0 else "book",
            }
            nodes.append(node)
        return nodes

    @pytest.fixture
    def mock_nodes_single_source(self):
        """Create mock nodes all from same source."""
        nodes = []
        for i in range(3):
            node = MagicMock()
            node.score = 0.8
            node.node.get_content.return_value = "Same source content"
            node.node.metadata = {
                "filename": "single_doc.pdf",
                "doc_type": "paper",
            }
            nodes.append(node)
        return nodes

    def test_score_statistics(self, mock_nodes_varied_scores):
        """Test score distribution metrics."""
        metrics = compute_retrieval_metrics(mock_nodes_varied_scores)

        # Check mean
        expected_mean = sum([0.95, 0.87, 0.76, 0.65, 0.52]) / 5
        assert metrics.score_mean == pytest.approx(expected_mean, rel=0.01)

        # Check min/max
        assert metrics.score_min == 0.52
        assert metrics.score_max == 0.95

        # Check median
        assert metrics.score_median == 0.76

        # Check standard deviation exists
        assert metrics.score_std is not None
        assert metrics.score_std > 0

        # Check range
        assert metrics.score_range == pytest.approx(0.43, rel=0.01)

    def test_quartiles(self, mock_nodes_varied_scores):
        """Test quartile calculations."""
        metrics = compute_retrieval_metrics(mock_nodes_varied_scores)

        assert metrics.score_q1 is not None
        assert metrics.score_q3 is not None
        assert metrics.score_iqr is not None

        # Q1 should be less than median, Q3 should be greater
        assert metrics.score_q1 < metrics.score_median
        assert metrics.score_q3 > metrics.score_median

        # IQR should be positive
        assert metrics.score_iqr > 0

    def test_diversity_metrics(self, mock_nodes_varied_scores):
        """Test diversity calculations."""
        metrics = compute_retrieval_metrics(mock_nodes_varied_scores)

        # 5 different files
        assert metrics.unique_sources == 5

        # 2 doc types (paper and book)
        assert metrics.source_types == 2

        # High entropy since all sources are different
        assert metrics.source_entropy is not None
        assert metrics.source_entropy > 2.0  # log2(5) ≈ 2.32

    def test_diversity_single_source(self, mock_nodes_single_source):
        """Test diversity when all nodes from same source."""
        metrics = compute_retrieval_metrics(mock_nodes_single_source)

        assert metrics.unique_sources == 1
        assert metrics.source_types == 1
        assert metrics.source_entropy == 0.0  # All from same source

    def test_coverage_metrics(self, mock_nodes_varied_scores):
        """Test coverage calculations."""
        metrics = compute_retrieval_metrics(mock_nodes_varied_scores)

        assert metrics.total_chunks == 5
        assert metrics.total_context_chars > 0
        assert metrics.avg_chunk_length > 0

        # Estimated tokens should be roughly chars / 4
        expected_tokens = metrics.total_context_chars // 4
        assert metrics.estimated_tokens == expected_tokens

    def test_quality_indicators(self, mock_nodes_varied_scores):
        """Test quality ratios."""
        metrics = compute_retrieval_metrics(mock_nodes_varied_scores)

        # Scores: [0.95, 0.87, 0.76, 0.65, 0.52]
        # High (>=0.7): 3 out of 5 = 0.6
        # Low (<0.4): 0 out of 5 = 0.0
        assert metrics.high_confidence_ratio == pytest.approx(0.6, rel=0.01)
        assert metrics.low_confidence_ratio == 0.0

    def test_quality_with_low_scores(self):
        """Test quality indicators with low scores."""
        nodes = []
        scores = [0.35, 0.28, 0.42, 0.72]  # 2 low, 1 medium, 1 high
        for i, score in enumerate(scores):
            node = MagicMock()
            node.score = score
            node.node.get_content.return_value = "content"
            node.node.metadata = {"filename": f"doc{i}.pdf", "doc_type": "paper"}
            nodes.append(node)

        metrics = compute_retrieval_metrics(nodes)

        # High (>=0.7): 1/4 = 0.25
        # Low (<0.4): 2/4 = 0.5
        assert metrics.high_confidence_ratio == pytest.approx(0.25, rel=0.01)
        assert metrics.low_confidence_ratio == pytest.approx(0.5, rel=0.01)

    def test_empty_nodes(self):
        """Test with empty node list."""
        metrics = compute_retrieval_metrics([])

        # Score stats should be None
        assert metrics.score_mean is None
        assert metrics.score_median is None
        assert metrics.score_min is None
        assert metrics.score_max is None

        # Counts should be zero
        assert metrics.unique_sources == 0
        assert metrics.total_chunks == 0
        assert metrics.total_context_chars == 0

    def test_nodes_without_scores(self):
        """Test nodes with missing scores."""
        nodes = []
        for i in range(3):
            node = MagicMock()
            node.score = None
            node.node.get_content.return_value = "content"
            node.node.metadata = {"filename": f"doc{i}.pdf"}
            nodes.append(node)

        metrics = compute_retrieval_metrics(nodes)

        # No valid scores
        assert metrics.score_mean is None
        assert metrics.score_std is None

        # But other metrics should work
        assert metrics.total_chunks == 3
        assert metrics.unique_sources == 3

    def test_single_node_no_std(self):
        """Test that single node has no standard deviation."""
        node = MagicMock()
        node.score = 0.8
        node.node.get_content.return_value = "content"
        node.node.metadata = {"filename": "doc.pdf", "doc_type": "paper"}

        metrics = compute_retrieval_metrics([node])

        assert metrics.score_mean == 0.8
        assert metrics.score_std is None  # Need 2+ samples

    def test_metadata_extraction_variants(self):
        """Test different metadata key variants."""
        nodes = []

        # Node with 'filename'
        node1 = MagicMock()
        node1.score = 0.9
        node1.node.get_content.return_value = "content"
        node1.node.metadata = {"filename": "doc1.pdf"}
        nodes.append(node1)

        # Node with 'file_name'
        node2 = MagicMock()
        node2.score = 0.8
        node2.node.get_content.return_value = "content"
        node2.node.metadata = {"file_name": "doc2.pdf"}
        nodes.append(node2)

        # Node with 'source_url'
        node3 = MagicMock()
        node3.score = 0.7
        node3.node.get_content.return_value = "content"
        node3.node.metadata = {"source_url": "https://example.com/doc3"}
        nodes.append(node3)

        metrics = compute_retrieval_metrics(nodes)

        # Should recognize all 3 as unique sources
        assert metrics.unique_sources == 3

    def test_serialization(self, mock_nodes_varied_scores):
        """Test to_dict() serialization."""
        metrics = compute_retrieval_metrics(mock_nodes_varied_scores)
        data = metrics.to_dict()

        # Check structure
        assert "score_distribution" in data
        assert "diversity" in data
        assert "coverage" in data
        assert "quality" in data

        # Check score distribution fields
        assert "mean" in data["score_distribution"]
        assert "median" in data["score_distribution"]
        assert "std" in data["score_distribution"]
        assert "iqr" in data["score_distribution"]

        # Check values are serialized correctly
        assert data["score_distribution"]["mean"] is not None
        assert isinstance(data["diversity"]["unique_sources"], int)
        assert isinstance(data["coverage"]["total_chunks"], int)

    def test_to_dict_with_none_values(self):
        """Test serialization with None values."""
        metrics = RetrievalMetrics()
        data = metrics.to_dict()

        # Should handle None values gracefully
        assert data["score_distribution"]["mean"] is None
        assert data["diversity"]["source_entropy"] is None

    def test_long_content(self):
        """Test with very long content."""
        node = MagicMock()
        node.score = 0.85
        node.node.get_content.return_value = "x" * 10000  # 10k chars
        node.node.metadata = {"filename": "long.pdf", "doc_type": "paper"}

        metrics = compute_retrieval_metrics([node])

        assert metrics.total_context_chars == 10000
        assert metrics.avg_chunk_length == 10000.0
        assert metrics.estimated_tokens == 2500  # 10000 / 4

    def test_numpy_float32_serialization(self):
        """Test that numpy float32 scores are properly serialized to JSON."""
        import json

        import numpy as np

        # Create nodes with numpy float32 scores (as returned by retrieval systems)
        nodes = []
        for i, score in enumerate([0.95, 0.87, 0.76]):
            node = MagicMock()
            node.score = np.float32(score)  # Simulate numpy score
            node.node.get_content.return_value = "Test content"
            node.node.metadata = {
                "filename": f"doc{i}.pdf",
                "doc_type": "paper",
            }
            nodes.append(node)

        metrics = compute_retrieval_metrics(nodes)
        metrics_dict = metrics.to_dict()

        # Should serialize without error
        json_str = json.dumps(metrics_dict)
        assert json_str is not None

        # Verify all numeric values are native Python types
        assert isinstance(
            metrics_dict["score_distribution"]["mean"], (float, type(None))
        )
        assert isinstance(metrics_dict["diversity"]["unique_sources"], int)
        assert isinstance(metrics_dict["coverage"]["total_chunks"], int)
