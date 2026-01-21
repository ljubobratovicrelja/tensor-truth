"""Test that RAG service properly includes metrics in response chunks."""

from unittest.mock import MagicMock, patch

import pytest


class TestRAGServiceMetrics:
    """Test RAG service metrics integration."""

    def test_rag_service_yields_chunk_with_metrics(self):
        """Test that query() yields RAGChunk with metrics field populated.

        This verifies the RAG service actually includes metrics in the
        streaming response chunks.
        """
        from tensortruth.services.rag_service import RAGService

        # Create a mock engine with necessary attributes
        mock_engine = MagicMock()
        mock_llm = MagicMock()
        mock_llm.thinking = False

        # Mock the streaming response
        class MockChunk:
            def __init__(self, delta):
                self.delta = delta
                self.additional_kwargs = {}

        mock_llm.stream_chat.return_value = [MockChunk("test response")]
        mock_engine._llm = mock_llm
        mock_engine.memory = None

        # Mock retriever
        mock_retriever = MagicMock()
        mock_node = MagicMock()
        mock_node.score = 0.85
        mock_node.get_content.return_value = "Test content " * 100
        mock_node.node.get_content.return_value = "Test content " * 100
        mock_node.node.metadata = {
            "filename": "test.pdf",
            "doc_type": "paper",
        }
        mock_retriever.retrieve.return_value = [mock_node]
        mock_engine._retriever = mock_retriever
        mock_engine._node_postprocessors = []

        # Create service and set engine
        service = RAGService()
        service._engine = mock_engine

        # Execute query and collect chunks
        chunks = list(service.query("test query"))

        # Find the complete chunk (has is_complete=True)
        complete_chunk = None
        for chunk in chunks:
            if chunk.is_complete:
                complete_chunk = chunk
                break

        # Verify complete chunk exists and has metrics
        assert complete_chunk is not None, "No complete chunk found!"
        assert hasattr(
            complete_chunk, "metrics"
        ), "Complete chunk missing metrics attribute!"
        assert complete_chunk.metrics is not None, "Complete chunk metrics is None!"

        # Verify metrics structure
        metrics = complete_chunk.metrics
        assert isinstance(metrics, dict), "Metrics should be a dict!"
        assert "score_distribution" in metrics
        assert "diversity" in metrics
        assert "coverage" in metrics
        assert "quality" in metrics

        # Verify metrics have values
        assert metrics["score_distribution"]["mean"] is not None
        assert metrics["diversity"]["unique_sources"] > 0
        assert metrics["coverage"]["total_chunks"] > 0

    def test_compute_retrieval_metrics_called_with_reranked_nodes(self):
        """Test that metrics are computed AFTER postprocessor chain.

        This ensures metrics reflect final reranked scores, not raw embeddings.
        """
        from tensortruth.services.rag_service import RAGService

        mock_engine = MagicMock()
        mock_llm = MagicMock()
        mock_llm.thinking = False
        mock_llm.stream_chat.return_value = [
            MagicMock(delta="test", additional_kwargs={})
        ]
        mock_engine._llm = mock_llm
        mock_engine.memory = None

        mock_retriever = MagicMock()
        mock_node = MagicMock()
        mock_node.score = 0.75  # Original score
        mock_node.get_content.return_value = "Content " * 50
        mock_node.node.get_content.return_value = "Content " * 50
        mock_node.node.metadata = {"filename": "test.pdf", "doc_type": "paper"}
        mock_retriever.retrieve.return_value = [mock_node]
        mock_engine._retriever = mock_retriever

        # Mock postprocessor that changes scores (reranker)
        mock_postprocessor = MagicMock()

        def rerank_nodes(nodes, query_bundle):
            # Simulate reranker changing score
            for node in nodes:
                node.score = 0.95  # Reranked score
            return nodes

        mock_postprocessor.postprocess_nodes = rerank_nodes
        mock_engine._node_postprocessors = [mock_postprocessor]

        service = RAGService()
        service._engine = mock_engine

        # Execute query
        chunks = list(service.query("test"))
        complete_chunk = [c for c in chunks if c.is_complete][0]

        # Verify metrics use reranked score (0.95), not original (0.75)
        metrics = complete_chunk.metrics
        assert metrics["score_distribution"]["mean"] == pytest.approx(0.95, rel=0.01)
        assert metrics["score_distribution"]["max"] == pytest.approx(0.95, rel=0.01)

    def test_llm_only_mode_has_no_metrics(self):
        """Test that query_llm_only() returns None metrics."""
        from tensortruth.services.rag_service import RAGService

        service = RAGService()

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.thinking = False
        mock_llm.stream_chat.return_value = [
            MagicMock(delta="response", additional_kwargs={})
        ]

        with patch("tensortruth.services.rag_service.get_llm", return_value=mock_llm):
            chunks = list(service.query_llm_only("test", {}))
            complete_chunk = [c for c in chunks if c.is_complete][0]

            # LLM-only should have explicit None metrics
            assert hasattr(complete_chunk, "metrics")
            assert complete_chunk.metrics is None
