"""Unit tests for RAG service functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


@pytest.mark.unit
def test_postprocessor_application():
    """Test that postprocessors are applied during retrieval."""
    from tensortruth.services.rag_service import RAGService

    # Create mock engine with retriever
    engine = Mock()

    # Create mock nodes (12 raw nodes from retriever)
    raw_nodes = [
        NodeWithScore(node=TextNode(text=f"Node {i}"), score=0.9 - i * 0.05)
        for i in range(12)
    ]
    engine._retriever.retrieve.return_value = raw_nodes

    # Mock reranker that reduces to 5 nodes
    reranked_nodes = [
        NodeWithScore(node=TextNode(text=f"Reranked Node {i}"), score=0.95 - i * 0.05)
        for i in range(5)
    ]

    reranker = Mock()
    reranker.postprocess_nodes.return_value = reranked_nodes
    engine._node_postprocessors = [reranker]

    # Mock LLM for streaming
    engine._llm = Mock()
    engine._llm.thinking = False

    # Mock streaming response
    mock_chunk = Mock()
    mock_chunk.delta = "test"
    mock_chunk.additional_kwargs = {}
    engine._llm.stream_chat.return_value = [mock_chunk]

    # Mock memory
    engine.memory = Mock()
    engine.memory.get.return_value = []
    engine.memory.put = Mock()

    # Create service and set engine
    service = RAGService(Mock())
    service._engine = engine

    # Execute query
    results = list(service.query("test query"))

    # Verify postprocessor was called
    assert reranker.postprocess_nodes.called

    # Find the final chunk with source_nodes
    final_chunks = [c for c in results if hasattr(c, "is_complete") and c.is_complete]
    assert len(final_chunks) == 1

    final_chunk = final_chunks[0]

    # Verify final source count is 5 (reranked), not 12 (raw)
    assert len(final_chunk.source_nodes) == 5

    # Verify source nodes are the reranked ones
    assert final_chunk.source_nodes == reranked_nodes


@pytest.mark.unit
def test_postprocessor_failure_handling():
    """Test that postprocessor failures don't break streaming."""
    from tensortruth.services.rag_service import RAGService

    # Create mock engine with retriever
    engine = Mock()

    # Create mock nodes
    raw_nodes = [
        NodeWithScore(node=TextNode(text=f"Node {i}"), score=0.9 - i * 0.05)
        for i in range(5)
    ]
    engine._retriever.retrieve.return_value = raw_nodes

    # Mock reranker that fails
    reranker = Mock()
    reranker.postprocess_nodes.side_effect = Exception("Reranker failed")
    engine._node_postprocessors = [reranker]

    # Mock LLM for streaming
    engine._llm = Mock()
    engine._llm.thinking = False

    # Mock streaming response
    mock_chunk = Mock()
    mock_chunk.delta = "test"
    mock_chunk.additional_kwargs = {}
    engine._llm.stream_chat.return_value = [mock_chunk]

    # Mock memory
    engine.memory = Mock()
    engine.memory.get.return_value = []
    engine.memory.put = Mock()

    # Create service and set engine
    service = RAGService(Mock())
    service._engine = engine

    # Execute query - should not raise exception
    results = list(service.query("test query"))

    # Find the final chunk with source_nodes
    final_chunks = [c for c in results if hasattr(c, "is_complete") and c.is_complete]
    assert len(final_chunks) == 1

    final_chunk = final_chunks[0]

    # Should fall back to raw nodes (5)
    assert len(final_chunk.source_nodes) == 5
    assert final_chunk.source_nodes == raw_nodes


@pytest.mark.unit
def test_postprocessor_multiple_stages():
    """Test that multiple postprocessors are applied in sequence."""
    from tensortruth.services.rag_service import RAGService

    # Create mock engine with retriever
    engine = Mock()

    # Create mock nodes (10 raw nodes)
    raw_nodes = [
        NodeWithScore(node=TextNode(text=f"Node {i}"), score=0.9 - i * 0.05)
        for i in range(10)
    ]
    engine._retriever.retrieve.return_value = raw_nodes

    # First postprocessor: reranker (reduces to 7)
    reranked_nodes = [
        NodeWithScore(node=TextNode(text=f"Reranked {i}"), score=0.85 - i * 0.05)
        for i in range(7)
    ]
    reranker = Mock()
    reranker.postprocess_nodes.return_value = reranked_nodes

    # Second postprocessor: cutoff filter (reduces to 5)
    filtered_nodes = reranked_nodes[:5]
    cutoff_filter = Mock()
    cutoff_filter.postprocess_nodes.return_value = filtered_nodes

    engine._node_postprocessors = [reranker, cutoff_filter]

    # Mock LLM for streaming
    engine._llm = Mock()
    engine._llm.thinking = False

    # Mock streaming response
    mock_chunk = Mock()
    mock_chunk.delta = "test"
    mock_chunk.additional_kwargs = {}
    engine._llm.stream_chat.return_value = [mock_chunk]

    # Mock memory
    engine.memory = Mock()
    engine.memory.get.return_value = []
    engine.memory.put = Mock()

    # Create service and set engine
    service = RAGService(Mock())
    service._engine = engine

    # Execute query
    results = list(service.query("test query"))

    # Verify both postprocessors were called in order
    assert reranker.postprocess_nodes.called
    assert cutoff_filter.postprocess_nodes.called

    # Verify reranker was called first (with raw nodes)
    reranker_call_args = reranker.postprocess_nodes.call_args
    assert reranker_call_args[0][0] == raw_nodes

    # Find the final chunk
    final_chunks = [c for c in results if hasattr(c, "is_complete") and c.is_complete]
    assert len(final_chunks) == 1

    final_chunk = final_chunks[0]

    # Verify final count is 5 (after both postprocessors)
    assert len(final_chunk.source_nodes) == 5
    assert final_chunk.source_nodes == filtered_nodes
