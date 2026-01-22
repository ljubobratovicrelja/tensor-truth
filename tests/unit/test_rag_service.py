"""Unit tests for RAG service functionality."""

from unittest.mock import Mock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode


def _create_mock_config():
    """Create a mock config with properly structured history_cleaning."""
    config = Mock()
    # Configure history_cleaning to be disabled so tests work without actual cleaning
    config.history_cleaning.enabled = False
    return config


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
    service = RAGService(_create_mock_config())
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
    service = RAGService(_create_mock_config())
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
    service = RAGService(_create_mock_config())
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


# =============================================================================
# Prompt Selection Tests
# =============================================================================


def _create_mock_engine_with_nodes(nodes, memory_history=None):
    """Helper to create a mock engine with specified nodes."""
    engine = Mock()
    engine._retriever.retrieve.return_value = nodes
    engine._node_postprocessors = []  # No postprocessors by default

    # Mock LLM
    engine._llm = Mock()
    engine._llm.thinking = False

    # Mock streaming response
    mock_chunk = Mock()
    mock_chunk.delta = "Test response"
    mock_chunk.additional_kwargs = {}
    engine._llm.stream_chat.return_value = [mock_chunk]

    # Mock memory
    engine.memory = Mock()
    engine.memory.get.return_value = memory_history or []
    engine.memory.put = Mock()

    return engine


@pytest.mark.unit
def test_prompt_selection_no_sources():
    """Test that NO_SOURCES prompt is used when all sources are filtered out."""
    from tensortruth.services.rag_service import RAGService

    # Create engine with empty nodes (simulating all filtered out)
    engine = _create_mock_engine_with_nodes([])

    # Create service with confidence threshold
    service = RAGService(_create_mock_config())
    service._engine = engine
    service._current_params = {"confidence_cutoff": 0.5}

    # Execute query
    list(service.query("How do I make a carrot cake?"))

    # Verify LLM was called
    assert engine._llm.stream_chat.called

    # Get the messages passed to stream_chat
    call_args = engine._llm.stream_chat.call_args
    messages = call_args[0][0]

    # The last message should contain the formatted prompt
    prompt_content = messages[-1].content

    # Verify NO_SOURCES prompt markers are present
    assert "NO RETRIEVED DOCUMENTS" in prompt_content
    assert "SYSTEM ALERT: The knowledge base returned zero matches" in prompt_content
    assert (
        "MANDATORY FORMATTING: Start your response with one of the following labels"
        in prompt_content
    )
    assert "NO INDEXED DATA FOUND" in prompt_content


@pytest.mark.unit
def test_prompt_selection_low_confidence():
    """Test that LOW_CONFIDENCE prompt is used when best score is below threshold."""
    from tensortruth.services.rag_service import RAGService

    # Create nodes with low scores (below 0.5 threshold)
    low_score_nodes = [
        NodeWithScore(node=TextNode(text="Some irrelevant content"), score=0.3),
        NodeWithScore(node=TextNode(text="Another weak match"), score=0.25),
    ]

    engine = _create_mock_engine_with_nodes(low_score_nodes)

    # Create service with confidence threshold higher than best score
    service = RAGService(_create_mock_config())
    service._engine = engine
    service._current_params = {"confidence_cutoff": 0.5}

    # Execute query
    list(service.query("What is attention mechanism?"))

    # Verify LLM was called
    assert engine._llm.stream_chat.called

    # Get the messages passed to stream_chat
    call_args = engine._llm.stream_chat.call_args
    messages = call_args[0][0]
    prompt_content = messages[-1].content

    # Verify LOW_CONFIDENCE prompt markers are present
    assert "LOW CONFIDENCE MATCH" in prompt_content
    assert "DATA INTEGRITY WARNING" in prompt_content
    assert "RETRIEVED CONTEXT (LOW RELEVANCE)" in prompt_content
    assert "MANDATORY PREFACE" in prompt_content
    # Verify context is still included
    assert "Some irrelevant content" in prompt_content


@pytest.mark.unit
def test_prompt_selection_good_confidence():
    """Test that normal prompt is used when sources have good confidence."""
    from tensortruth.services.rag_service import RAGService

    # Create nodes with good scores (above 0.5 threshold)
    good_score_nodes = [
        NodeWithScore(
            node=TextNode(text="Highly relevant transformer content"), score=0.85
        ),
        NodeWithScore(
            node=TextNode(text="Another good match about attention"), score=0.75
        ),
    ]

    engine = _create_mock_engine_with_nodes(good_score_nodes)

    # Create service with confidence threshold below best score
    service = RAGService(_create_mock_config())
    service._engine = engine
    service._current_params = {"confidence_cutoff": 0.5}

    # Execute query
    list(service.query("Explain transformers"))

    # Verify LLM was called
    assert engine._llm.stream_chat.called

    # Get the messages passed to stream_chat
    call_args = engine._llm.stream_chat.call_args
    messages = call_args[0][0]
    prompt_content = messages[-1].content

    # Verify normal prompt markers are present (not low confidence or no sources)
    assert "Technical Research & Development Assistant" in prompt_content
    assert "CONTEXT START" in prompt_content
    assert "CONTEXT END" in prompt_content
    # Verify context is included
    assert "Highly relevant transformer content" in prompt_content
    # Verify LOW_CONFIDENCE markers are NOT present
    assert "LOW CONFIDENCE MATCH" not in prompt_content
    assert "NO RETRIEVED DOCUMENTS" not in prompt_content


@pytest.mark.unit
def test_prompt_selection_no_threshold():
    """Test that normal prompt is used when no confidence threshold is set."""
    from tensortruth.services.rag_service import RAGService

    # Create nodes with low scores
    low_score_nodes = [
        NodeWithScore(node=TextNode(text="Content about neural networks"), score=0.3),
    ]

    engine = _create_mock_engine_with_nodes(low_score_nodes)

    # Create service WITHOUT confidence threshold (0.0 or not set)
    service = RAGService(_create_mock_config())
    service._engine = engine
    service._current_params = {"confidence_cutoff": 0.0}  # No threshold

    # Execute query
    list(service.query("What are neural networks?"))

    # Verify LLM was called
    assert engine._llm.stream_chat.called

    # Get the messages passed to stream_chat
    call_args = engine._llm.stream_chat.call_args
    messages = call_args[0][0]
    prompt_content = messages[-1].content

    # Should use normal prompt even with low scores (no threshold configured)
    assert "CONTEXT START" in prompt_content
    assert "Content about neural networks" in prompt_content
    # Should NOT use low confidence prompt
    assert "LOW CONFIDENCE MATCH" not in prompt_content


@pytest.mark.unit
def test_prompt_selection_filters_all_nodes():
    """Test NO_SOURCES prompt when postprocessors filter out all nodes."""
    from tensortruth.services.rag_service import RAGService

    # Create nodes that will be filtered out
    nodes = [
        NodeWithScore(node=TextNode(text="Some content"), score=0.4),
        NodeWithScore(node=TextNode(text="More content"), score=0.3),
    ]

    engine = _create_mock_engine_with_nodes(nodes)

    # Add a postprocessor that filters everything out
    filter_all = Mock()
    filter_all.postprocess_nodes.return_value = []  # Filter all nodes
    engine._node_postprocessors = [filter_all]

    # Create service
    service = RAGService(_create_mock_config())
    service._engine = engine
    service._current_params = {"confidence_cutoff": 0.5}

    # Execute query
    list(service.query("Tell me about deep learning"))

    # Verify LLM was called
    assert engine._llm.stream_chat.called

    # Get the messages passed to stream_chat
    call_args = engine._llm.stream_chat.call_args
    messages = call_args[0][0]
    prompt_content = messages[-1].content

    # Should use NO_SOURCES prompt since all were filtered
    assert "NO RETRIEVED DOCUMENTS" in prompt_content
    assert "knowledge base returned zero matches" in prompt_content


@pytest.mark.unit
def test_prompt_includes_chat_history():
    """Test that chat history is included in the prompt."""
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    from tensortruth.services.rag_service import RAGService

    # Create good nodes
    nodes = [
        NodeWithScore(node=TextNode(text="Content about CNNs"), score=0.8),
    ]

    # Create chat history
    chat_history = [
        ChatMessage(role=MessageRole.USER, content="What is a CNN?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="A CNN is a convolutional neural network.",
        ),
    ]

    engine = _create_mock_engine_with_nodes(nodes, memory_history=chat_history)

    # Create service
    service = RAGService(_create_mock_config())
    service._engine = engine
    service._current_params = {"confidence_cutoff": 0.5}

    # Execute query
    list(service.query("Tell me more about them"))

    # Verify LLM was called
    assert engine._llm.stream_chat.called

    # Get the messages passed to stream_chat
    call_args = engine._llm.stream_chat.call_args
    messages = call_args[0][0]
    prompt_content = messages[-1].content

    # Verify chat history is in the prompt
    assert "What is a CNN?" in prompt_content
    assert "convolutional neural network" in prompt_content
