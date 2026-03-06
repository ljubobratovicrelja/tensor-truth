"""Unit tests for RAG service functionality."""

from unittest.mock import Mock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode


def _create_mock_config():
    """Create a mock config with properly structured history_cleaning."""
    config = Mock()
    # Configure history_cleaning to be disabled so tests work without actual cleaning
    config.history_cleaning.enabled = False
    # Add max_history_turns for ChatHistoryService
    config.conversation.max_history_turns = 3
    return config


def _set_mock_engine(service, engine):
    """Set mock engine and cache its components on the service.

    Helper to properly initialize cached fields when bypassing load_engine().
    """
    service._engine = engine
    service._llm = engine._llm
    service._retriever = engine._retriever


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

    # Create service and set engine
    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)

    # Execute query (no session_messages, no history)
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

    # Create service and set engine
    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)

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

    # Create service and set engine
    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)

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


def _create_mock_engine_with_nodes(nodes):
    """Helper to create a mock engine with specified nodes."""
    engine = Mock()
    engine._retriever.retrieve.return_value = nodes
    engine._node_postprocessors = []  # No postprocessors by default

    # Mock LLM with proper attributes for condenser
    engine._llm = Mock()
    engine._llm.thinking = False
    engine._llm.base_url = "http://localhost:11434"
    engine._llm.model = "llama3.1:8b"
    engine._llm.temperature = 0.2
    engine._llm.request_timeout = 120.0

    # Mock streaming response
    mock_chunk = Mock()
    mock_chunk.delta = "Test response"
    mock_chunk.additional_kwargs = {}
    engine._llm.stream_chat.return_value = [mock_chunk]

    return engine


@pytest.mark.unit
def test_prompt_selection_no_sources():
    """Test that NO_SOURCES prompt is used when all sources are filtered out."""
    from tensortruth.services.rag_service import RAGService

    # Create engine with empty nodes (simulating all filtered out)
    engine = _create_mock_engine_with_nodes([])

    # Create service with confidence threshold
    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)
    service._current_params = {"confidence_cutoff": 0.5}

    # Execute query (no session_messages, no history)
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
    _set_mock_engine(service, engine)
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
    assert "INTEGRITY CHECK" in prompt_content
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
    _set_mock_engine(service, engine)
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
    _set_mock_engine(service, engine)
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
    _set_mock_engine(service, engine)
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
    """Test that chat history is included in the prompt when passed as session_messages."""
    from unittest.mock import patch

    from tensortruth.services.rag_service import RAGService

    # Create good nodes
    nodes = [
        NodeWithScore(node=TextNode(text="Content about CNNs"), score=0.8),
    ]

    engine = _create_mock_engine_with_nodes(nodes)

    # Create service
    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)
    service._current_params = {"confidence_cutoff": 0.5}

    # Create session messages (the new API uses dicts, not ChatMessage)
    session_messages = [
        {"role": "user", "content": "What is a CNN?"},
        {"role": "assistant", "content": "A CNN is a convolutional neural network."},
    ]

    with patch("tensortruth.services.rag_service.create_condenser_llm") as mock_ccl:
        mock_ccl.return_value = Mock()
        # Execute query with session_messages
        list(
            service.query("Tell me more about them", session_messages=session_messages)
        )

    # Verify LLM was called
    assert engine._llm.stream_chat.called

    # Get the messages passed to stream_chat
    call_args = engine._llm.stream_chat.call_args
    messages = call_args[0][0]
    prompt_content = messages[-1].content

    # Verify chat history is in the prompt
    assert "What is a CNN?" in prompt_content
    assert "convolutional neural network" in prompt_content


@pytest.mark.unit
def test_query_with_empty_session_messages():
    """Test that query works with empty session_messages."""
    from tensortruth.services.rag_service import RAGService

    nodes = [
        NodeWithScore(node=TextNode(text="Content"), score=0.8),
    ]

    engine = _create_mock_engine_with_nodes(nodes)

    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)
    service._current_params = {}

    # Execute query with empty session_messages
    results = list(service.query("Test query", session_messages=[]))

    # Should complete without error
    final_chunks = [c for c in results if hasattr(c, "is_complete") and c.is_complete]
    assert len(final_chunks) == 1


@pytest.mark.unit
def test_query_with_none_session_messages():
    """Test that query works with None session_messages."""
    from tensortruth.services.rag_service import RAGService

    nodes = [
        NodeWithScore(node=TextNode(text="Content"), score=0.8),
    ]

    engine = _create_mock_engine_with_nodes(nodes)

    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)
    service._current_params = {}

    # Execute query with None session_messages (default)
    results = list(service.query("Test query"))

    # Should complete without error
    final_chunks = [c for c in results if hasattr(c, "is_complete") and c.is_complete]
    assert len(final_chunks) == 1


# =============================================================================
# Module Reload Detection Tests
# =============================================================================


@pytest.mark.unit
def test_needs_reload_detects_module_addition():
    """needs_reload() should return True when modules change."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())

    # Simulate loaded state with module_a
    service._engine = Mock()  # Engine is "loaded"
    service._current_modules = ["module_a"]
    service._current_params = {"model": "test-model", "temperature": 0.3}
    service._current_config_hash = service._compute_config_hash(
        ["module_a"], service._current_params, None
    )

    # Adding module_b should trigger reload
    assert service.needs_reload(
        ["module_a", "module_b"], service._current_params, None
    ), "needs_reload() should return True when modules are added"


@pytest.mark.unit
def test_needs_reload_detects_module_removal():
    """needs_reload() should return True when modules are removed."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())

    # Simulate loaded state with two modules
    service._engine = Mock()
    service._current_modules = ["module_a", "module_b"]
    service._current_params = {"model": "test-model"}
    service._current_config_hash = service._compute_config_hash(
        ["module_a", "module_b"], service._current_params, None
    )

    # Removing module_b should trigger reload
    assert service.needs_reload(
        ["module_a"], service._current_params, None
    ), "needs_reload() should return True when modules are removed"


@pytest.mark.unit
def test_needs_reload_same_modules_no_reload():
    """needs_reload() should return False when modules haven't changed."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())

    # Simulate loaded state
    service._engine = Mock()
    service._current_modules = ["module_a", "module_b"]
    service._current_params = {"model": "test-model", "temperature": 0.3}
    service._current_config_hash = service._compute_config_hash(
        ["module_a", "module_b"], service._current_params, None
    )

    # Same modules (different order) should NOT trigger reload
    assert not service.needs_reload(
        ["module_b", "module_a"], service._current_params, None
    ), "needs_reload() should return False when modules are same (different order)"


@pytest.mark.unit
def test_config_hash_with_nested_params():
    """Config hash should work with nested dicts in params (not raise TypeError)."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())

    # Params with nested dict - this would fail with frozenset()
    nested_params = {
        "model": "test-model",
        "temperature": 0.3,
        "nested": {"key1": "value1", "key2": 123},
        "list_param": [1, 2, 3],
    }

    # This should not raise TypeError: unhashable type: 'dict'
    hash_result = service._compute_config_hash(["module_a"], nested_params, None)

    # Should return a valid tuple
    assert hash_result is not None
    assert isinstance(hash_result, tuple)


@pytest.mark.unit
def test_config_hash_consistency_with_nested_params():
    """Same nested params should produce the same hash."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())

    nested_params_1 = {
        "model": "test-model",
        "nested": {"a": 1, "b": 2},
    }
    nested_params_2 = {
        "model": "test-model",
        "nested": {"b": 2, "a": 1},  # Same dict, different key order
    }

    hash1 = service._compute_config_hash(["module_a"], nested_params_1, None)
    hash2 = service._compute_config_hash(["module_a"], nested_params_2, None)

    # Hashes should be equal for equivalent params
    assert (
        hash1 == hash2
    ), "Config hash should be consistent for equivalent nested params"


@pytest.mark.unit
def test_needs_reload_with_additional_index_addition():
    """needs_reload() should return True when additional index is added."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())

    # Simulate loaded state without session index
    service._engine = Mock()
    service._current_modules = ["module_a"]
    service._current_params = {"model": "test-model"}
    service._current_config_hash = service._compute_config_hash(
        ["module_a"],
        service._current_params,
        None,  # No session index
    )

    # Adding session index should trigger reload
    assert service.needs_reload(
        ["module_a"], service._current_params, ["/path/to/session/index"]
    ), "needs_reload() should return True when additional index is added"


@pytest.mark.unit
def test_config_hash_different_paths_produce_different_hashes():
    """Different additional_index_paths must produce different hashes (bug regression)."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())
    params = {"model": "test-model"}

    hash_a = service._compute_config_hash(
        ["module_a"], params, ["/path/to/project_A/index"]
    )
    hash_b = service._compute_config_hash(
        ["module_a"], params, ["/path/to/project_B/index"]
    )

    assert hash_a != hash_b, (
        "Different index paths must produce different hashes "
        "(old bool-based hash would have treated these as equal)"
    )


@pytest.mark.unit
def test_config_hash_empty_paths_vs_none():
    """Empty list and None should produce equivalent hashes."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())
    params = {"model": "test-model"}

    hash_empty = service._compute_config_hash(["module_a"], params, [])
    hash_none = service._compute_config_hash(["module_a"], params, None)

    assert hash_empty == hash_none, "[] and None should be treated equivalently"


@pytest.mark.unit
def test_config_hash_path_order_independent():
    """Path order should not affect the hash (sorted internally)."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())
    params = {"model": "test-model"}

    hash_ab = service._compute_config_hash(["module_a"], params, ["/path/a", "/path/b"])
    hash_ba = service._compute_config_hash(["module_a"], params, ["/path/b", "/path/a"])

    assert hash_ab == hash_ba, "Path order should not affect config hash"


@pytest.mark.unit
def test_config_hash_multiple_paths():
    """Adding an extra path should change the hash."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())
    params = {"model": "test-model"}

    hash_one = service._compute_config_hash(["module_a"], params, ["/path/a"])
    hash_two = service._compute_config_hash(
        ["module_a"], params, ["/path/a", "/path/b"]
    )

    assert hash_one != hash_two, "Adding a path should change the hash"


@pytest.mark.unit
def test_needs_reload_detects_index_path_change():
    """Switching projects (different index paths) should trigger reload."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())

    # Simulate loaded state with project A's index
    service._engine = Mock()
    service._current_modules = ["module_a"]
    service._current_params = {"model": "test-model"}
    service._current_config_hash = service._compute_config_hash(
        ["module_a"], service._current_params, ["/projects/A/index"]
    )

    # Switching to project B's index should trigger reload
    assert service.needs_reload(
        ["module_a"], service._current_params, ["/projects/B/index"]
    ), "Switching project index paths should trigger reload"


@pytest.mark.unit
def test_condenser_called_with_history():
    """Test that condense_query is called when history is present."""
    from unittest.mock import patch

    from tensortruth.services.rag_service import RAGService

    # Create service with mock engine
    service = RAGService(_create_mock_config())
    engine = Mock()
    engine._retriever = Mock()
    engine._retriever.retrieve.return_value = []
    engine._node_postprocessors = []
    engine._llm = Mock()
    engine._llm.thinking = False
    engine._llm.base_url = "http://localhost:11434"
    engine._llm.model = "llama3.1:8b"
    engine._llm.temperature = 0.2
    engine._llm.request_timeout = 120.0
    engine._llm.stream_chat.return_value = []

    # Mock condenser
    condenser = Mock()
    condenser.template = "History: {chat_history}\nQuestion: {question}"
    engine._condense_prompt_template = condenser

    _set_mock_engine(service, engine)

    # Mock session messages (non-empty history)
    session_messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]

    with (
        patch("tensortruth.services.rag_service.create_condenser_llm") as mock_ccl,
        patch("tensortruth.services.rag_service.condense_query") as mock_condense,
    ):
        mock_ccl.return_value = Mock()
        mock_condense.return_value = "Condensed query"

        # Execute query with history
        list(service.query("Follow-up question", session_messages=session_messages))

        # Verify condense_query was called
        assert mock_condense.called
        call_args = mock_condense.call_args
        assert call_args[1]["question"] == "Follow-up question"
        assert call_args[1]["fallback_on_error"] is True


@pytest.mark.unit
def test_condenser_skipped_with_empty_history():
    """Test that condensation is skipped when history is empty."""
    from unittest.mock import patch

    from tensortruth.services.rag_service import RAGService

    # Create service with mock engine
    service = RAGService(_create_mock_config())
    engine = Mock()
    engine._retriever = Mock()
    engine._retriever.retrieve.return_value = []
    engine._node_postprocessors = []
    engine._llm = Mock()
    engine._llm.thinking = False
    engine._llm.base_url = "http://localhost:11434"
    engine._llm.model = "llama3.1:8b"
    engine._llm.temperature = 0.2
    engine._llm.request_timeout = 120.0
    engine._llm.stream_chat.return_value = []

    # Mock condenser
    condenser = Mock()
    condenser.template = "History: {chat_history}\nQuestion: {question}"
    engine._condense_prompt_template = condenser

    _set_mock_engine(service, engine)

    with patch("tensortruth.services.rag_service.condense_query") as mock_condense:
        mock_condense.return_value = "Condensed query"

        # Execute query WITHOUT history (empty session_messages)
        list(service.query("First question", session_messages=[]))

        # Verify condense_query was NOT called (history is empty)
        assert not mock_condense.called


@pytest.mark.unit
def test_condenser_error_handling():
    """Test that condenser errors are handled gracefully with fallback."""
    from unittest.mock import patch

    from tensortruth.services.rag_service import RAGService

    # Create service with mock engine
    service = RAGService(_create_mock_config())
    engine = Mock()
    engine._retriever = Mock()
    engine._retriever.retrieve.return_value = []
    engine._node_postprocessors = []
    engine._llm = Mock()
    engine._llm.thinking = False
    engine._llm.base_url = "http://localhost:11434"
    engine._llm.model = "llama3.1:8b"
    engine._llm.temperature = 0.2
    engine._llm.request_timeout = 120.0

    # Mock streaming response
    mock_chunk = Mock()
    mock_chunk.delta = "test response"
    mock_chunk.additional_kwargs = {}
    engine._llm.stream_chat.return_value = [mock_chunk]

    # Mock condenser
    condenser = Mock()
    condenser.template = "History: {chat_history}\nQuestion: {question}"
    engine._condense_prompt_template = condenser

    _set_mock_engine(service, engine)

    # Mock session messages (non-empty history)
    session_messages = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]

    with (
        patch("tensortruth.services.rag_service.create_condenser_llm") as mock_ccl,
        patch("tensortruth.services.rag_service.condense_query") as mock_condense,
    ):
        mock_ccl.return_value = Mock()
        # Make condense_query return the original question (fallback behavior)
        mock_condense.return_value = "Follow-up question"

        # Execute query - should not raise even if condenser has issues
        results = list(
            service.query("Follow-up question", session_messages=session_messages)
        )

        # Verify query completed successfully
        assert len(results) > 0


# =============================================================================
# Retrieve Method Tests (Story 3: RAG as a Retrieval Tool)
# =============================================================================


@pytest.mark.unit
def test_retrieve_returns_result_with_sources():
    """retrieve() should return RAGRetrievalResult with source nodes and metrics."""
    from tensortruth.services.rag_service import RAGService

    nodes = [
        NodeWithScore(node=TextNode(text="Content about transformers"), score=0.85),
        NodeWithScore(node=TextNode(text="More about attention"), score=0.75),
    ]

    engine = _create_mock_engine_with_nodes(nodes)
    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)
    service._current_params = {"confidence_cutoff": 0.5}

    result = service.retrieve("What are transformers?")

    assert result.num_sources == 2
    assert result.confidence_level == "normal"
    assert result.metrics is not None
    assert result.source_nodes == nodes
    assert result.condensed_query == "What are transformers?"


@pytest.mark.unit
def test_retrieve_low_confidence():
    """retrieve() should report low confidence when best score is below threshold."""
    from tensortruth.services.rag_service import RAGService

    nodes = [
        NodeWithScore(node=TextNode(text="Weak match"), score=0.3),
    ]

    engine = _create_mock_engine_with_nodes(nodes)
    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)
    service._current_params = {"confidence_cutoff": 0.5}

    result = service.retrieve("Obscure topic")

    assert result.confidence_level == "low"
    assert result.num_sources == 1


@pytest.mark.unit
def test_retrieve_no_sources():
    """retrieve() should report 'none' confidence when no sources returned."""
    from tensortruth.services.rag_service import RAGService

    engine = _create_mock_engine_with_nodes([])
    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)
    service._current_params = {}

    result = service.retrieve("Nonexistent topic")

    assert result.confidence_level == "none"
    assert result.num_sources == 0
    assert result.source_nodes == []


@pytest.mark.unit
def test_retrieve_no_engine():
    """retrieve() should return empty result when no engine is loaded."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())

    result = service.retrieve("Any query")

    assert result.confidence_level == "none"
    assert result.num_sources == 0
    assert result.source_nodes == []


@pytest.mark.unit
def test_retrieve_applies_postprocessors():
    """retrieve() should apply postprocessors (reranking)."""
    from tensortruth.services.rag_service import RAGService

    raw_nodes = [
        NodeWithScore(node=TextNode(text=f"Node {i}"), score=0.9 - i * 0.05)
        for i in range(10)
    ]

    reranked_nodes = [
        NodeWithScore(node=TextNode(text=f"Reranked {i}"), score=0.95 - i * 0.05)
        for i in range(5)
    ]

    engine = _create_mock_engine_with_nodes(raw_nodes)
    reranker = Mock()
    reranker.postprocess_nodes.return_value = reranked_nodes
    engine._node_postprocessors = [reranker]

    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)
    service._current_params = {}

    result = service.retrieve("Test reranking")

    assert reranker.postprocess_nodes.called
    assert result.num_sources == 5
    assert result.source_nodes == reranked_nodes


@pytest.mark.unit
def test_retrieve_emits_progress():
    """retrieve() should call progress_callback with ToolProgress objects."""
    from tensortruth.services.rag_service import RAGService

    nodes = [
        NodeWithScore(node=TextNode(text="Content"), score=0.8),
    ]

    engine = _create_mock_engine_with_nodes(nodes)
    engine._node_postprocessors = [Mock(postprocess_nodes=Mock(return_value=nodes))]

    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)
    service._current_params = {}

    progress_reports = []
    service.retrieve("Test", progress_callback=progress_reports.append)

    # Should have at least "retrieving" and "reranking" phases
    phases = [p.phase for p in progress_reports]
    assert "retrieving" in phases
    assert "reranking" in phases


@pytest.mark.unit
def test_retrieve_does_not_call_llm():
    """retrieve() must NOT invoke the LLM (no stream_chat, no chat calls)."""
    from tensortruth.services.rag_service import RAGService

    nodes = [
        NodeWithScore(node=TextNode(text="Content"), score=0.8),
    ]

    engine = _create_mock_engine_with_nodes(nodes)
    service = RAGService(_create_mock_config())
    _set_mock_engine(service, engine)
    service._current_params = {}

    service.retrieve("Should not call LLM")

    engine._llm.stream_chat.assert_not_called()
    if hasattr(engine._llm, "chat"):
        engine._llm.chat.assert_not_called()


@pytest.mark.unit
def test_load_engine_caches_llm_and_retriever():
    """Verify load_engine caches LLM and retriever."""
    from unittest.mock import patch

    from tensortruth.services.rag_service import RAGService

    with patch("tensortruth.services.rag_service.load_engine_for_modules") as mock_load:
        mock_engine = Mock()
        mock_engine._llm = Mock()
        mock_engine._retriever = Mock()
        mock_load.return_value = mock_engine

        service = RAGService(_create_mock_config())
        service.load_engine(["test"], {})

        assert service._llm is mock_engine._llm
        assert service._retriever is mock_engine._retriever


@pytest.mark.unit
def test_clear_resets_cached_components():
    """Verify clear() resets cached LLM and retriever."""
    from tensortruth.services.rag_service import RAGService

    service = RAGService(_create_mock_config())
    service._llm = Mock()
    service._retriever = Mock()
    service._engine = Mock()

    service.clear()

    assert service._llm is None
    assert service._retriever is None
