"""Tests for BrowseExecutor."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensortruth.agents.router.browse.executor import BrowseExecutor
from tensortruth.agents.router.browse.state import BrowseState, WorkflowPhase


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    search_tool = MagicMock()
    search_tool.async_fn = AsyncMock()
    search_tool.acall = AsyncMock()
    search_tool.metadata.name = "search_web"

    fetch_tool = MagicMock()
    fetch_tool.async_fn = AsyncMock()
    fetch_tool.acall = AsyncMock()
    fetch_tool.metadata.name = "fetch_pages_batch"

    return {
        "search_web": search_tool,
        "fetch_pages_batch": fetch_tool,
    }


@pytest.fixture
def executor(mock_tools):
    """Create BrowseExecutor instance."""
    return BrowseExecutor(mock_tools)


@pytest.fixture
def initial_state():
    """Create initial browse state."""
    return BrowseState(
        query="what are neural networks",
        phase=WorkflowPhase.INITIAL,
        min_pages_required=3,
        max_content_chars=50000,
    )


@pytest.mark.asyncio
async def test_execute_search_updates_state(executor, initial_state, mock_tools):
    """Test that execute_search updates state correctly."""
    # Mock search results
    search_results = [
        {"url": "https://example.com/1", "title": "Result 1", "snippet": "..."},
        {"url": "https://example.com/2", "title": "Result 2", "snippet": "..."},
        {"url": "https://example.com/3", "title": "Result 3", "snippet": "..."},
    ]
    mock_tools["search_web"].acall.return_value = json.dumps(search_results)

    # Execute search
    updated_state = await executor.execute_search(initial_state)

    # Verify tool was called with multiple queries
    mock_tools["search_web"].acall.assert_called_once()
    call_args = mock_tools["search_web"].acall.call_args
    assert "queries" in call_args.kwargs
    queries = call_args.kwargs["queries"]
    assert len(queries) == 3
    assert "neural networks" in queries[0]

    # Verify state updates
    assert updated_state.search_results == search_results
    assert updated_state.phase == WorkflowPhase.SEARCHED
    assert "search_web" in updated_state.actions_taken


@pytest.mark.asyncio
async def test_execute_fetch_with_pipeline(executor, mock_tools):
    """Test that execute_fetch uses SourceFetchPipeline correctly."""
    from tensortruth.core.source import SourceNode, SourceStatus, SourceType

    # Create state with search results
    state = BrowseState(
        query="test query",
        phase=WorkflowPhase.SEARCHED,
        min_pages_required=3,
        max_content_chars=10000,
        search_results=[
            {"url": "https://example.com/1", "title": "Result 1", "snippet": "..."},
            {"url": "https://example.com/2", "title": "Result 2", "snippet": "..."},
            {"url": "https://example.com/3", "title": "Result 3", "snippet": "..."},
        ],
    )

    # Mock pipeline results
    mock_fitted_pages = [
        ("https://example.com/1", "Result 1", "Content 1"),
        ("https://example.com/2", "Result 2", "Content 2"),
    ]
    mock_source_nodes = [
        SourceNode(
            id="1",
            url="https://example.com/1",
            title="Result 1",
            source_type=SourceType.WEB,
            status=SourceStatus.SUCCESS,
            content="Content 1",
            content_chars=1000,
        ),
        SourceNode(
            id="2",
            url="https://example.com/2",
            title="Result 2",
            source_type=SourceType.WEB,
            status=SourceStatus.SUCCESS,
            content="Content 2",
            content_chars=1000,
        ),
    ]
    mock_allocations = {
        "https://example.com/1": 1000,
        "https://example.com/2": 1000,
    }

    with patch("tensortruth.core.source_pipeline.SourceFetchPipeline") as MockPipeline:
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.execute = AsyncMock(
            return_value=(mock_fitted_pages, mock_source_nodes, mock_allocations)
        )
        MockPipeline.return_value = mock_pipeline_instance

        # Execute fetch
        updated_state = await executor.execute_fetch(state)

        # Verify pipeline was created with correct params
        MockPipeline.assert_called_once()
        call_kwargs = MockPipeline.call_args.kwargs
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["max_pages"] == 3

        # Verify state updates
        assert len(updated_state.pages) == 2
        assert updated_state.total_content_chars == 2000
        assert updated_state.phase == WorkflowPhase.FETCHED
        assert updated_state.fetch_iterations == 1
        assert "fetch_sources" in updated_state.actions_taken


@pytest.mark.asyncio
async def test_execute_fetch_empty_results(executor, mock_tools):
    """Test that execute_fetch handles empty pipeline results."""
    # Create state with search results
    state = BrowseState(
        query="test query",
        phase=WorkflowPhase.SEARCHED,
        min_pages_required=3,
        max_content_chars=50000,
        search_results=[
            {"url": "https://example.com/1", "title": "Result 1", "snippet": "..."},
        ],
    )

    with patch("tensortruth.core.source_pipeline.SourceFetchPipeline") as MockPipeline:
        mock_pipeline_instance = MagicMock()
        # Return empty results (all pages failed)
        mock_pipeline_instance.execute = AsyncMock(return_value=([], [], {}))
        MockPipeline.return_value = mock_pipeline_instance

        # Execute fetch
        updated_state = await executor.execute_fetch(state)

        # Verify state updates
        assert len(updated_state.pages) == 0
        assert updated_state.total_content_chars == 0
        assert updated_state.phase == WorkflowPhase.FETCHED


@pytest.mark.asyncio
async def test_executor_tracks_iterations(executor, mock_tools):
    """Test that executor tracks fetch iterations correctly."""
    from tensortruth.core.source import SourceNode, SourceStatus, SourceType

    state = BrowseState(
        query="test query",
        phase=WorkflowPhase.SEARCHED,
        min_pages_required=3,
        max_content_chars=50000,
        search_results=[
            {
                "url": f"https://example.com/{i}",
                "title": f"Result {i}",
                "snippet": "...",
            }
            for i in range(10)
        ],
        fetch_iterations=0,
    )

    mock_fitted_pages = [("https://example.com/1", "Result 1", "Content")]
    mock_source_nodes = [
        SourceNode(
            id="1",
            url="https://example.com/1",
            title="Result 1",
            source_type=SourceType.WEB,
            status=SourceStatus.SUCCESS,
            content="Content",
            content_chars=500,
        )
    ]
    mock_allocations = {"https://example.com/1": 500}

    with patch("tensortruth.core.source_pipeline.SourceFetchPipeline") as MockPipeline:
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.execute = AsyncMock(
            return_value=(mock_fitted_pages, mock_source_nodes, mock_allocations)
        )
        MockPipeline.return_value = mock_pipeline_instance

        # Execute fetch multiple times
        state = await executor.execute_fetch(state)
        assert state.fetch_iterations == 1

        state = await executor.execute_fetch(state)
        assert state.fetch_iterations == 2

        state = await executor.execute_fetch(state)
        assert state.fetch_iterations == 3


@pytest.mark.asyncio
async def test_execute_fetch_with_callbacks(executor, mock_tools):
    """Test that execute_fetch passes callbacks to pipeline."""
    from tensortruth.agents.config import AgentCallbacks

    # Create state with search results
    state = BrowseState(
        query="test query",
        phase=WorkflowPhase.SEARCHED,
        min_pages_required=3,
        max_content_chars=50000,
        search_results=[
            {"url": "https://example.com/1", "title": "Result 1", "snippet": "..."},
        ],
    )

    # Create callbacks
    progress_messages = []
    callbacks = AgentCallbacks(
        on_progress=lambda msg: progress_messages.append(msg),
    )

    with patch("tensortruth.core.source_pipeline.SourceFetchPipeline") as MockPipeline:
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.execute = AsyncMock(return_value=([], [], {}))
        MockPipeline.return_value = mock_pipeline_instance

        # Execute fetch with callbacks
        await executor.execute_fetch(state, callbacks)

        # Verify pipeline was created with progress_callback
        call_kwargs = MockPipeline.call_args.kwargs
        assert call_kwargs["progress_callback"] is not None


def test_generate_queries(executor):
    """Test that executor generates diverse queries."""
    queries = executor._generate_queries("neural networks")

    assert len(queries) == 3
    assert "neural networks" in queries[0]
    assert "neural networks" in queries[1]
    assert "neural networks" in queries[2]
    # Check for diversity
    assert "overview" in queries[0] or "technical" in queries[1] or "2026" in queries[2]
