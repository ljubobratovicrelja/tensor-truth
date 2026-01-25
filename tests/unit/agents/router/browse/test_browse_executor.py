"""Tests for BrowseExecutor."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tensortruth.agents.router.browse.executor import BrowseExecutor
from tensortruth.agents.router.browse.state import BrowseState, WorkflowPhase


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    search_tool = MagicMock()
    search_tool.async_fn = AsyncMock()
    search_tool.metadata.name = "search_web"

    fetch_tool = MagicMock()
    fetch_tool.async_fn = AsyncMock()
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
    mock_tools["search_web"].async_fn.return_value = json.dumps(search_results)

    # Execute search
    updated_state = await executor.execute_search(initial_state)

    # Verify tool was called with multiple queries
    mock_tools["search_web"].async_fn.assert_called_once()
    call_args = mock_tools["search_web"].async_fn.call_args
    assert "queries" in call_args.kwargs
    queries = call_args.kwargs["queries"]
    assert len(queries) == 3
    assert "neural networks" in queries[0]

    # Verify state updates
    assert updated_state.search_results == search_results
    assert updated_state.phase == WorkflowPhase.SEARCHED
    assert "search_web" in updated_state.actions_taken


@pytest.mark.asyncio
async def test_execute_fetch_with_overflow(executor, mock_tools):
    """Test that execute_fetch handles overflow correctly."""
    # Create state with search results
    state = BrowseState(
        query="test query",
        phase=WorkflowPhase.SEARCHED,
        min_pages_required=3,
        max_content_chars=10000,
        search_results=[
            {"url": "https://example.com/1", "title": "Result 1"},
            {"url": "https://example.com/2", "title": "Result 2"},
            {"url": "https://example.com/3", "title": "Result 3"},
        ],
    )

    # Mock fetch result with overflow
    fetch_result = {
        "pages": [
            {"url": "https://example.com/1", "status": "success", "content": "..."},
            {"url": "https://example.com/2", "status": "success", "content": "..."},
        ],
        "overflow": True,
        "total_chars": 12000,
    }
    mock_tools["fetch_pages_batch"].async_fn.return_value = json.dumps(fetch_result)

    # Execute fetch
    updated_state = await executor.execute_fetch(state)

    # Verify tool was called with overflow protection
    mock_tools["fetch_pages_batch"].async_fn.assert_called_once()
    call_args = mock_tools["fetch_pages_batch"].async_fn.call_args
    assert call_args.kwargs["max_content_chars"] == 10000
    assert call_args.kwargs["min_pages"] == 3

    # Verify state updates
    assert updated_state.pages == fetch_result["pages"]
    assert updated_state.content_overflow is True
    assert updated_state.total_content_chars == 12000
    assert updated_state.phase == WorkflowPhase.FETCHED
    assert updated_state.fetch_iterations == 1
    assert "fetch_pages_batch" in updated_state.actions_taken


@pytest.mark.asyncio
async def test_execute_fetch_without_overflow(executor, mock_tools):
    """Test that execute_fetch works without overflow."""
    # Create state with search results
    state = BrowseState(
        query="test query",
        phase=WorkflowPhase.SEARCHED,
        min_pages_required=3,
        max_content_chars=50000,
        search_results=[
            {"url": "https://example.com/1", "title": "Result 1"},
            {"url": "https://example.com/2", "title": "Result 2"},
            {"url": "https://example.com/3", "title": "Result 3"},
        ],
    )

    # Mock fetch result without overflow
    fetch_result = {
        "pages": [
            {"url": "https://example.com/1", "status": "success", "content": "..."},
            {"url": "https://example.com/2", "status": "success", "content": "..."},
            {"url": "https://example.com/3", "status": "success", "content": "..."},
        ],
        "overflow": False,
        "total_chars": 8000,
    }
    mock_tools["fetch_pages_batch"].async_fn.return_value = json.dumps(fetch_result)

    # Execute fetch
    updated_state = await executor.execute_fetch(state)

    # Verify state updates
    assert updated_state.content_overflow is False
    assert updated_state.total_content_chars == 8000
    assert len(updated_state.pages) == 3


@pytest.mark.asyncio
async def test_executor_tracks_iterations(executor, mock_tools):
    """Test that executor tracks fetch iterations correctly."""
    state = BrowseState(
        query="test query",
        phase=WorkflowPhase.SEARCHED,
        min_pages_required=3,
        max_content_chars=50000,
        search_results=[
            {"url": f"https://example.com/{i}", "title": f"Result {i}"}
            for i in range(10)
        ],
        fetch_iterations=0,
    )

    fetch_result = {
        "pages": [
            {"url": "https://example.com/1", "status": "success", "content": "..."}
        ],
        "overflow": False,
        "total_chars": 1000,
    }
    mock_tools["fetch_pages_batch"].async_fn.return_value = json.dumps(fetch_result)

    # Execute fetch multiple times
    state = await executor.execute_fetch(state)
    assert state.fetch_iterations == 1

    state = await executor.execute_fetch(state)
    assert state.fetch_iterations == 2

    state = await executor.execute_fetch(state)
    assert state.fetch_iterations == 3


@pytest.mark.asyncio
async def test_execute_fetch_selects_correct_urls(executor, mock_tools):
    """Test that executor selects the right URLs based on next_url_index."""
    state = BrowseState(
        query="test query",
        phase=WorkflowPhase.SEARCHED,
        min_pages_required=3,
        max_content_chars=50000,
        search_results=[
            {"url": f"https://example.com/{i}", "title": f"Result {i}"}
            for i in range(10)
        ],
        next_url_index=0,
    )

    fetch_result = {
        "pages": [],
        "overflow": False,
        "total_chars": 0,
    }
    mock_tools["fetch_pages_batch"].async_fn.return_value = json.dumps(fetch_result)

    # First fetch should get URLs 0-2
    await executor.execute_fetch(state)
    call_args = mock_tools["fetch_pages_batch"].async_fn.call_args
    urls = call_args.kwargs["urls"]
    assert urls == [
        "https://example.com/0",
        "https://example.com/1",
        "https://example.com/2",
    ]


def test_generate_queries(executor):
    """Test that executor generates diverse queries."""
    queries = executor._generate_queries("neural networks")

    assert len(queries) == 3
    assert "neural networks" in queries[0]
    assert "neural networks" in queries[1]
    assert "neural networks" in queries[2]
    # Check for diversity
    assert "overview" in queries[0] or "technical" in queries[1] or "2026" in queries[2]
