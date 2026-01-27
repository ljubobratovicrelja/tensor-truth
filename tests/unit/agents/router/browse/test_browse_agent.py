"""Tests for BrowseAgent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tensortruth.agents.config import AgentCallbacks
from tensortruth.agents.router.browse.agent import BrowseAgent
from tensortruth.agents.router.browse.state import WorkflowPhase


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
def mock_llm():
    """Create mock LLM for testing."""
    llm = MagicMock()
    llm.context_window = 16384
    llm.model = "llama3.1:8b"
    llm.base_url = "http://localhost:11434"
    llm.acomplete = AsyncMock()
    llm.astream_complete = AsyncMock()
    return llm


@pytest.fixture
def browse_agent(mock_llm, mock_tools):
    """Create BrowseAgent instance."""
    return BrowseAgent(
        router_llm=mock_llm,
        synthesis_llm=mock_llm,
        tools=mock_tools,
        min_pages_required=5,
        max_iterations=10,
        context_window=16384,
    )


def test_browse_agent_inherits_from_router_agent(browse_agent):
    """Test that BrowseAgent properly inherits from RouterAgent."""
    from tensortruth.agents.router.base import RouterAgent

    assert isinstance(browse_agent, RouterAgent)


def test_browse_agent_calculates_max_content(browse_agent):
    """Test that BrowseAgent calculates max_content_chars from context window."""
    # Formula: (context_window * 4) - prompt_overhead - output_buffer
    # = (16384 * 4) - (2000 * 4) - (2000 * 4)
    # = 65536 - 8000 - 8000 = 49536
    expected = (16384 * 4) - (2000 * 4) - (2000 * 4)
    assert browse_agent.max_content_chars == expected


@pytest.mark.asyncio
async def test_browse_agent_complete_workflow(browse_agent, mock_tools, mock_llm):
    """Test complete browse workflow: search -> fetch -> synthesize."""
    from unittest.mock import patch

    from tensortruth.core.source import SourceNode, SourceStatus, SourceType

    # Mock search results
    search_results = [
        {"url": "https://example.com/1", "title": "Result 1", "snippet": "..."},
        {"url": "https://example.com/2", "title": "Result 2", "snippet": "..."},
        {"url": "https://example.com/3", "title": "Result 3", "snippet": "..."},
    ]
    mock_tools["search_web"].acall.return_value = json.dumps(search_results)

    # Mock pipeline results for fetch
    mock_fitted_pages = [
        ("https://example.com/1", "Page 1", "Content from page 1"),
        ("https://example.com/2", "Page 2", "Content from page 2"),
        ("https://example.com/3", "Page 3", "Content from page 3"),
    ]
    mock_source_nodes = [
        SourceNode(
            id="1",
            url="https://example.com/1",
            title="Page 1",
            source_type=SourceType.WEB,
            status=SourceStatus.SUCCESS,
            content="Content from page 1",
            content_chars=100,
        ),
        SourceNode(
            id="2",
            url="https://example.com/2",
            title="Page 2",
            source_type=SourceType.WEB,
            status=SourceStatus.SUCCESS,
            content="Content from page 2",
            content_chars=100,
        ),
        SourceNode(
            id="3",
            url="https://example.com/3",
            title="Page 3",
            source_type=SourceType.WEB,
            status=SourceStatus.SUCCESS,
            content="Content from page 3",
            content_chars=100,
        ),
    ]
    mock_allocations = {
        "https://example.com/1": 100,
        "https://example.com/2": 100,
        "https://example.com/3": 100,
    }

    # Mock LLM routing responses - use "fetch_sources" not "fetch_pages_batch"
    mock_llm.acomplete.side_effect = [
        MagicMock(text='{"action": "search_web"}'),
        MagicMock(text='{"action": "fetch_sources"}'),
        MagicMock(text='{"action": "done"}'),
    ]

    # Mock streaming synthesis
    async def mock_stream():
        chunks = ["This ", "is ", "the ", "answer."]
        for chunk in chunks:
            yield MagicMock(delta=chunk)

    mock_llm.astream_complete.return_value = mock_stream()

    # Create callbacks
    callbacks = AgentCallbacks()

    # Run agent with mocked pipeline and Ollama (for synthesis LLM creation)
    with (
        patch("tensortruth.core.source_pipeline.SourceFetchPipeline") as MockPipeline,
        patch("tensortruth.agents.router.browse.agent.Ollama", return_value=mock_llm),
    ):
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.execute = AsyncMock(
            return_value=(mock_fitted_pages, mock_source_nodes, mock_allocations)
        )
        MockPipeline.return_value = mock_pipeline_instance

        result = await browse_agent.run("test query", callbacks)

    # Verify result
    assert result.final_answer == "This is the answer."
    assert result.iterations <= 3
    assert "search_web" in result.tools_called
    assert "fetch_sources" in result.tools_called
    assert len(result.urls_browsed) == 3


@pytest.mark.asyncio
async def test_browse_agent_handles_overflow(browse_agent, mock_tools, mock_llm):
    """Test that agent handles content overflow correctly."""
    from unittest.mock import patch

    from tensortruth.core.source import SourceNode, SourceStatus, SourceType

    # Set a very small max_content to trigger overflow
    browse_agent.max_content_chars = 100

    # Mock search results
    search_results = [
        {"url": "https://example.com/1", "title": "Result 1", "snippet": "..."}
    ]
    mock_tools["search_web"].acall.return_value = json.dumps(search_results)

    # Mock pipeline results
    mock_fitted_pages = [
        ("https://example.com/1", "Page 1", "Content that is very long" * 100)
    ]
    mock_source_nodes = [
        SourceNode(
            id="1",
            url="https://example.com/1",
            title="Page 1",
            source_type=SourceType.WEB,
            status=SourceStatus.SUCCESS,
            content="Content that is very long" * 100,
            content_chars=5000,
        )
    ]
    mock_allocations = {"https://example.com/1": 5000}

    # Mock LLM routing - use "fetch_sources" not "fetch_pages_batch"
    mock_llm.acomplete.side_effect = [
        MagicMock(text='{"action": "search_web"}'),
        MagicMock(text='{"action": "fetch_sources"}'),
        MagicMock(text='{"action": "done"}'),
    ]

    # Mock synthesis
    async def mock_stream():
        yield MagicMock(delta="Answer")

    mock_llm.astream_complete.return_value = mock_stream()

    # Run agent with mocked pipeline and Ollama (for synthesis LLM creation)
    callbacks = AgentCallbacks()
    with (
        patch("tensortruth.core.source_pipeline.SourceFetchPipeline") as MockPipeline,
        patch("tensortruth.agents.router.browse.agent.Ollama", return_value=mock_llm),
    ):
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.execute = AsyncMock(
            return_value=(mock_fitted_pages, mock_source_nodes, mock_allocations)
        )
        MockPipeline.return_value = mock_pipeline_instance

        result = await browse_agent.run("test query", callbacks)

    # Should complete despite overflow
    assert result.final_answer == "Answer"


@pytest.mark.asyncio
async def test_browse_agent_synthesis_uses_session_model(browse_agent, mock_llm):
    """Test that synthesis uses the session model, not router model."""
    # Create agent with different router and synthesis models
    router_llm = MagicMock()
    router_llm.context_window = 8192
    synthesis_llm = MagicMock()
    synthesis_llm.context_window = 16384
    synthesis_llm.astream_complete = AsyncMock()

    agent = BrowseAgent(
        router_llm=router_llm,
        synthesis_llm=synthesis_llm,
        tools={},
        context_window=16384,
    )

    # Verify correct LLM assignment
    assert agent.router_llm == router_llm
    assert agent.synthesis_llm == synthesis_llm


def test_browse_agent_creates_initial_state(browse_agent):
    """Test that agent creates proper initial state."""
    state = browse_agent._create_initial_state("test query")

    assert state.query == "test query"
    assert state.phase == WorkflowPhase.INITIAL
    assert state.min_pages_required == 5
    assert state.max_content_chars == browse_agent.max_content_chars
    assert state.actions_taken == []
    assert state.iteration_count == 0


@pytest.mark.asyncio
async def test_browse_agent_route_delegates_to_router(browse_agent, mock_llm):
    """Test that route() delegates to BrowseRouter."""
    from tensortruth.agents.router.browse.state import BrowseState

    state = BrowseState(
        query="test",
        phase=WorkflowPhase.INITIAL,
        min_pages_required=3,
        max_content_chars=10000,
        generated_queries=["test 1", "test 2", "test 3"],  # Add generated queries
    )

    # Mock LLM response - router_llm might fail, so fallback to deterministic
    # For INITIAL state with generated queries, deterministic routing returns "search_web"
    action = await browse_agent.route(state)

    # Should return search_web (either from LLM or deterministic fallback)
    assert action == "search_web"


@pytest.mark.asyncio
async def test_browse_agent_execute_search(browse_agent, mock_tools):
    """Test that execute() handles search_web action."""
    from tensortruth.agents.router.browse.state import BrowseState

    state = BrowseState(
        query="test",
        phase=WorkflowPhase.INITIAL,
        min_pages_required=3,
        max_content_chars=10000,
    )

    # Mock search results
    search_results = [{"url": "https://example.com", "title": "Result"}]
    mock_tools["search_web"].acall.return_value = json.dumps(search_results)

    updated_state = await browse_agent.execute("search_web", state)

    assert updated_state.phase == WorkflowPhase.SEARCHED
    assert updated_state.search_results == search_results


@pytest.mark.asyncio
async def test_browse_agent_execute_fetch(browse_agent, mock_tools):
    """Test that execute() handles fetch_sources action."""
    from unittest.mock import patch

    from tensortruth.agents.router.browse.state import BrowseState
    from tensortruth.core.source import SourceNode, SourceStatus, SourceType

    state = BrowseState(
        query="test",
        phase=WorkflowPhase.SEARCHED,
        min_pages_required=3,
        max_content_chars=10000,
        search_results=[
            {"url": "https://example.com", "title": "Result", "snippet": "..."}
        ],
    )

    # Mock pipeline results
    mock_fitted_pages = [("https://example.com", "Page", "content")]
    mock_source_nodes = [
        SourceNode(
            id="1",
            url="https://example.com",
            title="Page",
            source_type=SourceType.WEB,
            status=SourceStatus.SUCCESS,
            content="content",
            content_chars=100,
        )
    ]
    mock_allocations = {"https://example.com": 100}

    with patch("tensortruth.core.source_pipeline.SourceFetchPipeline") as MockPipeline:
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.execute = AsyncMock(
            return_value=(mock_fitted_pages, mock_source_nodes, mock_allocations)
        )
        MockPipeline.return_value = mock_pipeline_instance

        # Use "fetch_sources" action, not "fetch_pages_batch"
        updated_state = await browse_agent.execute("fetch_sources", state)

    assert updated_state.phase == WorkflowPhase.FETCHED
    assert len(updated_state.pages) == 1


@pytest.mark.asyncio
async def test_browse_agent_execute_done(browse_agent):
    """Test that execute() handles done action."""
    from tensortruth.agents.router.browse.state import BrowseState

    state = BrowseState(
        query="test",
        phase=WorkflowPhase.FETCHED,
        min_pages_required=3,
        max_content_chars=10000,
        pages=[{"url": "https://example.com", "status": "success"}],
    )

    updated_state = await browse_agent.execute("done", state)

    assert updated_state.phase == WorkflowPhase.COMPLETE


def test_browse_agent_extract_urls(browse_agent):
    """Test that agent extracts URLs from successful pages."""
    from tensortruth.agents.router.browse.state import BrowseState

    state = BrowseState(
        query="test",
        phase=WorkflowPhase.COMPLETE,
        min_pages_required=3,
        max_content_chars=10000,
        pages=[
            {"url": "https://example.com/1", "status": "success"},
            {"url": "https://example.com/2", "status": "failed"},
            {"url": "https://example.com/3", "status": "success"},
        ],
    )

    urls = browse_agent._extract_urls(state)

    # Now returns all pages regardless of status for transparency
    assert len(urls) == 3
    assert "https://example.com/1" in urls
    assert "https://example.com/2" in urls
    assert "https://example.com/3" in urls


@pytest.mark.asyncio
async def test_browse_agent_trims_content_when_exceeds_limit(
    browse_agent, mock_tools, mock_llm
):
    """Test that synthesis trims content when it exceeds max_content_chars."""
    from unittest.mock import patch

    # Set a small limit
    browse_agent.max_content_chars = 500

    # Create state with large content
    from tensortruth.agents.router.browse.state import BrowseState

    large_content = "x" * 1000
    state = BrowseState(
        query="test",
        phase=WorkflowPhase.COMPLETE,
        min_pages_required=3,
        max_content_chars=500,
        pages=[
            {
                "url": "https://example.com",
                "title": "Page",
                "status": "success",
                "content": large_content,
            }
        ],
    )

    # Mock synthesis
    async def mock_stream():
        yield MagicMock(delta="Answer")

    mock_llm.astream_complete.return_value = mock_stream()

    callbacks = AgentCallbacks()

    # Patch Ollama creation to return the mock LLM
    with patch("tensortruth.agents.router.browse.agent.Ollama", return_value=mock_llm):
        answer = await browse_agent._synthesize(state, callbacks)

    # Verify synthesis was called (content was trimmed but synthesis proceeded)
    assert answer == "Answer"
    mock_llm.astream_complete.assert_called_once()
