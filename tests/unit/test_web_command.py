"""
Unit tests for WebSearchCommand in the API routes commands module.

Tests the streaming web search command that uses web_search_stream pipeline.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tensortruth.utils.web_search import WebSearchChunk, WebSearchSource


def make_mock_stream(tokens=None, sources=None):
    """Create a mock async generator that yields WebSearchChunk objects."""
    tokens = tokens if tokens is not None else ["Test ", "summary"]
    sources = sources if sources is not None else []

    async def mock_generator(**kwargs):
        # Yield search progress
        yield WebSearchChunk(
            agent_progress={
                "agent": "web_search",
                "phase": "searching",
                "message": "Searching",
            }
        )
        # Yield status
        yield WebSearchChunk(status="generating")
        # Yield tokens
        for token in tokens:
            yield WebSearchChunk(token=token)
        # Yield sources and complete
        yield WebSearchChunk(sources=sources, is_complete=True)

    return mock_generator


@pytest.mark.asyncio
async def test_web_command_requires_query():
    """WebSearchCommand should error if no query provided."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    await cmd.execute("", {}, websocket)

    websocket.send_json.assert_called()
    call_args = websocket.send_json.call_args[0][0]
    assert call_args["type"] == "error"
    assert "Usage" in call_args["detail"] or "query" in call_args["detail"].lower()


@pytest.mark.asyncio
async def test_web_command_requires_query_whitespace():
    """WebSearchCommand should error if only whitespace provided."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    await cmd.execute("   ", {}, websocket)

    websocket.send_json.assert_called()
    call_args = websocket.send_json.call_args[0][0]
    assert call_args["type"] == "error"


@pytest.mark.asyncio
async def test_web_command_parses_semicolon_separator():
    """WebSearchCommand parses query;instructions correctly."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(),
    ) as mock_search:
        await cmd.execute(
            "python async;focus on performance", {"params": {}}, websocket
        )

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["query"] == "python async"
        assert call_kwargs["custom_instructions"] == "focus on performance"


@pytest.mark.asyncio
async def test_web_command_parses_comma_separator():
    """WebSearchCommand parses query,instructions correctly."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(),
    ) as mock_search:
        await cmd.execute(
            "machine learning,explain like I'm 5", {"params": {}}, websocket
        )

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["query"] == "machine learning"
        assert call_kwargs["custom_instructions"] == "explain like I'm 5"


@pytest.mark.asyncio
async def test_web_command_semicolon_takes_precedence():
    """Semicolon separator takes precedence over comma."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(),
    ) as mock_search:
        # Query with both semicolon and comma - semicolon should be used
        await cmd.execute(
            "query with, comma;instructions here", {"params": {}}, websocket
        )

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["query"] == "query with, comma"
        assert call_kwargs["custom_instructions"] == "instructions here"


@pytest.mark.asyncio
async def test_web_command_no_separator():
    """WebSearchCommand handles query without instructions."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(),
    ) as mock_search:
        await cmd.execute("just a query", {"params": {}}, websocket)

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["query"] == "just a query"
        assert call_kwargs["custom_instructions"] is None


@pytest.mark.asyncio
async def test_web_command_sends_token_messages():
    """WebSearchCommand streams tokens to websocket."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(tokens=["Hello ", "world"]),
    ):
        await cmd.execute("test query", {"params": {}}, websocket)

        # Find token messages
        calls = [c[0][0] for c in websocket.send_json.call_args_list]
        token_calls = [c for c in calls if c.get("type") == "token"]
        assert len(token_calls) == 2
        assert token_calls[0]["content"] == "Hello "
        assert token_calls[1]["content"] == "world"


@pytest.mark.asyncio
async def test_web_command_sends_done_message():
    """WebSearchCommand sends done message with full response."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(tokens=["Final ", "summary"]),
    ):
        await cmd.execute("test query", {"params": {}}, websocket)

        # Find the done message
        calls = [c[0][0] for c in websocket.send_json.call_args_list]
        done_calls = [c for c in calls if c.get("type") == "done"]
        assert len(done_calls) == 1
        assert done_calls[0]["content"] == "Final summary"
        assert done_calls[0]["confidence_level"] == "web_search"


@pytest.mark.asyncio
async def test_web_command_handles_errors():
    """WebSearchCommand handles exceptions gracefully."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    async def error_generator(**kwargs):
        raise Exception("Network error")
        yield  # Make it a generator

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=error_generator,
    ):
        await cmd.execute("test query", {"params": {}}, websocket)

        # Should send error, not crash
        calls = [c[0][0] for c in websocket.send_json.call_args_list]
        error_calls = [c for c in calls if c.get("type") == "error"]
        assert len(error_calls) == 1
        assert "Network error" in error_calls[0]["detail"]


@pytest.mark.asyncio
async def test_web_command_uses_session_params():
    """WebSearchCommand extracts parameters from session."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    session = {
        "params": {
            "model": "custom-model:8b",
            "ollama_url": "http://custom:11434",
            "web_search_max_results": 15,
            "web_search_pages_to_fetch": 8,
            "context_window": 32768,
        }
    }

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(),
    ) as mock_search:
        await cmd.execute("test", session, websocket)

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["model_name"] == "custom-model:8b"
        assert call_kwargs["ollama_url"] == "http://custom:11434"
        assert call_kwargs["max_results"] == 15
        assert call_kwargs["max_pages"] == 8
        assert call_kwargs["context_window"] == 32768


@pytest.mark.asyncio
async def test_web_command_uses_defaults():
    """WebSearchCommand uses sensible defaults when params missing."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(),
    ) as mock_search:
        await cmd.execute("test", {"params": {}}, websocket)

        call_kwargs = mock_search.call_args[1]
        # Check defaults are applied
        assert call_kwargs["max_results"] == 10
        assert call_kwargs["max_pages"] == 5
        assert call_kwargs["context_window"] == 16384


@pytest.mark.asyncio
async def test_web_command_registered_in_registry():
    """WebSearchCommand is registered with correct name and aliases."""
    from tensortruth.api.routes.commands import registry

    # Check primary name
    web_cmd = registry.get("web")
    assert web_cmd is not None
    assert web_cmd.name == "web"

    # Check aliases
    search_cmd = registry.get("search")
    assert search_cmd is not None
    assert search_cmd is web_cmd  # Same instance

    websearch_cmd = registry.get("websearch")
    assert websearch_cmd is not None
    assert websearch_cmd is web_cmd  # Same instance


@pytest.mark.asyncio
async def test_web_command_sends_sources_as_source_nodes():
    """WebSearchCommand converts sources to SourceNode format for unified UI."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    # Create mock sources
    mock_sources = [
        WebSearchSource(
            url="https://example.com/1",
            title="Page 1",
            status="success",
            error=None,
            snippet="Snippet 1",
        ),
        WebSearchSource(
            url="https://example.com/2",
            title="Page 2",
            status="failed",
            error="Timeout",
            snippet="Snippet 2",
        ),
    ]

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(sources=mock_sources),
    ):
        await cmd.execute("test query", {"params": {}}, websocket)

        # Find the sources message (now using unified "sources" type)
        calls = [c[0][0] for c in websocket.send_json.call_args_list]
        sources_calls = [c for c in calls if c.get("type") == "sources"]
        assert len(sources_calls) == 1

        # Verify SourceNode format
        source_nodes = sources_calls[0]["data"]
        assert len(source_nodes) == 2

        # First source (success)
        assert source_nodes[0]["metadata"]["source_url"] == "https://example.com/1"
        assert source_nodes[0]["metadata"]["display_name"] == "Page 1"
        assert source_nodes[0]["metadata"]["doc_type"] == "web"
        assert source_nodes[0]["metadata"]["fetch_status"] == "success"
        assert source_nodes[0]["score"] == 1.0

        # Second source (failed)
        assert source_nodes[1]["metadata"]["fetch_status"] == "failed"
        assert source_nodes[1]["metadata"]["fetch_error"] == "Timeout"
        assert source_nodes[1]["score"] == 0.0


@pytest.mark.asyncio
async def test_web_command_no_sources_message_when_empty():
    """WebSearchCommand doesn't send sources when sources list is empty."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(sources=[]),
    ):
        await cmd.execute("test query", {"params": {}}, websocket)

        # Should not have sources message
        calls = [c[0][0] for c in websocket.send_json.call_args_list]
        sources_calls = [c for c in calls if c.get("type") == "sources"]
        assert len(sources_calls) == 0


@pytest.mark.asyncio
async def test_web_command_sends_agent_progress():
    """WebSearchCommand sends agent_progress messages for search phases."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(),
    ):
        await cmd.execute("test query", {"params": {}}, websocket)

        # Find agent_progress messages
        calls = [c[0][0] for c in websocket.send_json.call_args_list]
        progress_calls = [c for c in calls if c.get("type") == "agent_progress"]
        assert len(progress_calls) >= 1
        assert progress_calls[0]["agent"] == "web_search"
        assert progress_calls[0]["phase"] == "searching"


@pytest.mark.asyncio
async def test_web_command_sends_status_generating():
    """WebSearchCommand sends status 'generating' before tokens."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(),
    ):
        await cmd.execute("test query", {"params": {}}, websocket)

        # Find status messages
        calls = [c[0][0] for c in websocket.send_json.call_args_list]
        status_calls = [c for c in calls if c.get("type") == "status"]
        assert len(status_calls) >= 1
        assert status_calls[0]["status"] == "generating"


@pytest.mark.asyncio
async def test_web_command_title_pending_for_first_message():
    """WebSearchCommand sets title_pending=True for first message in session."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    # Session with only one message (user's message)
    session = {"params": {}, "messages": [{"role": "user", "content": "test"}]}

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(),
    ):
        await cmd.execute("test query", session, websocket)

        # Find done message
        calls = [c[0][0] for c in websocket.send_json.call_args_list]
        done_calls = [c for c in calls if c.get("type") == "done"]
        assert len(done_calls) == 1
        assert done_calls[0].get("title_pending") is True


@pytest.mark.asyncio
async def test_web_command_no_title_pending_for_subsequent_messages():
    """WebSearchCommand sets title_pending=False for subsequent messages."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    # Session with multiple messages
    session = {
        "params": {},
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "second"},
        ],
    }

    with patch(
        "tensortruth.api.routes.commands.web_search_stream",
        side_effect=make_mock_stream(),
    ):
        await cmd.execute("test query", session, websocket)

        # Find done message
        calls = [c[0][0] for c in websocket.send_json.call_args_list]
        done_calls = [c for c in calls if c.get("type") == "done"]
        assert len(done_calls) == 1
        assert done_calls[0].get("title_pending") is False
