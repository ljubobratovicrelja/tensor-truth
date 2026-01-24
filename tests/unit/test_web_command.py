"""
Unit tests for WebSearchCommand in the API routes commands module.

Tests the new API-based web search command that uses web_search_async pipeline.
"""

from unittest.mock import AsyncMock, patch

import pytest


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

    with patch("tensortruth.api.routes.commands.web_search_async") as mock_search:
        mock_search.return_value = "Test summary"

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

    with patch("tensortruth.api.routes.commands.web_search_async") as mock_search:
        mock_search.return_value = "Test summary"

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

    with patch("tensortruth.api.routes.commands.web_search_async") as mock_search:
        mock_search.return_value = "Test summary"

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

    with patch("tensortruth.api.routes.commands.web_search_async") as mock_search:
        mock_search.return_value = "Test summary"

        await cmd.execute("just a query", {"params": {}}, websocket)

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["query"] == "just a query"
        assert call_kwargs["custom_instructions"] is None


@pytest.mark.asyncio
async def test_web_command_sends_done_message():
    """WebSearchCommand sends done message with result."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch("tensortruth.api.routes.commands.web_search_async") as mock_search:
        mock_search.return_value = "Final summary with citations"

        await cmd.execute("test query", {"params": {}}, websocket)

        # Find the done message
        calls = [c[0][0] for c in websocket.send_json.call_args_list]
        done_calls = [c for c in calls if c.get("type") == "done"]
        assert len(done_calls) == 1
        assert done_calls[0]["content"] == "Final summary with citations"


@pytest.mark.asyncio
async def test_web_command_handles_errors():
    """WebSearchCommand handles exceptions gracefully."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch("tensortruth.api.routes.commands.web_search_async") as mock_search:
        mock_search.side_effect = Exception("Network error")

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

    with patch("tensortruth.api.routes.commands.web_search_async") as mock_search:
        mock_search.return_value = "Summary"

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

    with patch("tensortruth.api.routes.commands.web_search_async") as mock_search:
        mock_search.return_value = "Summary"

        await cmd.execute("test", {"params": {}}, websocket)

        call_kwargs = mock_search.call_args[1]
        # Check defaults are applied
        assert call_kwargs["max_results"] == 10
        assert call_kwargs["max_pages"] == 5
        assert call_kwargs["context_window"] == 16384


@pytest.mark.asyncio
async def test_web_command_passes_progress_callback():
    """WebSearchCommand passes a progress callback to web_search_async."""
    from tensortruth.api.routes.commands import WebSearchCommand

    cmd = WebSearchCommand()
    websocket = AsyncMock()

    with patch("tensortruth.api.routes.commands.web_search_async") as mock_search:
        mock_search.return_value = "Summary"

        await cmd.execute("test", {"params": {}}, websocket)

        call_kwargs = mock_search.call_args[1]
        assert "progress_callback" in call_kwargs
        assert call_kwargs["progress_callback"] is not None


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
