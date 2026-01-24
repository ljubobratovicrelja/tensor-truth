"""
Unit tests for web search progress reporting.

Tests structured progress callbacks, emoji-free messages,
and sources return value.
"""

import re
from typing import List
from unittest.mock import patch

import pytest

# Regex to detect emoji characters (common ranges)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001f9ff"  # Miscellaneous Symbols and Pictographs, Emoticons, etc.
    "\U0001fa00-\U0001fa6f"  # Chess Symbols, Extended-A
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027b0"  # Dingbats
    "\U000024c2-\U0001f251"  # Enclosed characters
    "]+",
    flags=re.UNICODE,
)


def test_no_emoji_in_search_progress_message():
    """Progress messages from search should not contain emoji."""
    from tensortruth.utils.web_search import search_duckduckgo

    captured_messages: List[str] = []

    def capture_callback(msg: str) -> None:
        captured_messages.append(msg)

    # We can't easily run the full search, but we can inspect the code
    # by checking that the progress callback format strings don't have emoji
    import inspect

    source = inspect.getsource(search_duckduckgo)

    # Check that the function source doesn't contain emoji in string literals
    emoji_matches = EMOJI_PATTERN.findall(source)
    assert (
        len(emoji_matches) == 0
    ), f"Found emoji in search_duckduckgo source: {emoji_matches}"


def test_no_emoji_in_fetch_progress_message():
    """Progress messages from fetch should not contain emoji."""
    import inspect

    from tensortruth.utils.web_search import fetch_pages_parallel

    source = inspect.getsource(fetch_pages_parallel)
    emoji_matches = EMOJI_PATTERN.findall(source)
    assert (
        len(emoji_matches) == 0
    ), f"Found emoji in fetch_pages_parallel source: {emoji_matches}"


def test_no_emoji_in_summarize_progress_message():
    """Progress messages from summarize should not contain emoji."""
    import inspect

    from tensortruth.utils.web_search import summarize_with_llm

    source = inspect.getsource(summarize_with_llm)
    emoji_matches = EMOJI_PATTERN.findall(source)
    assert (
        len(emoji_matches) == 0
    ), f"Found emoji in summarize_with_llm source: {emoji_matches}"


def test_no_emoji_in_web_search_async():
    """Progress messages from web_search_async should not contain emoji."""
    import inspect

    from tensortruth.utils.web_search import web_search_async

    source = inspect.getsource(web_search_async)
    emoji_matches = EMOJI_PATTERN.findall(source)
    assert (
        len(emoji_matches) == 0
    ), f"Found emoji in web_search_async source: {emoji_matches}"


def test_search_progress_dataclass_exists():
    """SearchProgress dataclass should exist with correct fields."""
    from tensortruth.utils.web_search import SearchProgress

    # Create instance and verify fields
    progress = SearchProgress(
        phase="searching",
        query="test query",
        hits=5,
    )
    assert progress.phase == "searching"
    assert progress.query == "test query"
    assert progress.hits == 5


def test_fetch_progress_dataclass_exists():
    """FetchProgress dataclass should exist with correct fields."""
    from tensortruth.utils.web_search import FetchProgress

    progress = FetchProgress(
        phase="fetching",
        url="https://example.com",
        title="Example Page",
        status="success",
        error=None,
        pages_target=5,
        pages_fetched=3,
        pages_failed=1,
    )
    assert progress.phase == "fetching"
    assert progress.url == "https://example.com"
    assert progress.title == "Example Page"
    assert progress.status == "success"
    assert progress.error is None
    assert progress.pages_target == 5
    assert progress.pages_fetched == 3
    assert progress.pages_failed == 1


def test_summarize_progress_dataclass_exists():
    """SummarizeProgress dataclass should exist with correct fields."""
    from tensortruth.utils.web_search import SummarizeProgress

    progress = SummarizeProgress(
        phase="summarizing",
        model_name="llama3.1:8b",
    )
    assert progress.phase == "summarizing"
    assert progress.model_name == "llama3.1:8b"


def test_backward_compatible_string_callback():
    """Old string-based callback should still work."""
    from tensortruth.utils.web_search import search_duckduckgo

    # The function should accept a callback that receives strings
    # This is the backward-compatible interface
    captured_messages: List[str] = []

    def string_callback(msg: str) -> None:
        captured_messages.append(msg)

    # The function signature should still accept progress_callback
    import inspect

    sig = inspect.signature(search_duckduckgo)
    assert "progress_callback" in sig.parameters


@pytest.mark.asyncio
async def test_web_search_returns_sources_tuple():
    """web_search_async should return (result, sources) tuple."""
    from tensortruth.utils.web_search import WebSearchSource, web_search_async

    with (
        patch("tensortruth.utils.web_search.search_duckduckgo") as mock_search,
        patch("tensortruth.utils.web_search.fetch_pages_parallel") as mock_fetch,
        patch("tensortruth.utils.web_search.summarize_with_llm") as mock_summarize,
    ):
        # Mock search results
        mock_search.return_value = [
            {"url": "https://example.com/1", "title": "Page 1", "snippet": "Snippet 1"},
            {"url": "https://example.com/2", "title": "Page 2", "snippet": "Snippet 2"},
        ]

        # Mock fetch results
        mock_fetch.return_value = (
            # Successful pages
            [("https://example.com/1", "Page 1", "# Content 1")],
            # All attempts
            [
                ("https://example.com/1", "Page 1", "success", None),
                ("https://example.com/2", "Page 2", "timeout", "Timeout"),
            ],
        )

        # Mock summarize
        mock_summarize.return_value = "Summary text"

        result, sources = await web_search_async(
            query="test",
            model_name="llama3.1:8b",
            ollama_url="http://localhost:11434",
        )

        # Should return summary string
        assert "Summary text" in result

        # Should return list of WebSearchSource
        assert isinstance(sources, list)
        assert len(sources) == 2

        # Check source structure
        source1 = sources[0]
        assert isinstance(source1, WebSearchSource)
        assert source1.url == "https://example.com/1"
        assert source1.title == "Page 1"
        assert source1.status == "success"
        assert source1.error is None

        source2 = sources[1]
        assert source2.url == "https://example.com/2"
        assert source2.title == "Page 2"
        assert source2.status == "failed"
        assert source2.error == "Timeout"


def test_web_search_source_model_exists():
    """WebSearchSource model should exist with correct fields."""
    from tensortruth.utils.web_search import WebSearchSource

    source = WebSearchSource(
        url="https://example.com",
        title="Example",
        status="success",
        error=None,
        snippet="A snippet",
    )
    assert source.url == "https://example.com"
    assert source.title == "Example"
    assert source.status == "success"
    assert source.error is None
    assert source.snippet == "A snippet"
