"""Unit tests for tensortruth.services.builtin_tools module."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from tensortruth.services.builtin_tools import (
    fetch_page,
    fetch_pages_batch,
    search_focused,
    search_web,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_search_results():
    """Sample DuckDuckGo search results."""
    return [
        {
            "url": "https://example.com/page1",
            "title": "Python Async Programming Guide",
            "snippet": "Learn async programming in Python...",
        },
        {
            "url": "https://example.com/page2",
            "title": "Async Best Practices",
            "snippet": "Best practices for async code...",
        },
    ]


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content from fetch."""
    return "# Python Guide\n\nThis is a comprehensive guide to Python programming..."


# ============================================================================
# Tests for search_web
# ============================================================================


@pytest.mark.asyncio
async def test_search_web_success(sample_search_results):
    """Test successful web search with single query string."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=sample_search_results,
    ) as mock_search:
        result = await search_web("Python async")

        # Verify search was called
        mock_search.assert_called_once_with("Python async", max_results=5)

        # Verify result is valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["url"] == "https://example.com/page1"
        # Single query should add 'query' field to results
        assert parsed[0]["query"] == "Python async"


@pytest.mark.asyncio
async def test_search_web_max_results_per_query():
    """Test search_web respects max_results_per_query parameter."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_search:
        await search_web("test query", max_results_per_query=10)

        # Verify max_results_per_query was passed through
        mock_search.assert_called_once_with("test query", max_results=10)


@pytest.mark.asyncio
async def test_search_web_empty_results():
    """Test search_web with empty results."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await search_web("nonexistent query")

        # Verify result is valid JSON with empty list
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 0


@pytest.mark.asyncio
async def test_search_web_exception():
    """Test search_web handles exceptions gracefully."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        side_effect=Exception("Network error"),
    ):
        result = await search_web("test query")

        # Should not raise - returns error as JSON
        parsed = json.loads(result)
        assert "error" in parsed
        assert "failed" in parsed["error"].lower()  # Generic failure message
        assert "results" in parsed
        assert parsed["results"] == []


@pytest.mark.asyncio
async def test_search_web_json_format(sample_search_results):
    """Test search_web returns properly formatted JSON."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=sample_search_results,
    ):
        result = await search_web("test")

        # Verify it's valid JSON
        parsed = json.loads(result)

        # Verify all required fields present
        for item in parsed:
            assert "url" in item
            assert "title" in item
            assert "snippet" in item


@pytest.mark.asyncio
async def test_search_web_returns_string():
    """Test search_web always returns a string."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=[{"url": "test", "title": "test", "snippet": "test"}],
    ):
        result = await search_web("test")

        # Must return string, not dict
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_search_web_multiple_queries():
    """Test search_web with multiple queries executes in parallel."""
    query_results = {
        "Python async": [
            {
                "url": "https://example.com/async",
                "title": "Async Guide",
                "snippet": "Async programming...",
            }
        ],
        "asyncio tutorial": [
            {
                "url": "https://example.com/tutorial",
                "title": "Tutorial",
                "snippet": "Learn asyncio...",
            }
        ],
        "async best practices": [
            {
                "url": "https://example.com/practices",
                "title": "Best Practices",
                "snippet": "Best practices...",
            }
        ],
    }

    async def mock_search(query, max_results):
        return query_results.get(query, [])

    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        side_effect=mock_search,
    ) as mock_search_fn:
        result = await search_web(
            queries=["Python async", "asyncio tutorial", "async best practices"],
            max_results_per_query=5,
        )

        # Verify all queries were searched
        assert mock_search_fn.call_count == 3

        # Verify valid JSON returned
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 3  # One result from each query

        # Verify each result has the query field
        assert parsed[0]["query"] == "Python async"
        assert parsed[1]["query"] == "asyncio tutorial"
        assert parsed[2]["query"] == "async best practices"


@pytest.mark.asyncio
async def test_search_web_deduplicates_urls():
    """Test search_web deduplicates results by URL."""
    duplicate_results = [
        {
            "url": "https://example.com/page",
            "title": "Page",
            "snippet": "Content...",
        }
    ]

    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=duplicate_results,
    ):
        result = await search_web(queries=["query1", "query2"])

        parsed = json.loads(result)
        # Should only appear once despite 2 queries returning the same URL
        assert len(parsed) == 1
        assert parsed[0]["url"] == "https://example.com/page"


@pytest.mark.asyncio
async def test_search_web_handles_partial_query_failure():
    """Test search_web continues when some queries fail."""

    async def mock_search(query, max_results):
        if "fail" in query:
            raise Exception("Query failed")
        return [
            {
                "url": f"https://example.com/{query.replace(' ', '_')}",
                "title": query,
                "snippet": "...",
            }
        ]

    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        side_effect=mock_search,
    ):
        result = await search_web(queries=["query1", "fail query", "query2"])

        parsed = json.loads(result)
        # Should have results from successful queries only
        assert len(parsed) == 2
        assert parsed[0]["query"] == "query1"
        assert parsed[1]["query"] == "query2"


@pytest.mark.asyncio
async def test_search_web_single_query_backward_compatible():
    """Test search_web with single string query (backward compatibility)."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=[
            {
                "url": "https://example.com/test",
                "title": "Test",
                "snippet": "Test...",
            }
        ],
    ) as mock_search:
        # Pass single string (not list)
        result = await search_web(queries="single query")

        # Should still work
        mock_search.assert_called_once_with("single query", max_results=5)

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["query"] == "single query"


# ============================================================================
# Tests for fetch_page
# ============================================================================


@pytest.mark.asyncio
async def test_fetch_page_success(sample_markdown_content):
    """Test successful page fetch."""
    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        return_value=(sample_markdown_content, "success", None),
    ) as mock_fetch:
        result = await fetch_page("https://example.com")

        # Verify fetch was called
        assert mock_fetch.called

        # Verify result is markdown content
        assert result == sample_markdown_content
        assert result.startswith("# Python Guide")


@pytest.mark.asyncio
async def test_fetch_page_timeout():
    """Test fetch_page handles timeout."""
    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        return_value=(None, "timeout", "Request timed out"),
    ):
        result = await fetch_page("https://example.com", timeout=5)

        # Should return error message, not raise
        assert result.startswith("Error:")
        assert "timeout" in result.lower() or "Request timed out" in result


@pytest.mark.asyncio
async def test_fetch_page_http_error():
    """Test fetch_page handles HTTP errors."""
    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        return_value=(None, "http_error", "404 Not Found"),
    ):
        result = await fetch_page("https://example.com/missing")

        # Should return error message
        assert result.startswith("Error:")
        assert "404" in result or "Not Found" in result


@pytest.mark.asyncio
async def test_fetch_page_exception():
    """Test fetch_page handles exceptions gracefully."""
    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        side_effect=Exception("Connection failed"),
    ):
        result = await fetch_page("https://example.com")

        # Should not raise - returns error string
        assert result.startswith("Error:")
        assert "Connection failed" in result


@pytest.mark.asyncio
async def test_fetch_page_markdown_format():
    """Test fetch_page returns markdown format."""
    markdown = "# Title\n\n## Section\n\nParagraph with **bold** and *italic*."
    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        return_value=(markdown, "success", None),
    ):
        result = await fetch_page("https://example.com")

        # Verify markdown formatting preserved
        assert "# Title" in result
        assert "## Section" in result
        assert "**bold**" in result


@pytest.mark.asyncio
async def test_fetch_page_special_domain_wikipedia():
    """Test fetch_page handles special domains (Wikipedia)."""
    wikipedia_content = "# Python (programming language)\n\nPython is a..."
    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        return_value=(wikipedia_content, "success", None),
    ):
        result = await fetch_page("https://en.wikipedia.org/wiki/Python")

        # Verify Wikipedia content fetched
        assert "Python" in result
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_fetch_page_returns_string():
    """Test fetch_page always returns a string."""
    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        return_value=("content", "success", None),
    ):
        result = await fetch_page("https://example.com")

        # Must return string
        assert isinstance(result, str)


# ============================================================================
# Tests for fetch_pages_batch
# ============================================================================


@pytest.mark.asyncio
async def test_fetch_pages_batch_success():
    """Test successful batch page fetching."""

    async def mock_fetch(url, session, timeout):
        # Simulate different content for each URL
        if "page1" in url:
            return "# Page 1\n\nContent from page 1", "success", None
        elif "page2" in url:
            return "# Page 2\n\nContent from page 2", "success", None
        else:
            return "# Page 3\n\nContent from page 3", "success", None

    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        side_effect=mock_fetch,
    ):
        result = await fetch_pages_batch(
            [
                "https://example.com/page1",
                "https://example.com/page2",
                "https://example.com/page3",
            ]
        )

        # Verify result is valid JSON with new format
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "pages" in parsed
        assert "overflow" in parsed
        assert "total_chars" in parsed

        pages = parsed["pages"]
        assert isinstance(pages, list)
        assert len(pages) == 3

        # Verify all successful
        assert all(p["status"] == "success" for p in pages)
        assert all(p["content"] is not None for p in pages)
        assert all(p["error"] is None for p in pages)

        # Verify URLs are preserved
        assert pages[0]["url"] == "https://example.com/page1"
        assert pages[1]["url"] == "https://example.com/page2"
        assert pages[2]["url"] == "https://example.com/page3"

        # Verify overflow tracking
        assert parsed["overflow"] is False
        assert parsed["total_chars"] > 0


@pytest.mark.asyncio
async def test_fetch_pages_batch_partial_failure():
    """Test batch fetching handles partial failures gracefully."""

    async def mock_fetch(url, session, timeout):
        if "fail" in url:
            return None, "http_error", "404 Not Found"
        else:
            return f"# Success\n\nContent from {url}", "success", None

    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        side_effect=mock_fetch,
    ):
        result = await fetch_pages_batch(
            [
                "https://example.com/page1",
                "https://example.com/fail",
                "https://example.com/page2",
            ]
        )

        parsed = json.loads(result)
        pages = parsed["pages"]
        assert len(pages) == 3

        # First and third should succeed
        assert pages[0]["status"] == "success"
        assert pages[0]["content"] is not None
        assert pages[2]["status"] == "success"
        assert pages[2]["content"] is not None

        # Middle should fail
        assert pages[1]["status"] == "failed"
        assert pages[1]["content"] is None
        assert "404" in pages[1]["error"]


@pytest.mark.asyncio
async def test_fetch_pages_batch_empty_urls():
    """Test batch fetching with empty URL list."""
    result = await fetch_pages_batch([])

    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    assert "pages" in parsed
    assert isinstance(parsed["pages"], list)
    assert len(parsed["pages"]) == 0
    assert parsed["overflow"] is False
    assert parsed["total_chars"] == 0


@pytest.mark.asyncio
async def test_fetch_pages_batch_exception_handling():
    """Test batch fetching handles exceptions gracefully."""

    async def mock_fetch(url, session, timeout):
        if "exception" in url:
            raise Exception("Network error")
        else:
            return "# Content", "success", None

    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        side_effect=mock_fetch,
    ):
        result = await fetch_pages_batch(
            ["https://example.com/page1", "https://example.com/exception"]
        )

        parsed = json.loads(result)
        pages = parsed["pages"]
        assert len(pages) == 2

        # First should succeed
        assert pages[0]["status"] == "success"

        # Second should have exception as error
        assert pages[1]["status"] == "failed"
        assert "Network error" in pages[1]["error"]


@pytest.mark.asyncio
async def test_fetch_pages_batch_returns_json():
    """Test batch fetching always returns valid JSON."""
    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        return_value=("content", "success", None),
    ):
        result = await fetch_pages_batch(["https://example.com"])

        # Must be valid JSON
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "pages" in parsed
        assert isinstance(parsed["pages"], list)


@pytest.mark.asyncio
async def test_fetch_pages_batch_skipped_status():
    """Test batch fetching maps parse_error and too_short to skipped."""

    async def mock_fetch(url, session, timeout):
        if "short" in url:
            return None, "too_short", "Content too short"
        elif "parse" in url:
            return None, "parse_error", "Parse failed"
        else:
            return "# Content", "success", None

    with patch(
        "tensortruth.services.builtin_tools.fetch_page_as_markdown",
        new_callable=AsyncMock,
        side_effect=mock_fetch,
    ):
        result = await fetch_pages_batch(
            [
                "https://example.com/short",
                "https://example.com/parse",
                "https://example.com/good",
            ]
        )

        parsed = json.loads(result)
        pages = parsed["pages"]

        # Both short and parse should be mapped to "skipped"
        assert pages[0]["status"] == "skipped"
        assert pages[1]["status"] == "skipped"
        assert pages[2]["status"] == "success"


# ============================================================================
# Tests for search_focused
# ============================================================================


@pytest.mark.asyncio
async def test_search_focused_success(sample_search_results):
    """Test successful focused search."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=sample_search_results,
    ) as mock_search:
        result = await search_focused("Python async", "stackoverflow.com")

        # Verify search was called with site: query
        mock_search.assert_called_once_with(
            "site:stackoverflow.com Python async", max_results=5
        )

        # Verify result is valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, list)


@pytest.mark.asyncio
async def test_search_focused_query_building():
    """Test search_focused builds correct site query."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_search:
        await search_focused("test query", "github.com", max_results=3)

        # Verify site: prefix added correctly
        call_args = mock_search.call_args[0][0]
        assert call_args == "site:github.com test query"


@pytest.mark.asyncio
async def test_search_focused_different_domains():
    """Test search_focused with various domains."""
    domains = ["stackoverflow.com", "github.com", "docs.python.org"]

    for domain in domains:
        with patch(
            "tensortruth.services.builtin_tools.search_duckduckgo",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_search:
            await search_focused("test", domain)

            # Verify correct domain in query
            call_args = mock_search.call_args[0][0]
            assert f"site:{domain}" in call_args


@pytest.mark.asyncio
async def test_search_focused_exception():
    """Test search_focused handles exceptions gracefully."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        side_effect=Exception("Search failed"),
    ):
        result = await search_focused("test", "example.com")

        # Should not raise - returns error as JSON
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Search failed" in parsed["error"]
        assert parsed["results"] == []


@pytest.mark.asyncio
async def test_search_focused_json_format(sample_search_results):
    """Test search_focused returns properly formatted JSON."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=sample_search_results,
    ):
        result = await search_focused("test", "example.com")

        # Verify it's valid JSON
        parsed = json.loads(result)

        # Verify structure
        for item in parsed:
            assert "url" in item
            assert "title" in item
            assert "snippet" in item


@pytest.mark.asyncio
async def test_search_focused_returns_string():
    """Test search_focused always returns a string."""
    with patch(
        "tensortruth.services.builtin_tools.search_duckduckgo",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await search_focused("test", "example.com")

        # Must return string, not dict
        assert isinstance(result, str)


# ============================================================================
# Tests for general tool behavior
# ============================================================================


@pytest.mark.asyncio
async def test_all_tools_return_strings():
    """Test that all tools return strings, never raise."""
    with (
        patch(
            "tensortruth.services.builtin_tools.search_duckduckgo",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "tensortruth.services.builtin_tools.fetch_page_as_markdown",
            new_callable=AsyncMock,
            return_value=("content", "success", None),
        ),
    ):
        # All should return strings
        result1 = await search_web("test")
        result2 = await fetch_page("https://example.com")
        result3 = await search_focused("test", "example.com")

        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, str)


@pytest.mark.asyncio
async def test_tools_handle_empty_inputs():
    """Test tools handle empty/None inputs gracefully."""
    with (
        patch(
            "tensortruth.services.builtin_tools.search_duckduckgo",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "tensortruth.services.builtin_tools.fetch_page_as_markdown",
            new_callable=AsyncMock,
            return_value=(None, "parse_error", "Empty content"),
        ),
    ):
        # Empty query should still work (or return error as string)
        result1 = await search_web("")
        result2 = await fetch_page("")
        result3 = await search_focused("", "example.com")

        # All should return strings, not raise
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, str)
