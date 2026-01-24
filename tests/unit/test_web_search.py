"""Unit tests for tensortruth.utils.web_search module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from tensortruth.utils.web_search import (
    clean_html_for_content,
    fetch_page_as_markdown,
    fetch_pages_parallel,
    search_duckduckgo,
    summarize_with_llm,
    web_search,
    web_search_async,
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
            "title": "PyTorch 2.9 Features",
            "snippet": "New features in PyTorch 2.9 include...",
        },
        {
            "url": "https://example.com/page2",
            "title": "PyTorch Release Notes",
            "snippet": "Release notes for PyTorch 2.9...",
        },
        {
            "url": "https://example.com/page3",
            "title": "PyTorch Documentation",
            "snippet": "Complete documentation for PyTorch...",
        },
    ]


@pytest.fixture
def sample_html_clean():
    """Sample HTML with content."""
    return """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <main>
                <h1>Main Content</h1>
                <p>This is the main content of the page.</p>
                <code>print("hello")</code>
            </main>
        </body>
    </html>
    """


@pytest.fixture
def sample_html_noisy():
    """Sample HTML with lots of noise."""
    return """
    <html>
        <head>
            <title>Noisy Page</title>
            <script>alert("bad");</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <nav><a href="/home">Home</a></nav>
            <header>Site Header</header>
            <aside>Sidebar content</aside>
            <main>
                <h1>Real Content</h1>
                <p>This is the actual content.</p>
                <div class="advertisement">Buy now!</div>
                <div class="social">Share on social</div>
            </main>
            <footer>Copyright 2024</footer>
            <iframe src="ads.html"></iframe>
        </body>
    </html>
    """


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp ClientSession."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(
        return_value="<html><body><main><h1>Test</h1><p>Content</p></main></body></html>"
    )
    mock_session.get = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session


@pytest.fixture
def mock_ollama_llm():
    """Mock Ollama LLM."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """### Summary
This is a test summary of the web search results.

### Key Findings
- **Finding 1**: Important point about PyTorch
- **Finding 2**: Another key feature"""
    mock_llm.acomplete = AsyncMock(return_value=mock_response)
    return mock_llm


# ============================================================================
# Tests for search_duckduckgo
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestSearchDuckDuckGo:
    """Tests for DuckDuckGo search functionality."""

    async def test_successful_search(self, sample_search_results):
        """Test successful DuckDuckGo search."""
        with patch("tensortruth.utils.web_search.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = [
                {
                    "href": "https://example.com/page1",
                    "title": "PyTorch 2.9 Features",
                    "body": "New features in PyTorch 2.9 include...",
                },
                {
                    "href": "https://example.com/page2",
                    "title": "PyTorch Release Notes",
                    "body": "Release notes for PyTorch 2.9...",
                },
            ]
            mock_ddgs.return_value = mock_instance

            results = await search_duckduckgo("PyTorch 2.9", max_results=5)

            assert len(results) == 2
            assert results[0]["url"] == "https://example.com/page1"
            assert results[0]["title"] == "PyTorch 2.9 Features"
            assert results[0]["snippet"] == "New features in PyTorch 2.9 include..."
            mock_instance.text.assert_called_once_with(
                "PyTorch 2.9",
                region="us-en",
                safesearch="moderate",
                timelimit=None,
                max_results=5,
            )

    async def test_search_with_alternative_keys(self):
        """Test search with alternative key names (link instead of href)."""
        with patch("tensortruth.utils.web_search.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = [
                {
                    "link": "https://example.com/page1",  # 'link' instead of 'href'
                    "title": "Test Page",
                    "snippet": "Test snippet",  # 'snippet' instead of 'body'
                }
            ]
            mock_ddgs.return_value = mock_instance

            results = await search_duckduckgo("test query")

            assert len(results) == 1
            assert results[0]["url"] == "https://example.com/page1"
            assert results[0]["snippet"] == "Test snippet"

    async def test_search_empty_results(self):
        """Test search returning no results."""
        with patch("tensortruth.utils.web_search.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = []
            mock_ddgs.return_value = mock_instance

            results = await search_duckduckgo("nonexistent query")

            assert results == []

    async def test_search_with_retry_on_failure(self):
        """Test search retries on failure then succeeds."""
        with patch("tensortruth.utils.web_search.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            # First call fails, second succeeds
            mock_instance.text.side_effect = [
                Exception("Rate limited"),
                [{"href": "https://example.com", "title": "Test", "body": "Content"}],
            ]
            mock_ddgs.return_value = mock_instance

            results = await search_duckduckgo("test")

            assert len(results) == 1
            assert mock_instance.text.call_count == 2

    async def test_search_fails_after_retries(self):
        """Test search returns empty list after all retries fail."""
        with patch("tensortruth.utils.web_search.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.side_effect = Exception("Persistent error")
            mock_ddgs.return_value = mock_instance

            results = await search_duckduckgo("test")

            assert results == []
            assert mock_instance.text.call_count == 3  # 3 attempts


# ============================================================================
# Tests for clean_html_for_content
# ============================================================================


@pytest.mark.unit
class TestCleanHtmlForContent:
    """Tests for HTML cleaning functionality."""

    def test_removes_scripts_and_styles(self, sample_html_noisy):
        """Test removal of script and style tags."""
        soup = BeautifulSoup(sample_html_noisy, "html.parser")
        cleaned = clean_html_for_content(soup)

        assert cleaned.find("script") is None
        assert cleaned.find("style") is None

    def test_removes_navigation_elements(self, sample_html_noisy):
        """Test removal of nav, header, footer, aside."""
        soup = BeautifulSoup(sample_html_noisy, "html.parser")
        cleaned = clean_html_for_content(soup)

        assert cleaned.find("nav") is None
        assert cleaned.find("header") is None
        assert cleaned.find("footer") is None
        assert cleaned.find("aside") is None

    def test_removes_iframes(self, sample_html_noisy):
        """Test removal of iframe elements."""
        soup = BeautifulSoup(sample_html_noisy, "html.parser")
        cleaned = clean_html_for_content(soup)

        assert cleaned.find("iframe") is None

    def test_removes_advertisement_classes(self, sample_html_noisy):
        """Test removal of common advertisement classes."""
        soup = BeautifulSoup(sample_html_noisy, "html.parser")
        cleaned = clean_html_for_content(soup)

        # Check that divs with 'advertisement' class are removed
        assert cleaned.find("div", class_="advertisement") is None
        assert cleaned.find("div", class_="social") is None

    def test_preserves_main_content(self, sample_html_clean):
        """Test that main content is preserved."""
        soup = BeautifulSoup(sample_html_clean, "html.parser")
        cleaned = clean_html_for_content(soup)

        # Main tag should still exist
        assert cleaned.find("main") is not None
        # Content should be preserved
        assert "Main Content" in cleaned.get_text()
        assert "This is the main content" in cleaned.get_text()

    def test_removes_empty_paragraphs(self):
        """Test removal of empty paragraphs."""
        html = "<html><body><p></p><p>   </p><p>Real content</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        cleaned = clean_html_for_content(soup)

        # Should have only one paragraph with content
        paragraphs = cleaned.find_all("p")
        assert len(paragraphs) == 1
        assert "Real content" in paragraphs[0].get_text()


# ============================================================================
# Tests for fetch_page_as_markdown
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestFetchPageAsMarkdown:
    """Tests for web page fetching and markdown conversion."""

    async def test_successful_fetch(self):
        """Test successful page fetch and conversion."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        # Add enough content to pass the 100 char minimum
        mock_response.text = AsyncMock(return_value="""<html><body><main>
                <h1>Title</h1>
                <p>This is enough content to pass the minimum length requirement
                for the page fetcher. It needs to be at least 100 characters long.</p>
            </main></body></html>""")

        # Properly mock async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_cm)

        markdown, status, error_msg = await fetch_page_as_markdown(
            "https://example.com", mock_session, timeout=10
        )

        assert status == "success"
        assert error_msg is None
        assert markdown is not None
        assert "<!-- Source: https://example.com -->" in markdown
        assert "Title" in markdown

    async def test_http_error(self):
        """Test handling of HTTP errors."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_cm)

        markdown, status, error_msg = await fetch_page_as_markdown(
            "https://example.com/404", mock_session
        )

        assert markdown is None
        assert status == "http_error"
        assert "404" in error_msg

    async def test_timeout(self):
        """Test handling of timeout."""
        mock_session = AsyncMock()
        # Mock the context manager to raise TimeoutError
        mock_cm = AsyncMock()
        mock_cm.__aenter__.side_effect = asyncio.TimeoutError()
        mock_session.get = MagicMock(return_value=mock_cm)

        markdown, status, error_msg = await fetch_page_as_markdown(
            "https://example.com/slow", mock_session, timeout=1
        )

        assert markdown is None
        assert status == "timeout"
        assert error_msg == "Timeout"

    async def test_network_error(self):
        """Test handling of network errors."""
        mock_session = AsyncMock()
        # Mock the context manager to raise network error
        mock_cm = AsyncMock()
        mock_cm.__aenter__.side_effect = Exception("Connection refused")
        mock_session.get = MagicMock(return_value=mock_cm)

        markdown, status, error_msg = await fetch_page_as_markdown(
            "https://example.com", mock_session
        )

        assert markdown is None
        assert status == "network_error"
        assert "Connection refused" in error_msg

    async def test_content_too_short(self):
        """Test rejection of pages with very little content."""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value="<html><body><p>Hi</p></body></html>"
        )
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_cm)

        markdown, status, error_msg = await fetch_page_as_markdown(
            "https://example.com/short", mock_session
        )

        assert markdown is None
        assert status == "too_short"
        assert error_msg == "Content too short"

    async def test_extracts_main_content(self):
        """Test extraction of main content area."""
        html = """
        <html>
            <body>
                <nav>Navigation</nav>
                <main>
                    <h1>Article Title</h1>
                    <p>This is the main article content that should be extracted.</p>
                </main>
                <footer>Footer</footer>
            </body>
        </html>
        """
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=html)

        # Properly mock async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_cm)

        markdown, status, error_msg = await fetch_page_as_markdown(
            "https://example.com", mock_session
        )

        assert status == "success"
        assert error_msg is None
        assert markdown is not None
        assert "Article Title" in markdown
        assert "main article content" in markdown
        # Navigation and footer should be removed by cleaning
        # (they might appear in source comment but not in content)


# ============================================================================
# Tests for fetch_pages_parallel
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestFetchPagesParallel:
    """Tests for parallel page fetching."""

    async def test_fetch_multiple_pages(self, sample_search_results):
        """Test fetching multiple pages in parallel."""
        with patch("tensortruth.utils.web_search.fetch_page_as_markdown") as mock_fetch:
            mock_fetch.side_effect = [
                ("<!-- Source: url1 -->\n# Page 1 content", "success", None),
                ("<!-- Source: url2 -->\n# Page 2 content", "success", None),
                ("<!-- Source: url3 -->\n# Page 3 content", "success", None),
            ]

            successful, all_attempts = await fetch_pages_parallel(
                sample_search_results, max_pages=3
            )

            assert len(successful) == 3
            assert len(all_attempts) == 3
            assert successful[0][0] == "https://example.com/page1"
            assert successful[0][1] == "PyTorch 2.9 Features"
            assert "Page 1 content" in successful[0][2]
            # Check all attempts show success
            assert all(attempt[2] == "success" for attempt in all_attempts)

    async def test_handles_partial_failures(self, sample_search_results):
        """Test handling when some pages fail to fetch."""
        with patch("tensortruth.utils.web_search.fetch_page_as_markdown") as mock_fetch:
            # Second page fails
            mock_fetch.side_effect = [
                ("<!-- Source: url1 -->\n# Page 1 content", "success", None),
                (None, "http_error", "HTTP 403"),  # Failed
                ("<!-- Source: url3 -->\n# Page 3 content", "success", None),
            ]

            successful, all_attempts = await fetch_pages_parallel(
                sample_search_results, max_pages=3
            )

            assert len(successful) == 2  # Only successful fetches
            assert len(all_attempts) == 3  # All attempts tracked
            assert successful[0][1] == "PyTorch 2.9 Features"
            assert successful[1][1] == "PyTorch Documentation"
            # Check that failed attempt is tracked
            assert all_attempts[1][2] == "http_error"
            assert all_attempts[1][3] == "HTTP 403"

    async def test_respects_max_pages_limit(self):
        """Test that max_pages parameter is respected."""
        large_results = [
            {"url": f"https://example.com/page{i}", "title": f"Page {i}", "snippet": ""}
            for i in range(10)
        ]

        with patch("tensortruth.utils.web_search.fetch_page_as_markdown") as mock_fetch:
            mock_fetch.return_value = ("# Content", "success", None)

            successful, all_attempts = await fetch_pages_parallel(
                large_results, max_pages=3
            )

            # Should have exactly 3 successful pages
            assert len(successful) == 3
            assert len(all_attempts) == 3

    async def test_handles_all_failures(self, sample_search_results):
        """Test handling when all page fetches fail."""
        with patch("tensortruth.utils.web_search.fetch_page_as_markdown") as mock_fetch:
            mock_fetch.return_value = (None, "http_error", "HTTP 403")  # All fail

            successful, all_attempts = await fetch_pages_parallel(sample_search_results)

            assert successful == []
            assert len(all_attempts) > 0  # Attempts were tracked
            assert all(attempt[2] == "http_error" for attempt in all_attempts)


# ============================================================================
# Tests for summarize_with_llm
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestSummarizeWithLLM:
    """Tests for LLM summarization."""

    async def test_successful_summarization(self):
        """Test successful LLM summarization."""
        pages = [
            (
                "https://example.com",
                "Test Page",
                "# Test\n\nThis is test content for summarization.",
            )
        ]

        with patch("tensortruth.utils.web_search.Ollama") as mock_ollama_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "### Summary\nThis is a test summary."
            mock_llm.acomplete = AsyncMock(return_value=mock_response)
            mock_ollama_class.return_value = mock_llm

            result = await summarize_with_llm(
                "test query",
                pages,
                "deepseek-r1:8b",
                "http://localhost:11434",
            )

            assert "Summary" in result
            assert "test summary" in result
            mock_llm.acomplete.assert_called_once()

    async def test_handles_empty_pages(self):
        """Test handling of empty pages list."""
        result = await summarize_with_llm(
            "test query",
            [],
            "deepseek-r1:8b",
            "http://localhost:11434",
        )

        assert "No pages could be fetched" in result

    async def test_truncates_long_content(self):
        """Test that very long content is truncated."""
        # Create a very long page
        long_content = "# Content\n\n" + ("Lorem ipsum dolor sit amet. " * 1000)
        pages = [("https://example.com", "Long Page", long_content)]

        with patch("tensortruth.utils.web_search.Ollama") as mock_ollama_class:
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Summary"
            mock_llm.acomplete = AsyncMock(return_value=mock_response)
            mock_ollama_class.return_value = mock_llm

            await summarize_with_llm(
                "test",
                pages,
                "deepseek-r1:8b",
                "http://localhost:11434",
            )

            # Check that the prompt was truncated
            call_args = mock_llm.acomplete.call_args[0][0]
            assert len(call_args) < len(long_content) + 1000  # Should be truncated

    async def test_handles_llm_error(self):
        """Test handling of LLM errors."""
        pages = [("https://example.com", "Test", "# Content")]

        with patch("tensortruth.utils.web_search.Ollama") as mock_ollama_class:
            mock_llm = MagicMock()
            mock_llm.acomplete.side_effect = Exception("LLM timeout")
            mock_ollama_class.return_value = mock_llm

            result = await summarize_with_llm(
                "test",
                pages,
                "deepseek-r1:8b",
                "http://localhost:11434",
            )

            assert "Summarization failed" in result
            assert "LLM timeout" in result


# ============================================================================
# Tests for web_search_async (main pipeline)
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestWebSearchAsync:
    """Tests for complete web search pipeline."""

    async def test_complete_pipeline_success(self):
        """Test successful end-to-end web search."""
        with patch("tensortruth.utils.web_search.search_duckduckgo") as mock_search:
            with patch(
                "tensortruth.utils.web_search.fetch_pages_parallel"
            ) as mock_fetch:
                with patch(
                    "tensortruth.utils.web_search.summarize_with_llm"
                ) as mock_summarize:
                    # Setup mocks
                    mock_search.return_value = [
                        {
                            "url": "https://example.com",
                            "title": "Test",
                            "snippet": "...",
                        }
                    ]
                    # fetch_pages_parallel now returns (successful, all_attempts)
                    successful = [("https://example.com", "Test Page", "# Content")]
                    all_attempts = [
                        ("https://example.com", "Test Page", "success", None)
                    ]
                    mock_fetch.return_value = (successful, all_attempts)
                    mock_summarize.return_value = "### Summary\nTest summary"

                    result, sources = await web_search_async(
                        "test query",
                        "deepseek-r1:8b",
                        "http://localhost:11434",
                    )

                    # Successful results include summary text
                    assert "Test summary" in result
                    # Sources are returned separately
                    assert len(sources) == 1
                    assert sources[0].url == "https://example.com"
                    assert sources[0].status == "success"

    async def test_no_search_results(self):
        """Test handling when search returns no results."""
        with patch("tensortruth.utils.web_search.search_duckduckgo") as mock_search:
            mock_search.return_value = []

            result, sources = await web_search_async(
                "nonexistent query",
                "deepseek-r1:8b",
                "http://localhost:11434",
            )

            assert "No results found" in result
            assert "nonexistent query" in result
            assert sources == []

    async def test_search_results_but_no_pages_fetched(self):
        """Test fallback when search succeeds but all page fetches fail."""
        with patch("tensortruth.utils.web_search.search_duckduckgo") as mock_search:
            with patch(
                "tensortruth.utils.web_search.fetch_pages_parallel"
            ) as mock_fetch:
                mock_search.return_value = [
                    {
                        "url": "https://example.com",
                        "title": "Test Page",
                        "snippet": "Test snippet",
                    }
                ]
                # Return empty successful list and some failed attempts
                mock_fetch.return_value = ([], [])  # All fetches failed

                result, sources = await web_search_async(
                    "test query",
                    "deepseek-r1:8b",
                    "http://localhost:11434",
                )

                assert "couldn't fetch full pages" in result
                assert "Test snippet" in result  # Shows snippets as fallback
                # Sources list is empty when all fetches fail
                assert sources == []


# ============================================================================
# Tests for web_search (sync wrapper)
# ============================================================================


@pytest.mark.unit
class TestWebSearch:
    """Tests for synchronous web search wrapper."""

    def test_sync_wrapper_calls_async(self):
        """Test that sync wrapper properly calls async version."""
        with patch("tensortruth.utils.web_search.asyncio.run") as mock_run:
            mock_run.return_value = "Test result"

            result = web_search(
                "test query",
                "deepseek-r1:8b",
                "http://localhost:11434",
                max_results=5,
                max_pages=3,
            )

            assert result == "Test result"
            mock_run.assert_called_once()


# ============================================================================
# Tests for rerank_search_results
# ============================================================================


@pytest.mark.unit
class TestRerankSearchResults:
    """Tests for search result reranking by title+snippet."""

    def test_returns_all_results_sorted_by_score(self):
        """All results returned, sorted highest score first."""
        from tensortruth.utils.web_search import rerank_search_results

        results = [
            {
                "url": "https://example.com/1",
                "title": "Low relevance",
                "snippet": "Not related",
            },
            {
                "url": "https://example.com/2",
                "title": "PyTorch features",
                "snippet": "New PyTorch 2.0",
            },
            {
                "url": "https://example.com/3",
                "title": "Medium",
                "snippet": "Some ML content",
            },
        ]

        # Mock reranker
        mock_reranker = MagicMock()

        # Mock postprocess_nodes to return nodes with scores
        def mock_postprocess(nodes, query_bundle):
            # Simulate reranker assigning scores - return in different order
            scored_nodes = []
            for node in nodes:
                mock_node = MagicMock()
                mock_node.node = node.node
                # Assign scores based on content (simulate reranker behavior)
                if "PyTorch" in node.text:
                    mock_node.score = 0.95
                elif "ML" in node.text:
                    mock_node.score = 0.6
                else:
                    mock_node.score = 0.2
                scored_nodes.append(mock_node)
            return scored_nodes

        mock_reranker.postprocess_nodes = mock_postprocess

        ranked = rerank_search_results("PyTorch 2.0", results, 10, mock_reranker)

        # Should return all results
        assert len(ranked) == 3
        # Should be sorted by score (highest first)
        assert ranked[0][1] == 0.95  # PyTorch result
        assert ranked[1][1] == 0.6  # ML result
        assert ranked[2][1] == 0.2  # Low relevance
        # Check URLs are correctly associated
        assert ranked[0][0]["url"] == "https://example.com/2"

    def test_handles_empty_results(self):
        """Empty input returns empty output."""
        from tensortruth.utils.web_search import rerank_search_results

        mock_reranker = MagicMock()
        mock_reranker.postprocess_nodes = MagicMock(return_value=[])

        ranked = rerank_search_results("test query", [], 10, mock_reranker)

        assert ranked == []

    def test_creates_text_from_title_and_snippet(self):
        """TextNode text should be 'title\\nsnippet'."""
        from tensortruth.utils.web_search import rerank_search_results

        results = [
            {
                "url": "https://example.com",
                "title": "Test Title",
                "snippet": "Test snippet content",
            }
        ]

        mock_reranker = MagicMock()
        captured_nodes = []

        def capture_nodes(nodes, query_bundle):
            captured_nodes.extend(nodes)
            # Return with score
            for node in nodes:
                node.score = 0.5
            return nodes

        mock_reranker.postprocess_nodes = capture_nodes

        rerank_search_results("query", results, 10, mock_reranker)

        # Verify the text format
        assert len(captured_nodes) == 1
        assert captured_nodes[0].text == "Test Title\nTest snippet content"


# ============================================================================
# Tests for rerank_fetched_pages
# ============================================================================


@pytest.mark.unit
class TestRerankFetchedPages:
    """Tests for fetched page content reranking."""

    def test_includes_custom_instructions_in_query(self):
        """Custom instructions appended to query for ranking."""
        from tensortruth.utils.web_search import rerank_fetched_pages

        pages = [("https://example.com", "Test", "Content here")]

        mock_reranker = MagicMock()
        captured_query = []

        def capture_query(nodes, query_bundle):
            captured_query.append(query_bundle.query_str)
            for node in nodes:
                node.score = 0.5
            return nodes

        mock_reranker.postprocess_nodes = capture_query

        rerank_fetched_pages(
            "PyTorch features", "make an overview", pages, mock_reranker
        )

        assert len(captured_query) == 1
        assert "PyTorch features" in captured_query[0]
        assert "make an overview" in captured_query[0]

    def test_works_without_custom_instructions(self):
        """Functions correctly when custom_instructions is None."""
        from tensortruth.utils.web_search import rerank_fetched_pages

        pages = [("https://example.com", "Test", "Content here")]

        mock_reranker = MagicMock()
        captured_query = []

        def capture_query(nodes, query_bundle):
            captured_query.append(query_bundle.query_str)
            for node in nodes:
                node.score = 0.5
            return nodes

        mock_reranker.postprocess_nodes = capture_query

        rerank_fetched_pages("PyTorch features", None, pages, mock_reranker)

        assert len(captured_query) == 1
        assert captured_query[0] == "PyTorch features"  # No additional text

    def test_returns_pages_with_scores(self):
        """Each page has associated relevance score."""
        from tensortruth.utils.web_search import rerank_fetched_pages

        pages = [
            ("https://example.com/1", "Page 1", "Content 1"),
            ("https://example.com/2", "Page 2", "Content 2"),
        ]

        mock_reranker = MagicMock()

        def assign_scores(nodes, query_bundle):
            for i, node in enumerate(nodes):
                node.score = 0.9 - (i * 0.2)  # 0.9, 0.7
            return nodes

        mock_reranker.postprocess_nodes = assign_scores

        ranked = rerank_fetched_pages("query", None, pages, mock_reranker)

        assert len(ranked) == 2
        # Each result is (page_tuple, score)
        assert ranked[0][1] == 0.9
        assert ranked[1][1] == 0.7
        # Pages are tuples
        assert ranked[0][0] == ("https://example.com/1", "Page 1", "Content 1")

    def test_truncates_long_content(self):
        """Content truncated for efficiency in ranking."""
        from tensortruth.utils.web_search import rerank_fetched_pages

        # Create very long content
        long_content = "x" * 10000
        pages = [("https://example.com", "Long Page", long_content)]

        mock_reranker = MagicMock()
        captured_nodes = []

        def capture_nodes(nodes, query_bundle):
            captured_nodes.extend(nodes)
            for node in nodes:
                node.score = 0.5
            return nodes

        mock_reranker.postprocess_nodes = capture_nodes

        rerank_fetched_pages("query", None, pages, mock_reranker)

        # Content should be truncated (default ~2000 chars)
        assert len(captured_nodes) == 1
        assert len(captured_nodes[0].text) <= 2500  # Some margin for truncation


# ============================================================================
# Tests for web_search_stream with reranking
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestWebSearchStreamWithReranking:
    """Tests for web search streaming with reranking enabled."""

    async def test_yields_ranking_phase_when_reranker_enabled(self):
        """Should yield agent_progress with phase='ranking'."""
        from tensortruth.utils.web_search import web_search_stream

        with patch("tensortruth.utils.web_search.search_duckduckgo") as mock_search:
            with patch(
                "tensortruth.utils.web_search.get_reranker_for_web"
            ) as mock_get_reranker:
                with patch(
                    "tensortruth.utils.web_search.rerank_search_results"
                ) as mock_rerank_search:
                    with patch(
                        "tensortruth.utils.web_search.fetch_page_as_markdown"
                    ) as mock_fetch:
                        with patch(
                            "tensortruth.utils.web_search.rerank_fetched_pages"
                        ) as mock_rerank_pages:
                            with patch(
                                "tensortruth.utils.web_search.summarize_with_llm_stream"
                            ) as mock_summarize:
                                # Setup mocks
                                mock_search.return_value = [
                                    {
                                        "url": "https://example.com",
                                        "title": "Test",
                                        "snippet": "...",
                                    }
                                ]
                                mock_reranker = MagicMock()
                                mock_get_reranker.return_value = mock_reranker
                                mock_rerank_search.return_value = [
                                    (
                                        {
                                            "url": "https://example.com",
                                            "title": "Test",
                                            "snippet": "...",
                                        },
                                        0.9,
                                    )
                                ]
                                mock_fetch.return_value = ("# Content", "success", None)
                                mock_rerank_pages.return_value = [
                                    (("https://example.com", "Test", "# Content"), 0.85)
                                ]

                                async def mock_stream():
                                    yield "Summary"

                                mock_summarize.return_value = mock_stream()

                                phases_seen = []
                                async for chunk in web_search_stream(
                                    query="test",
                                    model_name="llama3.2",
                                    ollama_url="http://localhost:11434",
                                    reranker_model="BAAI/bge-reranker-v2-m3",
                                ):
                                    if (
                                        chunk.agent_progress
                                        and chunk.agent_progress.get("phase")
                                    ):
                                        phases_seen.append(
                                            chunk.agent_progress["phase"]
                                        )

                                assert "ranking" in phases_seen

    async def test_skips_ranking_when_reranker_disabled(self):
        """No ranking phase when reranker_model is None."""
        from tensortruth.utils.web_search import web_search_stream

        with patch("tensortruth.utils.web_search.search_duckduckgo") as mock_search:
            with patch(
                "tensortruth.utils.web_search.fetch_page_as_markdown"
            ) as mock_fetch:
                with patch(
                    "tensortruth.utils.web_search.summarize_with_llm_stream"
                ) as mock_summarize:
                    mock_search.return_value = [
                        {
                            "url": "https://example.com",
                            "title": "Test",
                            "snippet": "...",
                        }
                    ]
                    mock_fetch.return_value = ("# Content", "success", None)

                    async def mock_stream():
                        yield "Summary"

                    mock_summarize.return_value = mock_stream()

                    phases_seen = []
                    async for chunk in web_search_stream(
                        query="test",
                        model_name="llama3.2",
                        ollama_url="http://localhost:11434",
                        reranker_model=None,  # Disabled
                    ):
                        if chunk.agent_progress and chunk.agent_progress.get("phase"):
                            phases_seen.append(chunk.agent_progress["phase"])

                    # Should have searching, fetching, summarizing but NOT ranking
                    assert "ranking" not in phases_seen
                    assert "searching" in phases_seen

    async def test_sources_have_relevance_scores(self):
        """Final sources include relevance_score field."""
        from tensortruth.utils.web_search import web_search_stream

        with patch("tensortruth.utils.web_search.search_duckduckgo") as mock_search:
            with patch(
                "tensortruth.utils.web_search.get_reranker_for_web"
            ) as mock_get_reranker:
                with patch(
                    "tensortruth.utils.web_search.rerank_search_results"
                ) as mock_rerank_search:
                    with patch(
                        "tensortruth.utils.web_search.fetch_page_as_markdown"
                    ) as mock_fetch:
                        with patch(
                            "tensortruth.utils.web_search.rerank_fetched_pages"
                        ) as mock_rerank_pages:
                            with patch(
                                "tensortruth.utils.web_search.summarize_with_llm_stream"
                            ) as mock_summarize:
                                mock_search.return_value = [
                                    {
                                        "url": "https://example.com",
                                        "title": "Test",
                                        "snippet": "...",
                                    }
                                ]
                                mock_reranker = MagicMock()
                                mock_get_reranker.return_value = mock_reranker
                                mock_rerank_search.return_value = [
                                    (
                                        {
                                            "url": "https://example.com",
                                            "title": "Test",
                                            "snippet": "...",
                                        },
                                        0.9,
                                    )
                                ]
                                mock_fetch.return_value = ("# Content", "success", None)
                                mock_rerank_pages.return_value = [
                                    (("https://example.com", "Test", "# Content"), 0.85)
                                ]

                                async def mock_stream():
                                    yield "Summary"

                                mock_summarize.return_value = mock_stream()

                                final_sources = None
                                async for chunk in web_search_stream(
                                    query="test",
                                    model_name="llama3.2",
                                    ollama_url="http://localhost:11434",
                                    reranker_model="BAAI/bge-reranker-v2-m3",
                                ):
                                    if chunk.sources is not None:
                                        final_sources = chunk.sources

                                assert final_sources is not None
                                assert len(final_sources) == 1
                                assert final_sources[0].relevance_score == 0.85
