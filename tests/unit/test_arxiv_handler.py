"""Tests for ArXiv handler."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tensortruth.utils.arxiv_handler import ArxivHandler


class TestArxivHandler:
    """Test suite for ArxivHandler."""

    @pytest.fixture
    def handler(self):
        """Create an ArxivHandler instance."""
        return ArxivHandler()

    def test_name(self, handler):
        """Test handler name property."""
        assert handler.name == "ArXiv"

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://arxiv.org/abs/1706.03762", True),
            ("https://arxiv.org/abs/2301.12345", True),
            ("https://arxiv.org/pdf/1706.03762.pdf", True),
            ("https://arxiv.org/abs/1706.03762v2", True),  # With version
            ("https://arxiv.org/pdf/2301.12345v1.pdf", True),
            ("http://arxiv.org/abs/1234.5678", True),
            ("https://export.arxiv.org/abs/1706.03762", True),  # Alternative domain
            # Should NOT match these:
            ("https://arxiv.org/list/cs.AI/recent", False),  # Listing page
            ("https://arxiv.org/search/", False),  # Search page
            ("https://arxiv.org", False),  # Homepage
            ("https://example.com/abs/1706.03762", False),  # Not arxiv.org
            ("https://github.com/repo", False),
            ("not-a-url", False),
        ],
    )
    def test_matches(self, handler, url, expected):
        """Test URL matching for ArXiv papers."""
        assert handler.matches(url) == expected

    @pytest.mark.parametrize(
        "url,expected_id",
        [
            ("https://arxiv.org/abs/1706.03762", "1706.03762"),
            ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
            ("https://arxiv.org/abs/1706.03762v2", "1706.03762"),  # Version removed
            ("https://arxiv.org/abs/2301.12345v1", "2301.12345"),
            ("https://arxiv.org/pdf/2301.12345.pdf", "2301.12345"),
        ],
    )
    def test_extract_arxiv_id(self, handler, url, expected_id):
        """Test ArXiv ID extraction from URLs."""
        assert handler._extract_arxiv_id(url) == expected_id

    def test_extract_arxiv_id_invalid(self, handler):
        """Test ArXiv ID extraction with invalid URLs."""
        assert handler._extract_arxiv_id("https://arxiv.org/list/cs.AI") is None
        assert handler._extract_arxiv_id("https://example.com") is None

    @pytest.mark.asyncio
    @patch("tensortruth.utils.arxiv_handler.arxiv.Search")
    async def test_fetch_success(self, mock_search_class, handler):
        """Test successful ArXiv paper fetch."""
        # Mock ArXiv paper
        mock_paper = Mock()
        mock_paper.title = "Attention Is All You Need"
        mock_paper.entry_id = "http://arxiv.org/abs/1706.03762v7"
        mock_paper.pdf_url = "https://arxiv.org/pdf/1706.03762v7"
        mock_paper.summary = (
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks."
        )
        mock_paper.published = datetime(2017, 6, 12, 17, 57, 34, tzinfo=timezone.utc)
        mock_paper.updated = datetime(2023, 8, 2, 8, 12, 45, tzinfo=timezone.utc)
        mock_paper.categories = ["cs.CL", "cs.AI"]
        mock_paper.primary_category = "cs.CL"
        mock_paper.doi = "10.5555/3295222.3295349"
        mock_paper.journal_ref = "NeurIPS 2017"
        mock_paper.comment = "15 pages, 5 figures"

        # Mock authors
        mock_author1 = Mock()
        mock_author1.name = "Ashish Vaswani"
        mock_author2 = Mock()
        mock_author2.name = "Noam Shazeer"
        mock_paper.authors = [mock_author1, mock_author2]

        # Mock search results
        mock_search = Mock()
        mock_search.results.return_value = iter([mock_paper])
        mock_search_class.return_value = mock_search

        # Mock session (not used but required by interface)
        mock_session = AsyncMock()

        url = "https://arxiv.org/abs/1706.03762"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert error is None
        assert markdown is not None
        assert "# Attention Is All You Need" in markdown
        assert "**ArXiv ID**: 1706.03762" in markdown
        assert "**Authors**: Ashish Vaswani, Noam Shazeer" in markdown
        assert "**Published**: 2017-06-12" in markdown
        assert "**Updated**: 2023-08-02" in markdown
        assert "**Categories**: cs.CL, cs.AI" in markdown
        assert "**Primary Category**: cs.CL" in markdown
        assert "**DOI**: 10.5555/3295222.3295349" in markdown
        assert "**Journal Reference**: NeurIPS 2017" in markdown
        assert "## Abstract" in markdown
        assert "dominant sequence transduction models" in markdown
        assert "## Comments" in markdown
        assert "15 pages, 5 figures" in markdown
        assert "https://arxiv.org/abs/1706.03762" in markdown

    @pytest.mark.asyncio
    @patch("tensortruth.utils.arxiv_handler.arxiv.Search")
    async def test_fetch_minimal_metadata(self, mock_search_class, handler):
        """Test fetch with minimal metadata (no DOI, journal, comment)."""
        mock_paper = Mock()
        mock_paper.title = "Test Paper"
        mock_paper.entry_id = "http://arxiv.org/abs/2301.12345v1"
        mock_paper.pdf_url = "https://arxiv.org/pdf/2301.12345v1"
        mock_paper.summary = "This is a test abstract. " * 10  # Make it long enough
        mock_paper.published = datetime(2023, 1, 29, tzinfo=timezone.utc)
        mock_paper.updated = datetime(2023, 1, 29, tzinfo=timezone.utc)
        mock_paper.categories = ["cs.AI"]
        mock_paper.primary_category = "cs.AI"
        mock_paper.doi = None
        mock_paper.journal_ref = None
        mock_paper.comment = None

        mock_author = Mock()
        mock_author.name = "John Doe"
        mock_paper.authors = [mock_author]

        mock_search = Mock()
        mock_search.results.return_value = iter([mock_paper])
        mock_search_class.return_value = mock_search

        mock_session = AsyncMock()
        url = "https://arxiv.org/abs/2301.12345"

        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert error is None
        assert "# Test Paper" in markdown
        assert "**Authors**: John Doe" in markdown
        assert "**DOI**" not in markdown  # Should not include DOI if None
        assert "**Journal Reference**" not in markdown
        assert "## Comments" not in markdown

    @pytest.mark.asyncio
    @patch("tensortruth.utils.arxiv_handler.arxiv.Search")
    async def test_fetch_paper_not_found(self, mock_search_class, handler):
        """Test handling of non-existent papers."""
        # Mock empty results (StopIteration)
        mock_search = Mock()
        mock_search.results.return_value = iter([])
        mock_search_class.return_value = mock_search

        mock_session = AsyncMock()
        url = "https://arxiv.org/abs/9999.99999"

        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "http_error"
        assert markdown is None
        assert error == "Paper not found"

    @pytest.mark.asyncio
    async def test_fetch_invalid_url(self, handler):
        """Test handling of invalid URLs (ID extraction fails)."""
        mock_session = AsyncMock()
        url = "https://arxiv.org/list/cs.AI/recent"

        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "parse_error"
        assert markdown is None
        assert "Could not extract ArXiv ID" in error

    @pytest.mark.asyncio
    @patch("tensortruth.utils.arxiv_handler.asyncio.wait_for")
    @patch("tensortruth.utils.arxiv_handler.arxiv.Search")
    async def test_fetch_timeout(self, mock_search_class, mock_wait_for, handler):
        """Test handling of API timeouts."""
        # Mock asyncio.wait_for to raise TimeoutError
        mock_wait_for.side_effect = asyncio.TimeoutError()

        # Mock search (not actually called due to timeout)
        mock_search = Mock()
        mock_search_class.return_value = mock_search

        mock_session = AsyncMock()
        url = "https://arxiv.org/abs/1706.03762"

        markdown, status, error = await handler.fetch(url, mock_session, timeout=1)

        assert status == "timeout"
        assert markdown is None
        assert error == "ArXiv API timeout"

    @pytest.mark.asyncio
    @patch("tensortruth.utils.arxiv_handler.arxiv.Search")
    async def test_fetch_api_error(self, mock_search_class, handler):
        """Test handling of API errors."""
        # Mock search that raises an exception
        mock_search_class.side_effect = Exception("API connection failed")

        mock_session = AsyncMock()
        url = "https://arxiv.org/abs/1706.03762"

        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "parse_error"
        assert markdown is None
        assert "ArXiv API error" in error

    @pytest.mark.asyncio
    @patch("tensortruth.utils.arxiv_handler.arxiv.Search")
    async def test_fetch_with_versioned_url(self, mock_search_class, handler):
        """Test fetching paper with version number in URL."""
        mock_paper = Mock()
        mock_paper.title = "Versioned Paper"
        mock_paper.entry_id = "http://arxiv.org/abs/1234.5678v3"
        mock_paper.pdf_url = "https://arxiv.org/pdf/1234.5678v3"
        mock_paper.summary = "This is a versioned paper abstract. " * 10
        mock_paper.published = datetime(2012, 4, 15, tzinfo=timezone.utc)
        mock_paper.updated = datetime(2012, 5, 20, tzinfo=timezone.utc)
        mock_paper.categories = ["cs.LG"]
        mock_paper.primary_category = "cs.LG"
        mock_paper.doi = None
        mock_paper.journal_ref = None
        mock_paper.comment = None

        mock_author = Mock()
        mock_author.name = "Jane Smith"
        mock_paper.authors = [mock_author]

        mock_search = Mock()
        mock_search.results.return_value = iter([mock_paper])
        mock_search_class.return_value = mock_search

        mock_session = AsyncMock()
        url = "https://arxiv.org/abs/1234.5678v3"  # URL with version

        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert "**ArXiv ID**: 1234.5678" in markdown  # Version should be stripped
        # Verify the correct ID was searched (without version)
        mock_search_class.assert_called_once_with(id_list=["1234.5678"])
