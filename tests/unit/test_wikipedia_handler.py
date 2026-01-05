"""Tests for Wikipedia handler."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from tensortruth.utils.wikipedia_handler import WikipediaHandler


class TestWikipediaHandler:
    """Test suite for WikipediaHandler."""

    @pytest.fixture
    def handler(self):
        """Create a WikipediaHandler instance."""
        return WikipediaHandler()

    def test_name(self, handler):
        """Test handler name property."""
        assert handler.name == "Wikipedia"

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://en.wikipedia.org/wiki/Python_(programming_language)", True),
            ("https://de.wikipedia.org/wiki/Python_(Programmiersprache)", True),
            ("https://fr.wikipedia.org/wiki/Python_(langage)", True),
            ("https://en.wikipedia.org/wiki/Machine_learning", True),
            ("https://wikipedia.org/wiki/Test", True),
            ("https://example.com/wiki/page", False),
            ("https://github.com/repo", False),
            ("https://en.wikipedia.org/about", False),  # No /wiki/ path
            ("not-a-url", False),
        ],
    )
    def test_matches(self, handler, url, expected):
        """Test URL matching for Wikipedia pages."""
        assert handler.matches(url) == expected

    @pytest.mark.parametrize(
        "url,expected_title",
        [
            ("https://en.wikipedia.org/wiki/Python", "Python"),
            ("https://en.wikipedia.org/wiki/Machine_learning", "Machine learning"),
            (
                "https://en.wikipedia.org/wiki/Artificial_intelligence#History",
                "Artificial intelligence",
            ),
            (
                "https://en.wikipedia.org/wiki/Deep%20learning",
                "Deep learning",
            ),  # URL encoded
        ],
    )
    def test_extract_page_title(self, handler, url, expected_title):
        """Test page title extraction from URLs."""
        assert handler._extract_page_title(url) == expected_title

    @pytest.mark.parametrize(
        "url,expected_lang",
        [
            ("https://en.wikipedia.org/wiki/Test", "en"),
            ("https://de.wikipedia.org/wiki/Test", "de"),
            ("https://fr.wikipedia.org/wiki/Test", "fr"),
            ("https://es.wikipedia.org/wiki/Test", "es"),
            ("https://wikipedia.org/wiki/Test", "en"),  # Default to English
        ],
    )
    def test_extract_language(self, handler, url, expected_lang):
        """Test language extraction from URLs."""
        assert handler._extract_language(url) == expected_lang

    @pytest.mark.asyncio
    @patch("tensortruth.utils.wikipedia_handler.wikipediaapi.Wikipedia")
    async def test_fetch_success(self, mock_wikipedia_class, handler):
        """Test successful Wikipedia page fetch."""
        # Mock Wikipedia page
        mock_wiki_page = Mock()
        mock_wiki_page.exists.return_value = True
        mock_wiki_page.title = "Test Page"
        mock_wiki_page.fullurl = "https://en.wikipedia.org/wiki/Test_Page"
        mock_wiki_page.summary = "This is a test summary."
        mock_wiki_page.text = "Full page text content."

        # Mock section
        mock_section = Mock()
        mock_section.title = "Section 1"
        mock_section.text = "Content for section 1."
        mock_section.sections = []
        mock_wiki_page.sections = [mock_section]

        # Mock Wikipedia client
        mock_wiki_client = Mock()
        mock_wiki_client.page.return_value = mock_wiki_page
        mock_wikipedia_class.return_value = mock_wiki_client

        # Mock session (not used but required by interface)
        mock_session = AsyncMock()

        url = "https://en.wikipedia.org/wiki/Test_Page"
        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert error is None
        assert markdown is not None
        assert "# Test Page" in markdown
        assert "This is a test summary." in markdown
        assert "### Section 1" in markdown
        assert "https://en.wikipedia.org/wiki/Test_Page" in markdown

    @pytest.mark.asyncio
    @patch("tensortruth.utils.wikipedia_handler.wikipediaapi.Wikipedia")
    async def test_fetch_page_not_found(self, mock_wikipedia_class, handler):
        """Test handling of non-existent pages."""
        # Mock Wikipedia page that doesn't exist
        mock_wiki_page = Mock()
        mock_wiki_page.exists.return_value = False

        mock_wiki_client = Mock()
        mock_wiki_client.page.return_value = mock_wiki_page
        mock_wikipedia_class.return_value = mock_wiki_client

        mock_session = AsyncMock()
        url = "https://en.wikipedia.org/wiki/Nonexistent_Page"

        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "http_error"
        assert markdown is None
        assert error == "Page not found"

    @pytest.mark.asyncio
    async def test_fetch_invalid_url(self, handler):
        """Test handling of invalid URLs (title extraction fails)."""
        mock_session = AsyncMock()
        url = "https://en.wikipedia.org/not-a-valid-wiki-url"

        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "parse_error"
        assert markdown is None
        assert "Could not extract page title" in error

    @pytest.mark.asyncio
    @patch("tensortruth.utils.wikipedia_handler.wikipediaapi.Wikipedia")
    async def test_fetch_with_different_language(self, mock_wikipedia_class, handler):
        """Test fetching from non-English Wikipedia."""
        mock_wiki_page = Mock()
        mock_wiki_page.exists.return_value = True
        mock_wiki_page.title = "Test"
        mock_wiki_page.fullurl = "https://de.wikipedia.org/wiki/Test"
        mock_wiki_page.summary = "Deutsche Zusammenfassung."
        mock_wiki_page.text = "Inhalt auf Deutsch."
        mock_wiki_page.sections = []

        mock_wiki_client = Mock()
        mock_wiki_client.page.return_value = mock_wiki_page
        mock_wikipedia_class.return_value = mock_wiki_client

        mock_session = AsyncMock()
        url = "https://de.wikipedia.org/wiki/Test"

        markdown, status, error = await handler.fetch(url, mock_session, timeout=10)

        assert status == "success"
        assert markdown is not None
        # Verify German language was used
        mock_wikipedia_class.assert_called_once()
        call_kwargs = mock_wikipedia_class.call_args[1]
        assert call_kwargs["language"] == "de"
