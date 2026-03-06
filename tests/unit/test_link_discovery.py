"""Unit tests for link extraction and metadata fetching (deep browsing)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tensortruth.utils.web_search import (
    extract_links_from_markdown,
    fetch_link_metadata,
)

# ============================================================================
# extract_links_from_markdown
# ============================================================================


class TestExtractLinksFromMarkdown:
    def test_basic_extraction(self):
        md = (
            "Check out [PyTorch docs](https://pytorch.org/docs) "
            "and [TensorFlow](https://www.tensorflow.org)."
        )
        links = extract_links_from_markdown(md, "https://example.com")
        assert len(links) == 2
        assert links[0] == ("PyTorch docs", "https://pytorch.org/docs")
        assert links[1] == ("TensorFlow", "https://www.tensorflow.org")

    def test_relative_urls_resolved(self):
        md = "See [details](/about/team) and [more](subpage)."
        links = extract_links_from_markdown(md, "https://example.com/page/")
        assert links[0] == ("details", "https://example.com/about/team")
        assert links[1] == ("more", "https://example.com/page/subpage")

    def test_fragment_only_filtered(self):
        md = "Jump to [section](#heading) or [other page](https://example.com)."
        links = extract_links_from_markdown(md, "https://example.com")
        assert len(links) == 1
        assert links[0][1] == "https://example.com"

    def test_fragment_stripped_from_url(self):
        md = "[Article](https://example.com/page#section)"
        links = extract_links_from_markdown(md, "https://example.com")
        assert links[0][1] == "https://example.com/page"

    def test_non_http_filtered(self):
        md = "[mail](mailto:a@b.com) [ftp](ftp://x.com) [ok](https://x.com)"
        links = extract_links_from_markdown(md, "https://example.com")
        assert len(links) == 1
        assert links[0][1] == "https://x.com"

    def test_boilerplate_filtered(self):
        md = (
            "[Login](/login) [Signup](/signup) [Edit](/edit) "
            "[Settings](/settings) [Good](/article)"
        )
        links = extract_links_from_markdown(md, "https://example.com")
        assert len(links) == 1
        assert links[0] == ("Good", "https://example.com/article")

    def test_exclude_urls(self):
        md = "[A](https://a.com) [B](https://b.com)"
        links = extract_links_from_markdown(
            md, "https://example.com", exclude_urls={"https://a.com"}
        )
        assert len(links) == 1
        assert links[0][1] == "https://b.com"

    def test_deduplication(self):
        md = "[Link1](https://x.com) [Link2](https://x.com)"
        links = extract_links_from_markdown(md, "https://example.com")
        assert len(links) == 1

    def test_max_15_links(self):
        md = " ".join(f"[L{i}](https://example.com/p{i})" for i in range(20))
        links = extract_links_from_markdown(md, "https://example.com")
        assert len(links) == 15

    def test_empty_anchor_filtered(self):
        md = "[](https://example.com/empty)"
        links = extract_links_from_markdown(md, "https://example.com")
        assert len(links) == 0

    def test_wikipedia_special_pages_filtered(self):
        md = (
            "[Help](https://en.wikipedia.org/wiki/Help:Contents) "
            "[Article](https://en.wikipedia.org/wiki/Python)"
        )
        links = extract_links_from_markdown(md, "https://en.wikipedia.org")
        assert len(links) == 1
        assert "Python" in links[0][1]


# ============================================================================
# fetch_link_metadata
# ============================================================================


class TestFetchLinkMetadata:
    @pytest.mark.asyncio
    async def test_basic_metadata_fetch(self):
        html = (
            b"<html><head>"
            b"<title>Test Page</title>"
            b'<meta name="description" content="A test description">'
            b"</head><body></body></html>"
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.content = _mock_content_iter(html)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        links = [("Test Link", "https://example.com/page")]
        results = await fetch_link_metadata(links, mock_session)

        assert len(results) == 1
        assert results[0]["title"] == "Test Page"
        assert results[0]["description"] == "A test description"
        assert results[0]["fetchable"] is True

    @pytest.mark.asyncio
    async def test_failed_fetch_returns_unfetchable(self):
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        links = [("Bad Link", "https://example.com/404")]
        results = await fetch_link_metadata(links, mock_session)

        assert len(results) == 1
        assert results[0]["fetchable"] is False
        assert results[0]["anchor_text"] == "Bad Link"

    @pytest.mark.asyncio
    async def test_exception_returns_unfetchable(self):
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Connection error"))

        links = [("Error Link", "https://example.com/error")]
        results = await fetch_link_metadata(links, mock_session)

        assert len(results) == 1
        assert results[0]["fetchable"] is False

    @pytest.mark.asyncio
    async def test_max_links_respected(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.content = _mock_content_iter(b"<head><title>T</title></head>")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        links = [(f"Link{i}", f"https://example.com/p{i}") for i in range(10)]
        results = await fetch_link_metadata(links, mock_session, max_links=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_non_html_content_type(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        links = [("PDF", "https://example.com/file.pdf")]
        results = await fetch_link_metadata(links, mock_session)

        assert results[0]["fetchable"] is False


def _mock_content_iter(data: bytes):
    """Create a mock async content iterator."""

    class MockContent:
        async def iter_chunked(self, size):
            yield data

    return MockContent()
