"""Unit tests for URL fetcher module."""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.scrapers.url_fetcher import fetch_url_as_markdown


def _make_response(
    status_code=200,
    content_type="text/html; charset=utf-8",
    content=b"",
):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"Content-Type": content_type}
    resp.content = content
    return resp


SAMPLE_HTML = b"""
<html>
<head><title>Test Page Title</title></head>
<body>
<main>
<h1>Hello World</h1>
<p>This is a test page with enough content to pass the minimum length check.
   It contains multiple paragraphs of text to make the test realistic.</p>
<p>Second paragraph with additional information about the topic at hand.</p>
</main>
</body>
</html>
"""

MINIMAL_HTML = b"""
<html>
<head><title>Empty</title></head>
<body><p>Hi</p></body>
</html>
"""


class TestFetchValidHtml:
    """Test successful HTML fetching and conversion."""

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_valid_html(self, mock_get):
        """Should fetch HTML, convert to markdown, and return content + title."""
        mock_get.return_value = _make_response(content=SAMPLE_HTML)

        markdown, title = fetch_url_as_markdown("https://example.com/page")

        assert "Hello World" in markdown
        assert "test page" in markdown
        assert title == "Test Page Title"
        mock_get.assert_called_once()


class TestFetchInvalidUrl:
    """Test URL validation."""

    def test_fetch_invalid_url_no_scheme(self):
        """Should raise ValueError for URL without scheme."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            fetch_url_as_markdown("example.com/page")

    def test_fetch_invalid_url_ftp_scheme(self):
        """Should raise ValueError for non-http(s) scheme."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            fetch_url_as_markdown("ftp://example.com/page")

    def test_fetch_invalid_url_no_netloc(self):
        """Should raise ValueError for URL without netloc."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            fetch_url_as_markdown("http://")


class TestFetchHttpError:
    """Test HTTP error handling."""

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_http_404(self, mock_get):
        """Should raise ConnectionError for 404 response."""
        mock_get.return_value = _make_response(status_code=404)

        with pytest.raises(ConnectionError, match="HTTP 404"):
            fetch_url_as_markdown("https://example.com/missing")

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_http_500(self, mock_get):
        """Should raise ConnectionError for 500 response."""
        mock_get.return_value = _make_response(status_code=500)

        with pytest.raises(ConnectionError, match="HTTP 500"):
            fetch_url_as_markdown("https://example.com/error")

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_timeout(self, mock_get):
        """Should raise ConnectionError on timeout."""
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        with pytest.raises(ConnectionError, match="Timeout"):
            fetch_url_as_markdown("https://example.com/slow")

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_network_error(self, mock_get):
        """Should raise ConnectionError on network failure."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("refused")

        with pytest.raises(ConnectionError, match="Network error"):
            fetch_url_as_markdown("https://example.com/down")


class TestFetchNonHtmlContentType:
    """Test content type validation."""

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_pdf_content_type(self, mock_get):
        """Should raise ValueError for PDF content type."""
        mock_get.return_value = _make_response(content_type="application/pdf")

        with pytest.raises(ValueError, match="URL does not point to HTML content"):
            fetch_url_as_markdown("https://example.com/file.pdf")

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_image_content_type(self, mock_get):
        """Should raise ValueError for image content type."""
        mock_get.return_value = _make_response(content_type="image/png")

        with pytest.raises(ValueError, match="URL does not point to HTML content"):
            fetch_url_as_markdown("https://example.com/image.png")

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_json_content_type(self, mock_get):
        """Should raise ValueError for JSON content type."""
        mock_get.return_value = _make_response(content_type="application/json")

        with pytest.raises(ValueError, match="URL does not point to HTML content"):
            fetch_url_as_markdown("https://example.com/api/data")


class TestFetchEmptyContent:
    """Test minimum content length enforcement."""

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_empty_content(self, mock_get):
        """Should raise ValueError for content shorter than 50 chars."""
        mock_get.return_value = _make_response(content=MINIMAL_HTML)

        with pytest.raises(ValueError, match="Fetched content is too short"):
            fetch_url_as_markdown("https://example.com/empty")


class TestFetchExtractsTitle:
    """Test title extraction."""

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_extracts_title(self, mock_get):
        """Should extract title from <title> tag."""
        mock_get.return_value = _make_response(content=SAMPLE_HTML)

        _, title = fetch_url_as_markdown("https://example.com/page")

        assert title == "Test Page Title"

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_no_title_returns_empty(self, mock_get):
        """Should return empty string when no title tag exists."""
        html = b"""
        <html><body>
        <main>
        <p>This page has no title tag but has enough content to pass
           the minimum length check for the fetcher validation.</p>
        </main>
        </body></html>
        """
        mock_get.return_value = _make_response(content=html)

        _, title = fetch_url_as_markdown("https://example.com/no-title")

        assert title == ""


class TestFetchLargeContentTruncated:
    """Test content truncation for very large pages."""

    @patch("tensortruth.scrapers.url_fetcher.requests.get")
    def test_fetch_large_content_truncated(self, mock_get):
        """Should truncate content at 500,000 chars."""
        # Create HTML with content larger than 500k chars
        large_text = "A" * 600_000
        html = f"""
        <html>
        <head><title>Large Page</title></head>
        <body><main><p>{large_text}</p></main></body>
        </html>
        """.encode()
        mock_get.return_value = _make_response(content=html)

        markdown, title = fetch_url_as_markdown("https://example.com/large")

        assert len(markdown) <= 500_000
        assert title == "Large Page"
