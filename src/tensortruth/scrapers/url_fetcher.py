"""URL fetching and HTML-to-markdown conversion for document ingestion."""

import logging
from typing import Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from tensortruth.utils.web_search import BROWSER_HEADERS, clean_html_for_content

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 500_000
MIN_CONTENT_CHARS = 50


def fetch_url_as_markdown(url: str, timeout: int = 15) -> Tuple[str, str]:
    """Fetch a URL and convert its HTML content to markdown.

    Args:
        url: The URL to fetch (must be http or https).
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (markdown_content, page_title).

    Raises:
        ValueError: If URL format is invalid, content type is not HTML,
            or fetched content is too short.
        ConnectionError: If HTTP request fails (non-200, timeout, network error).
    """
    # Validate URL format
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid URL format: {url}")

    # Fetch URL content
    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout)
    except requests.exceptions.Timeout:
        raise ConnectionError(f"Timeout fetching URL: {url}")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Network error fetching URL: {e}")

    if resp.status_code != 200:
        raise ConnectionError(f"HTTP {resp.status_code} fetching URL: {url}")

    # Check content type
    content_type = resp.headers.get("Content-Type", "")
    allowed_types = ("text/html", "text/plain", "application/xhtml+xml")
    if not any(content_type.startswith(t) for t in allowed_types):
        raise ValueError(f"URL does not point to HTML content: {content_type}")

    # Parse HTML
    soup = BeautifulSoup(resp.content, "html.parser")

    # Extract page title
    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else ""

    # Extract main content using multi-selector fallback
    content = None
    for selector in ["main", "article", '[role="main"]', ".content", "#content"]:
        content = soup.select_one(selector)
        if content:
            break

    # Fallback to body
    if not content:
        content = soup.find("body")

    if not content:
        raise ValueError("No content found in page")

    # Clean HTML
    content = clean_html_for_content(content)

    # Convert to markdown
    markdown = md(str(content), heading_style="ATX", code_language="python")

    # Truncate if very large
    if len(markdown) > MAX_CONTENT_CHARS:
        logger.warning(
            f"Content from {url} truncated from {len(markdown)} to {MAX_CONTENT_CHARS} chars"
        )
        markdown = markdown[:MAX_CONTENT_CHARS]

    # Check minimum content length
    if len(markdown.strip()) < MIN_CONTENT_CHARS:
        raise ValueError("Fetched content is too short")

    return markdown, page_title
