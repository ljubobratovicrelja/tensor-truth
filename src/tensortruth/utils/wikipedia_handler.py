"""Wikipedia-specific content handler for web search.

Uses Wikipedia API to fetch clean, structured content instead of scraping HTML.
"""

import asyncio
import logging
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

import aiohttp
import wikipediaapi

from .domain_handlers import ContentHandler, register_handler

logger = logging.getLogger(__name__)


class WikipediaHandler(ContentHandler):
    """Handler for Wikipedia URLs using the Wikipedia API."""

    @property
    def name(self) -> str:
        return "Wikipedia"

    def matches(self, url: str) -> bool:
        """Check if URL is a Wikipedia page."""
        try:
            parsed = urlparse(url)
            # Match any wikipedia.org domain (en.wikipedia.org, de.wikipedia.org, etc.)
            return "wikipedia.org" in parsed.netloc and "/wiki/" in parsed.path
        except Exception:
            return False

    def _extract_page_title(self, url: str) -> Optional[str]:
        """
        Extract Wikipedia page title from URL.

        Args:
            url: Wikipedia URL

        Returns:
            Page title or None if extraction fails
        """
        try:
            parsed = urlparse(url)
            # URL format: https://en.wikipedia.org/wiki/Page_Title
            # Extract "Page_Title" part
            path_parts = parsed.path.split("/wiki/")
            if len(path_parts) >= 2:
                # URL decode and replace underscores with spaces
                title = unquote(path_parts[1])
                # Remove anchor links (e.g., #section)
                title = title.split("#")[0]
                return title.replace("_", " ")
        except Exception as e:
            logger.warning(f"Failed to extract Wikipedia title from {url}: {e}")
        return None

    def _extract_language(self, url: str) -> str:
        """
        Extract language code from Wikipedia URL.

        Args:
            url: Wikipedia URL (e.g., https://en.wikipedia.org/wiki/...)

        Returns:
            Language code (e.g., 'en', 'de', 'fr') or 'en' as default
        """
        try:
            parsed = urlparse(url)
            # Domain format: en.wikipedia.org, de.wikipedia.org, etc.
            parts = parsed.netloc.split(".")
            if len(parts) >= 2:
                subdomain = parts[0]
                # Check if subdomain is a valid language code (2-3 letters)
                # and not 'www' or 'wikipedia' itself
                if (
                    subdomain
                    and len(subdomain) in [2, 3]
                    and subdomain not in ["www", "m"]
                ):
                    return subdomain
        except Exception:
            pass
        return "en"  # Default to English

    async def fetch(
        self, url: str, session: aiohttp.ClientSession, timeout: int = 10
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Fetch Wikipedia content using the Wikipedia API.

        Args:
            url: Wikipedia page URL
            session: aiohttp ClientSession (not used, but kept for interface compatibility)
            timeout: Timeout in seconds

        Returns:
            Tuple of (markdown_content, status, error_message)
        """
        logger.info(f"Fetching Wikipedia page: {url}")

        # Extract page title and language
        page_title = self._extract_page_title(url)
        if not page_title:
            return None, "parse_error", "Could not extract page title from URL"

        language = self._extract_language(url)

        try:
            # Create Wikipedia API client with user agent
            wiki_wiki = wikipediaapi.Wikipedia(
                user_agent="TensorTruth/1.0 (https://github.com/yourusername/tensor-truth)",
                language=language,
            )

            # Run Wikipedia API call in executor to avoid blocking
            loop = asyncio.get_event_loop()
            page = await asyncio.wait_for(
                loop.run_in_executor(None, wiki_wiki.page, page_title),
                timeout=timeout,
            )

            # Check if page exists
            if not page.exists():
                return None, "http_error", "Page not found"

            # Build markdown content
            markdown_lines = []

            # Add title
            markdown_lines.append(f"# {page.title}")
            markdown_lines.append("")

            # Add source metadata
            markdown_lines.append(f"<!-- Source: {page.fullurl} -->")
            markdown_lines.append("")

            # Add summary
            if page.summary:
                markdown_lines.append("## Summary")
                markdown_lines.append("")
                # Summary is typically the first paragraph
                summary_paragraphs = page.summary.strip().split("\n\n")
                for para in summary_paragraphs[:3]:  # Limit to 3 paragraphs
                    if para.strip():
                        markdown_lines.append(para.strip())
                        markdown_lines.append("")

            # Add main content with sections
            if page.text:
                markdown_lines.append("## Content")
                markdown_lines.append("")

                def add_section(section, level=3):
                    """Recursively add sections and subsections."""
                    if section.title and section.title != page.title:
                        # Add section header
                        markdown_lines.append(f"{'#' * level} {section.title}")
                        markdown_lines.append("")

                    # Add section text (limit to avoid huge pages)
                    if section.text:
                        text_lines = section.text.strip().split("\n")
                        # Limit to first 50 lines per section to avoid overwhelming content
                        for line in text_lines[:50]:
                            if line.strip():
                                markdown_lines.append(line.strip())
                        markdown_lines.append("")

                    # Recursively add subsections
                    for subsection in list(section.sections)[:5]:  # Limit subsections
                        add_section(subsection, level + 1)

                # Add all top-level sections
                for section in list(page.sections)[:10]:  # Limit to 10 main sections
                    add_section(section)

            markdown = "\n".join(markdown_lines)

            # Quality check
            if len(markdown.strip()) < 100:
                return None, "too_short", "Wikipedia content too short"

            logger.info(
                f"✅ Fetched Wikipedia page '{page.title}' ({len(markdown)} chars)"
            )
            return markdown, "success", None

        except asyncio.TimeoutError:
            error_msg = "Wikipedia API timeout"
            logger.warning(f"Timeout fetching Wikipedia page: {page_title}")
            return None, "timeout", error_msg

        except Exception as e:
            error_msg = f"Wikipedia API error: {str(e)}"
            logger.error(f"Error fetching Wikipedia page {page_title}: {e}")
            return None, "parse_error", error_msg


# Register the Wikipedia handler
register_handler(WikipediaHandler())
