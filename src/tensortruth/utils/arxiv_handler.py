"""ArXiv-specific content handler for web search.

Uses ArXiv API to fetch clean, structured metadata and abstracts.
"""

import asyncio
import logging
import re
from typing import Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import arxiv

from .domain_handlers import ContentHandler, register_handler

logger = logging.getLogger(__name__)


class ArxivHandler(ContentHandler):
    """Handler for ArXiv URLs using the ArXiv API."""

    @property
    def name(self) -> str:
        return "ArXiv"

    def matches(self, url: str) -> bool:
        """Check if URL is an ArXiv page."""
        try:
            parsed = urlparse(url)
            # Match arxiv.org domain with /abs/ or /pdf/ paths
            return "arxiv.org" in parsed.netloc and (
                "/abs/" in parsed.path or "/pdf/" in parsed.path
            )
        except Exception:
            return False

    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """
        Extract ArXiv paper ID from URL.

        Supports various formats:
        - https://arxiv.org/abs/1706.03762
        - https://arxiv.org/pdf/1706.03762.pdf
        - https://arxiv.org/abs/1706.03762v2 (with version)

        Args:
            url: ArXiv URL

        Returns:
            Paper ID (without version) or None if extraction fails
        """
        try:
            # Match patterns like: 1706.03762, 1706.03762v2, 2301.12345
            # ArXiv IDs are: YYMM.NNNNN or YYMM.NNNNNN (4 or 5 digits)
            match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
            if match:
                # Return ID without version number
                return match.group(1)
        except Exception as e:
            logger.warning(f"Failed to extract ArXiv ID from {url}: {e}")
        return None

    async def fetch(
        self, url: str, session: aiohttp.ClientSession, timeout: int = 10
    ) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Fetch ArXiv paper metadata using the ArXiv API.

        Args:
            url: ArXiv paper URL
            session: aiohttp ClientSession (not used, but kept for interface compatibility)
            timeout: Timeout in seconds

        Returns:
            Tuple of (markdown_content, status, error_message)
        """
        logger.info(f"Fetching ArXiv paper: {url}")

        # Extract paper ID
        arxiv_id = self._extract_arxiv_id(url)
        if not arxiv_id:
            return None, "parse_error", "Could not extract ArXiv ID from URL"

        try:
            # Create ArXiv search client
            search = arxiv.Search(id_list=[arxiv_id])

            # Run ArXiv API call in executor to avoid blocking
            loop = asyncio.get_event_loop()

            # Get the first (and only) result
            # Wrap in a function to handle StopIteration properly
            def get_paper():
                try:
                    return next(search.results())
                except StopIteration:
                    return None

            paper = await asyncio.wait_for(
                loop.run_in_executor(None, get_paper),
                timeout=timeout,
            )

            if paper is None:
                return None, "http_error", "Paper not found"

            # Build markdown content
            markdown_lines = []

            # Add title
            markdown_lines.append(f"# {paper.title}")
            markdown_lines.append("")

            # Add source metadata
            markdown_lines.append(f"<!-- Source: {url} -->")
            markdown_lines.append(f"<!-- ArXiv ID: {arxiv_id} -->")
            markdown_lines.append("")

            # Add metadata section
            markdown_lines.append("## Metadata")
            markdown_lines.append("")
            markdown_lines.append(f"**ArXiv ID**: {arxiv_id}")
            markdown_lines.append(
                f"**Authors**: {', '.join([a.name for a in paper.authors])}"
            )
            markdown_lines.append(
                f"**Published**: {paper.published.strftime('%Y-%m-%d')}"
            )
            markdown_lines.append(f"**Updated**: {paper.updated.strftime('%Y-%m-%d')}")

            # Add categories
            if paper.categories:
                markdown_lines.append(f"**Categories**: {', '.join(paper.categories)}")
            if paper.primary_category:
                markdown_lines.append(f"**Primary Category**: {paper.primary_category}")

            # Add links
            markdown_lines.append(f"**PDF URL**: {paper.pdf_url}")
            markdown_lines.append(f"**ArXiv URL**: {paper.entry_id}")

            # Add DOI and journal reference if available
            if paper.doi:
                markdown_lines.append(f"**DOI**: {paper.doi}")
            if paper.journal_ref:
                markdown_lines.append(f"**Journal Reference**: {paper.journal_ref}")

            markdown_lines.append("")

            # Add abstract
            markdown_lines.append("## Abstract")
            markdown_lines.append("")
            markdown_lines.append(paper.summary.strip())
            markdown_lines.append("")

            # Add comment if available
            if paper.comment:
                markdown_lines.append("## Comments")
                markdown_lines.append("")
                markdown_lines.append(paper.comment.strip())
                markdown_lines.append("")

            markdown = "\n".join(markdown_lines)

            # Quality check
            if len(markdown.strip()) < 100:
                return None, "too_short", "ArXiv content too short"

            logger.info(
                f"✅ Fetched ArXiv paper '{paper.title}' ({len(markdown)} chars)"
            )
            return markdown, "success", None

        except asyncio.TimeoutError:
            error_msg = "ArXiv API timeout"
            logger.warning(f"Timeout fetching ArXiv paper: {arxiv_id}")
            return None, "timeout", error_msg

        except Exception as e:
            error_msg = f"ArXiv API error: {str(e)}"
            logger.error(f"Error fetching ArXiv paper {arxiv_id}: {e}")
            return None, "parse_error", error_msg


# Register the ArXiv handler
register_handler(ArxivHandler())
