"""
Built-in tools for TensorTruth agents.

This module provides async tool functions that wrap web search and fetching
utilities for use by agents (e.g., browse agent). These tools follow the
FunctionTool pattern and return string results suitable for LLM consumption.
"""

import asyncio
import json
import logging
import re
from typing import List, Optional, Union

import aiohttp
import arxiv

from tensortruth.utils.validation import validate_arxiv_id
from tensortruth.utils.web_search import (
    fetch_page_as_markdown,
    search_duckduckgo,
)

logger = logging.getLogger(__name__)


async def search_web(
    queries: Union[str, List[str]], max_results_per_query: int = 5
) -> str:
    """Search the web using DuckDuckGo with single or multiple queries.

    Supports multi-query parallel execution for comprehensive coverage.
    Results are combined and deduplicated by URL.

    Args:
        queries: Single query string or list of query strings
        max_results_per_query: Maximum results per query (default: 5)

    Returns:
        JSON with combined results:
        [{"url": "...", "title": "...", "snippet": "...", "query": "..."}]
    """
    # Normalize to list
    query_list = queries if isinstance(queries, list) else [queries]

    logger.info(
        f"search_web called with {len(query_list)} queries, "
        f"max_results_per_query: {max_results_per_query}"
    )

    try:
        # Execute all queries in parallel
        tasks = [
            search_duckduckgo(query, max_results=max_results_per_query)
            for query in query_list
        ]
        results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine and deduplicate by URL
        seen_urls = set()
        combined_results = []
        failed_queries = []

        for query, results in zip(query_list, results_per_query):
            # Handle exceptions from individual queries
            if isinstance(results, Exception):
                logger.warning(f"Query '{query}' failed: {results}")
                failed_queries.append(query)
                continue

            # Add results, deduplicating by URL
            if not isinstance(results, list):
                continue
            for result in results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    result["query"] = query  # Track which search returned this
                    # Future enhancement point for reranker:
                    # result["relevance_score"] = 1.0  # Placeholder for reranker
                    combined_results.append(result)

        # If all queries failed, return error
        if len(failed_queries) == len(query_list):
            error_msg = f"All {len(query_list)} queries failed"
            logger.error(error_msg)
            error_result = {"error": error_msg, "results": []}
            return json.dumps(error_result, indent=2)

        logger.info(
            f"search_web returned {len(combined_results)} unique results "
            f"from {len(query_list)} queries ({len(failed_queries)} failed)"
        )

        return json.dumps(combined_results, indent=2)

    except Exception as e:
        logger.error(f"search_web failed: {e}", exc_info=True)
        error_result = {"error": str(e), "results": []}
        return json.dumps(error_result, indent=2)


async def fetch_page(url: str, timeout: int = 10) -> str:
    """
    Fetch a web page and convert it to clean markdown.

    This tool fetches a web page and converts it to readable markdown format.
    Uses domain-specific handlers for Wikipedia, GitHub, arXiv, YouTube, etc.
    and falls back to generic HTML scraping for other sites.

    Args:
        url: URL to fetch
        timeout: Timeout in seconds (default: 10)

    Returns:
        Markdown content on success, or error message on failure.
        Error format: "Error: <message>"

    Example:
        >>> content = await fetch_page("https://wikipedia.org/wiki/Python")
        >>> # Returns markdown content or "Error: Failed to fetch page"
    """
    logger.info(f"fetch_page called with url: {url}")

    try:
        # Create aiohttp session for fetching
        async with aiohttp.ClientSession() as session:
            markdown_content, status, error_message = await fetch_page_as_markdown(
                url, session, timeout
            )

            # Check if fetch was successful
            if status == "success" and markdown_content:
                return markdown_content
            else:
                # Return error message
                error_msg = error_message or f"Failed to fetch page (status: {status})"
                logger.warning(f"fetch_page failed for {url}: {error_msg}")
                return f"Error: {error_msg}"

    except Exception as e:
        # Return error message (don't raise - agents need to handle errors)
        logger.error(f"fetch_page exception for {url}: {e}", exc_info=True)
        return f"Error: {str(e)}"


async def fetch_pages_batch(
    urls: List[str],
    timeout: int = 10,
    max_content_chars: Optional[int] = None,
    min_pages: int = 1,
) -> str:
    """Fetch multiple web pages with overflow protection.

    Enhanced version with overflow protection for context window management.
    Fetches pages one by one (not parallel) and stops when content limit reached.

    This tool is optimized for fetching multiple pages efficiently by:
    - Fetching pages sequentially to enable overflow detection
    - Reusing a single HTTP session (connection pooling)
    - Handling partial failures gracefully
    - Stopping when content limit is reached

    Args:
        urls: List of URLs to fetch (recommended: 3-5 URLs)
        timeout: Timeout in seconds per page (default: 10)
        max_content_chars: Maximum total content size (None = unlimited)
        min_pages: Minimum pages to fetch before checking overflow

    Returns:
        JSON object with:
        {
            "pages": [{"url": "...", "title": "...", "status": "success"/"failed",
                      "content": "markdown...", "error": "..."}],
            "overflow": bool,
            "total_chars": int
        }

        Status values:
        - "success": Page fetched and converted to markdown
        - "failed": Network error, timeout, or HTTP error
        - "skipped": Content too short or parse error

    Example:
        >>> results = await fetch_pages_batch(
        ...     ["https://en.wikipedia.org/wiki/Python"],
        ...     max_content_chars=100000
        ... )
        >>> # Returns JSON with overflow tracking
    """
    logger.info(
        f"fetch_pages_batch called with {len(urls)} URLs, "
        f"max_content_chars={max_content_chars}, min_pages={min_pages}"
    )

    if not urls:
        return json.dumps({"pages": [], "overflow": False, "total_chars": 0}, indent=2)

    try:
        # Single shared session for all fetches (connection pooling)
        async with aiohttp.ClientSession() as session:
            # Fetch pages one by one to enable overflow detection
            outputs: List[dict] = []
            total_chars = 0
            overflow = False

            for url in urls:
                # Check overflow before fetching (after min pages)
                if max_content_chars and len(outputs) >= min_pages:
                    if total_chars >= max_content_chars:
                        overflow = True
                        logger.info(
                            f"fetch_pages_batch: Content overflow at {total_chars} chars "
                            f"(limit: {max_content_chars}), stopping"
                        )
                        break

                # Fetch page
                try:
                    result = await fetch_page_as_markdown(url, session, timeout)
                except Exception as e:
                    result = (None, "failed", str(e))

                # Process result
                if isinstance(result, Exception):
                    # Exception during fetch
                    outputs.append(
                        {
                            "url": url,
                            "title": url.split("/")[-1] or url,
                            "status": "failed",
                            "content": None,
                            "error": str(result),
                        }
                    )
                    logger.warning(
                        f"fetch_pages_batch: {url} failed with exception: {result}"
                    )
                else:
                    markdown, status, error_msg = result

                    # Extract title from URL if markdown parsing failed
                    title = url.split("/")[-1] or url

                    # Try to extract title from markdown (first # heading)
                    if markdown and status == "success":
                        for line in markdown.split("\n")[:20]:
                            if line.startswith("# "):
                                title = line[2:].strip()
                                break

                    if status == "success" and markdown:
                        content_size = len(markdown)
                        total_chars += content_size
                        outputs.append(
                            {
                                "url": url,
                                "title": title,
                                "status": "success",
                                "content": markdown,
                                "error": None,
                            }
                        )
                        logger.debug(
                            f"fetch_pages_batch: {url} fetched {content_size} chars "
                            f"(total: {total_chars})"
                        )
                    else:
                        # Map status to simpler categories
                        simple_status = (
                            "skipped"
                            if status in ("too_short", "parse_error")
                            else "failed"
                        )
                        outputs.append(
                            {
                                "url": url,
                                "title": title,
                                "status": simple_status,
                                "content": None,
                                "error": error_msg,
                            }
                        )

            success_count = sum(1 for o in outputs if o["status"] == "success")
            logger.info(
                f"fetch_pages_batch completed: {success_count}/{len(urls)} successful, "
                f"total_chars={total_chars}, overflow={overflow}"
            )

            return json.dumps(
                {"pages": outputs, "overflow": overflow, "total_chars": total_chars},
                indent=2,
            )

    except Exception as e:
        logger.error(f"fetch_pages_batch failed: {e}", exc_info=True)
        # Return error for all URLs
        error_outputs = [
            {
                "url": url,
                "title": url,
                "status": "failed",
                "content": None,
                "error": f"Batch fetch failed: {str(e)}",
            }
            for url in urls
        ]
        return json.dumps(
            {"pages": error_outputs, "overflow": False, "total_chars": 0}, indent=2
        )


async def search_focused(query: str, domain: str, max_results: int = 5) -> str:
    """
    Search within a specific domain using DuckDuckGo site search.

    This tool performs a domain-focused search by building a site:{domain} query.
    Useful for finding content on specific websites like Stack Overflow, GitHub,
    official documentation sites, etc.

    Args:
        query: Search query string
        domain: Domain to search within (e.g., "stackoverflow.com", "github.com")
        max_results: Maximum number of results to return (default: 5)

    Returns:
        JSON string containing search results with format:
        [{"url": "...", "title": "...", "snippet": "..."}]
        On error, returns: {"error": "...", "results": []}

    Example:
        >>> results = await search_focused("Python async", "stackoverflow.com")
        >>> # Returns: '[{"url": "...", "title": "...", "snippet": "..."}]'
    """
    logger.info(f"search_focused called with query: {query}, domain: {domain}")

    try:
        # Build site-specific search query
        focused_query = f"site:{domain} {query}"
        logger.debug(f"Built focused query: {focused_query}")

        # Call DuckDuckGo search utility
        results = await search_duckduckgo(focused_query, max_results=max_results)

        # Return results as JSON string
        return json.dumps(results, indent=2)

    except Exception as e:
        # Return error as JSON (don't raise - agents need to handle errors)
        logger.error(f"search_focused failed: {e}", exc_info=True)
        error_result = {"error": str(e), "results": []}
        return json.dumps(error_result, indent=2)


_ARXIV_SORT_MAP = {
    "relevance": arxiv.SortCriterion.Relevance,
    "submitted": arxiv.SortCriterion.SubmittedDate,
    "updated": arxiv.SortCriterion.LastUpdatedDate,
}


async def search_arxiv(
    query: str, max_results: int = 5, sort_by: str = "relevance"
) -> str:
    """Search arXiv for academic papers.

    Args:
        query: Search query for arXiv papers.
        max_results: Maximum number of papers to return (default: 5).
        sort_by: Sort order — 'relevance', 'submitted', or 'updated'.

    Returns:
        JSON array of paper objects, or error JSON on failure.
    """
    logger.info(f"search_arxiv called with query={query!r}, max_results={max_results}")

    try:
        sort_criterion = _ARXIV_SORT_MAP.get(sort_by, arxiv.SortCriterion.Relevance)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
        )

        loop = asyncio.get_event_loop()

        def _fetch():
            return list(search.results())

        papers = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch),
            timeout=30,
        )

        results = []
        for paper in papers:
            # Truncate abstract to first ~300 chars for search overview
            abstract = paper.summary.strip()
            if len(abstract) > 300:
                abstract = abstract[:300].rsplit(" ", 1)[0] + "..."
            # Strip version suffix (e.g. "2301.12345v2" -> "2301.12345")
            raw_id = paper.entry_id.split("/abs/")[-1]
            arxiv_id = re.sub(r"v\d+$", "", raw_id)
            results.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors[:5]],
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "categories": paper.categories,
                    "abstract_snippet": abstract,
                    "pdf_url": paper.pdf_url,
                }
            )

        logger.info(f"search_arxiv returned {len(results)} papers")

        # Build output with cue for the orchestrator
        output = {"results": results, "total": len(results)}
        if results:
            ids = ", ".join(r["arxiv_id"] for r in results)
            output["hint"] = (
                f"Call get_arxiv_paper on promising IDs for full abstracts, "
                f"authors, and metadata: {ids}"
            )
        return json.dumps(output, indent=2)

    except asyncio.TimeoutError:
        logger.warning("search_arxiv timed out")
        return json.dumps({"error": "arXiv API timeout", "results": []}, indent=2)
    except Exception as e:
        logger.error(f"search_arxiv failed: {e}", exc_info=True)
        return json.dumps({"error": str(e), "results": []}, indent=2)


async def get_arxiv_paper(paper_id: str) -> str:
    """Get detailed info about a specific arXiv paper.

    Args:
        paper_id: ArXiv paper ID (e.g. '2301.12345', 'hep-th/9901001',
            or full arXiv URL).

    Returns:
        Structured markdown with paper metadata and abstract, or error string.
    """
    logger.info(f"get_arxiv_paper called with paper_id={paper_id!r}")

    # Validate and normalize
    normalized_id = validate_arxiv_id(paper_id)
    if not normalized_id:
        return f"Error: Invalid arXiv ID format: {paper_id}"

    try:
        search = arxiv.Search(id_list=[normalized_id])

        loop = asyncio.get_event_loop()

        def _fetch():
            try:
                return next(search.results())
            except StopIteration:
                return None

        paper = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch),
            timeout=15,
        )

        if paper is None:
            return f"Error: Paper not found: {normalized_id}"

        # Build structured markdown (matches arxiv_handler.py pattern)
        lines = []
        lines.append(f"# {paper.title}")
        lines.append("")
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"**ArXiv ID**: {normalized_id}")
        lines.append(f"**Authors**: {', '.join(a.name for a in paper.authors)}")
        lines.append(f"**Published**: {paper.published.strftime('%Y-%m-%d')}")
        lines.append(f"**Updated**: {paper.updated.strftime('%Y-%m-%d')}")
        if paper.categories:
            lines.append(f"**Categories**: {', '.join(paper.categories)}")
        if paper.primary_category:
            lines.append(f"**Primary Category**: {paper.primary_category}")
        lines.append(f"**PDF URL**: {paper.pdf_url}")
        lines.append(f"**ArXiv URL**: {paper.entry_id}")
        if paper.doi:
            lines.append(f"**DOI**: {paper.doi}")
        if paper.journal_ref:
            lines.append(f"**Journal Reference**: {paper.journal_ref}")
        lines.append("")
        lines.append("## Abstract")
        lines.append("")
        lines.append(paper.summary.strip())
        lines.append("")
        if paper.comment:
            lines.append("## Comments")
            lines.append("")
            lines.append(paper.comment.strip())
            lines.append("")

        return "\n".join(lines)

    except asyncio.TimeoutError:
        logger.warning(f"get_arxiv_paper timed out for {normalized_id}")
        return f"Error: arXiv API timeout for paper {normalized_id}"
    except Exception as e:
        logger.error(f"get_arxiv_paper failed: {e}", exc_info=True)
        return f"Error: {str(e)}"
