"""Web search utilities using DuckDuckGo with LLM-based summarization."""

import asyncio
import logging
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
)

import aiohttp
from bs4 import BeautifulSoup, Tag
from ddgs import DDGS
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.llms.ollama import Ollama
from markdownify import markdownify as md

from tensortruth.core.deprecation import deprecated
from tensortruth.core.ollama import check_thinking_support
from tensortruth.core.synthesis import (
    CitationStyle,
    SynthesisConfig,
)
from tensortruth.core.synthesis import synthesize_with_llm_stream as core_synthesize

# Import handlers to register them (must be after this module is defined)
# These are imported at module load time to register handlers via decorators
from . import arxiv_handler  # noqa: F401, E402
from . import github_handler  # noqa: F401, E402
from . import wikipedia_handler  # noqa: F401, E402
from . import youtube_handler  # noqa: F401, E402
from .domain_handlers import get_handler_for_url  # noqa: E402

# =============================================================================
# Reranking Utilities
# =============================================================================


def get_reranker_for_web(
    model_name: str,
    device: str = "cuda",
    top_n: int = 100,
) -> SentenceTransformerRerank:
    """Get reranker instance for web search.

    Reuses the RAG ModelManager singleton for efficient model caching.

    Args:
        model_name: HuggingFace model path for the reranker
        device: Device to load model on ("cuda", "cpu", "mps")
        top_n: High top_n since we sort all results, not filter

    Returns:
        SentenceTransformerRerank instance ready for use
    """
    from tensortruth.services.model_manager import ModelManager

    manager = ModelManager.get_instance()
    return manager.get_reranker(model_name=model_name, top_n=top_n, device=device)


def rerank_search_results(
    query: str,
    results: List[Dict[str, str]],
    top_n: int,
    reranker: SentenceTransformerRerank,
) -> List[Tuple[Dict[str, str], float]]:
    """Rerank search results by title+snippet relevance.

    Creates TextNodes from title+snippet and uses the reranker to score them.
    Returns ALL results sorted by score (highest first), with scores attached.

    Args:
        query: The search query to rank against
        results: DDG search results with 'url', 'title', 'snippet' keys
        top_n: Unused, kept for API compatibility (returns all results)
        reranker: SentenceTransformerRerank instance to use

    Returns:
        List of (result_dict, score) tuples sorted by score descending
    """
    if not results:
        return []

    # Create TextNodes from title+snippet
    nodes_with_score = []
    for result in results:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        text = f"{title}\n{snippet}"

        node = TextNode(text=text)
        # Store original result in metadata for retrieval
        node.metadata = {"original_result": result}
        nodes_with_score.append(NodeWithScore(node=node, score=0.0))

    # Rerank using the reranker
    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker.postprocess_nodes(nodes_with_score, query_bundle)

    # Extract results with scores, sorted by score descending
    ranked_results: List[Tuple[Dict[str, str], float]] = []
    for node_with_score in ranked_nodes:
        original_result = node_with_score.node.metadata.get("original_result")
        if original_result:
            # Convert numpy float32 to Python float for JSON serialization
            score = (
                float(node_with_score.score)
                if node_with_score.score is not None
                else 0.0
            )
            ranked_results.append((original_result, score))

    # Sort by score descending (reranker may not guarantee order)
    ranked_results.sort(key=lambda x: x[1], reverse=True)

    return ranked_results


def rerank_fetched_pages(
    query: str,
    custom_instructions: Optional[str],
    pages: List[Tuple[str, str, str]],
    reranker: SentenceTransformerRerank,
    max_content_chars: int = 2000,
) -> List[Tuple[Tuple[str, str, str], float]]:
    """Rerank fetched pages by content relevance.

    Query is enhanced with custom_instructions if provided to influence
    the ranking based on user intent.

    Args:
        query: The search query to rank against
        custom_instructions: Optional user instructions to include in ranking
        pages: List of (url, title, content) tuples
        reranker: SentenceTransformerRerank instance to use
        max_content_chars: Max chars of content to use for ranking (efficiency)

    Returns:
        List of (page_tuple, score) tuples sorted by score descending
    """
    if not pages:
        return []

    # Build ranking query with optional custom instructions
    if custom_instructions:
        ranking_query = f"{query}\n\nUser focus: {custom_instructions}"
    else:
        ranking_query = query

    # Create TextNodes from page content (truncated for efficiency)
    nodes_with_score = []
    for page in pages:
        url, title, content = page
        # Truncate content for efficient ranking
        truncated_content = content[:max_content_chars]
        text = f"{title}\n\n{truncated_content}"

        node = TextNode(text=text)
        # Store original page tuple in metadata
        node.metadata = {"original_page": page}
        nodes_with_score.append(NodeWithScore(node=node, score=0.0))

    # Rerank using the reranker
    query_bundle = QueryBundle(query_str=ranking_query)
    ranked_nodes = reranker.postprocess_nodes(nodes_with_score, query_bundle)

    # Extract pages with scores, sorted by score descending
    ranked_pages: List[Tuple[Tuple[str, str, str], float]] = []
    for node_with_score in ranked_nodes:
        original_page = node_with_score.node.metadata.get("original_page")
        if original_page:
            # Convert numpy float32 to Python float for JSON serialization
            score = (
                float(node_with_score.score)
                if node_with_score.score is not None
                else 0.0
            )
            ranked_pages.append((original_page, score))

    # Sort by score descending
    ranked_pages.sort(key=lambda x: x[1], reverse=True)

    return ranked_pages


# Type variable for generic filter function
T = TypeVar("T")


def filter_by_threshold(
    ranked_items: List[Tuple[T, float]],
    threshold: float,
) -> Tuple[List[Tuple[T, float]], List[Tuple[T, float]]]:
    """Split ranked items into passing and rejected based on threshold.

    Items with score >= threshold pass; items with score < threshold are rejected.
    Order is preserved in both output lists.

    Args:
        ranked_items: List of (item, score) tuples
        threshold: Minimum score to pass (0.0-1.0)

    Returns:
        Tuple of (passing, rejected) lists, each containing (item, score) tuples
    """
    passing: List[Tuple[T, float]] = []
    rejected: List[Tuple[T, float]] = []

    for item, score in ranked_items:
        if score >= threshold:
            passing.append((item, score))
        else:
            rejected.append((item, score))

    return passing, rejected


def fit_sources_to_context(
    pages: List[Tuple[str, str, str]],  # (url, title, content)
    source_scores: Dict[str, float],
    context_window: int,
    input_context_pct: float = 0.6,
    max_source_context_pct: float = 0.15,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    """Fit sources to context window using fill-from-top strategy.

    Pages must be pre-sorted by relevance (highest first).
    Uses a greedy algorithm to maximize content from highest-ranked sources.

    Args:
        pages: List of (url, title, content) tuples, sorted by relevance
        source_scores: Dict of url -> relevance score for each source
        context_window: Total context window size in tokens
        input_context_pct: Percentage of context window for input (rest for output)
        max_source_context_pct: Max percentage of context per source

    Returns:
        Tuple of:
        - List of (url, title, truncated_content) that fit
        - Dict of url -> allocated_chars for each included source

    Strategy:
    1. Calculate total budget: context_window * input_context_pct * 4 (chars)
    2. Calculate per-source cap: context_window * max_source_context_pct * 4
    3. Greedily fill from top until budget exhausted
    4. Last source may be truncated to fit remaining budget
    """
    if not pages:
        return [], {}

    # Calculate budgets in characters (rough: 1 token ≈ 4 chars)
    total_budget = int(context_window * input_context_pct * 4)
    per_source_cap = int(context_window * max_source_context_pct * 4)

    fitted: List[Tuple[str, str, str]] = []
    allocations: Dict[str, int] = {}
    remaining_budget = total_budget

    for url, title, content in pages:
        if remaining_budget <= 0:
            break

        # Cap content at per-source maximum
        content_to_use = content[:per_source_cap]

        # If content fits fully within remaining budget
        if len(content_to_use) <= remaining_budget:
            fitted.append((url, title, content_to_use))
            allocations[url] = len(content_to_use)
            remaining_budget -= len(content_to_use)
        else:
            # Truncate to fit remaining budget
            truncated = content_to_use[:remaining_budget]
            fitted.append((url, title, truncated))
            allocations[url] = len(truncated)
            remaining_budget = 0

    return fitted, allocations


async def generate_no_sources_explanation(
    query: str,
    rejected_titles: List[Tuple[str, float]],  # (title, score)
    rejected_content: List[Tuple[str, float]],  # (title, score)
    title_threshold: float,
    content_threshold: float,
    model_name: str,
    ollama_url: str,
) -> AsyncGenerator[str, None]:
    """Generate LLM explanation when no sources pass thresholds.

    Streams tokens explaining:
    - What was searched
    - How many results were found
    - Why they were rejected (scores vs thresholds)
    - Suggestions for the user

    Args:
        query: Original search query
        rejected_titles: List of (title, score) for sources rejected at title stage
        rejected_content: List of (title, score) for sources rejected at content stage
        title_threshold: Minimum score threshold for titles
        content_threshold: Minimum score threshold for content
        model_name: LLM model name
        ollama_url: Ollama API URL

    Yields:
        str: Individual tokens from the LLM explanation
    """
    # Build context about what was rejected
    title_rejected_info = ""
    if rejected_titles:
        title_rejected_info = "\n".join(
            f'  - "{title}" (score: {score*100:.0f}%)'
            for title, score in rejected_titles[:5]  # Limit to top 5
        )

    content_rejected_info = ""
    if rejected_content:
        content_rejected_info = "\n".join(
            f'  - "{title}" (score: {score*100:.0f}%)'
            for title, score in rejected_content[:5]
        )

    total_rejected = len(rejected_titles) + len(rejected_content)

    prompt = f"""You are a helpful assistant explaining why \
a web search couldn't find relevant information.

Query: "{query}"

Search Results Summary:
- Total results found: {total_rejected}
- Title relevance threshold: {title_threshold*100:.0f}%
- Content relevance threshold: {content_threshold*100:.0f}%

Sources rejected at title/snippet stage (below {title_threshold*100:.0f}% relevance):
{title_rejected_info if title_rejected_info else "  (none)"}

Sources rejected at content stage (below {content_threshold*100:.0f}% relevance):
{content_rejected_info if content_rejected_info else "  (none)"}

Write a brief, helpful explanation (2-3 sentences) that:
1. Acknowledges the search was performed
2. Explains that no sources were relevant enough (without being too technical)
3. Suggests ways to improve the query (more specific terms, different phrasing, etc.)

Keep it concise and actionable. Don't apologize excessively."""

    try:
        thinking_enabled = check_thinking_support(model_name)

        llm = Ollama(
            model=model_name,
            base_url=ollama_url,
            request_timeout=60.0,
            temperature=0.3,
            thinking=thinking_enabled,
        )

        async for chunk in await llm.astream_complete(prompt):
            if chunk.delta:
                yield chunk.delta

    except Exception as e:
        logger.error(f"No-sources explanation generation failed: {e}")
        yield (
            f'I searched for "{query}" but couldn\'t find sources that met the '
            f"relevance threshold. Try rephrasing your query or using more specific terms."
        )


# =============================================================================
# Progress Dataclasses for Structured Callbacks
# =============================================================================


@dataclass
class SearchProgress:
    """Progress data for search phase."""

    phase: Literal["searching"] = "searching"
    query: str = ""
    hits: Optional[int] = None


@dataclass
class FetchProgress:
    """Progress data for fetch phase."""

    phase: Literal["fetching"] = "fetching"
    url: str = ""
    title: str = ""
    status: str = ""  # "fetching", "success", "failed", "skipped"
    error: Optional[str] = None
    pages_target: int = 0
    pages_fetched: int = 0
    pages_failed: int = 0


@dataclass
class SummarizeProgress:
    """Progress data for summarize phase."""

    phase: Literal["summarizing"] = "summarizing"
    model_name: str = ""


@dataclass
class WebSearchSource:
    """A single web search source result."""

    url: str
    title: str
    status: Literal["success", "failed", "skipped"]
    error: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None  # Full fetched content (for successful pages)
    content_chars: int = 0  # Character count of content passed to LLM
    relevance_score: Optional[float] = None  # Reranker score (0.0-1.0)


@dataclass
class WebSearchChunk:
    """A chunk from web search streaming, compatible with chat consumption.

    This dataclass mirrors RAGChunk's pattern to enable unified streaming:
    - agent_progress: For search/fetch phases (unique to web search agents)
    - status: Pipeline status like 'generating' (shared with RAG/LLM)
    - token: LLM output token for streaming (shared with RAG/LLM)
    - sources: Final sources at completion
    - is_complete: Whether this is the final chunk

    Attributes:
        agent_progress: Dict with agent phase info (searching, fetching, summarizing).
        status: Pipeline status indicator (e.g., 'generating').
        token: LLM output token delta.
        sources: List of WebSearchSource at completion.
        is_complete: Whether this is the final chunk with sources.
    """

    agent_progress: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    token: Optional[str] = None
    sources: Optional[List["WebSearchSource"]] = None
    is_complete: bool = False


@deprecated("Use SourceConverter.to_api_schema() with UnifiedSource instead.")
def web_source_to_source_node(source: WebSearchSource) -> Dict[str, Any]:
    """Convert WebSearchSource to SourceNode format for unified UI.

    This enables web search sources to be displayed using the same SourcesList
    component used for RAG sources, providing a consistent user experience.

    Args:
        source: WebSearchSource from web search pipeline

    Returns:
        Dict matching SourceNode schema with web-specific metadata
    """
    # Use full content if available (for successful fetches), else snippet
    text = source.content if source.content else (source.snippet or "")

    # Use relevance_score for display if available, else fallback to status-based score
    score = (
        source.relevance_score
        if source.relevance_score is not None
        else (1.0 if source.status == "success" else 0.0)
    )

    return {
        "text": text,
        "score": score,
        "metadata": {
            "source_url": source.url,
            "display_name": source.title,
            "doc_type": "web",
            "fetch_status": source.status,
            "fetch_error": source.error,
            "content_chars": source.content_chars,
        },
    }


# Type alias for progress callbacks
# Callbacks can accept strings (backward-compatible) or structured progress objects
ProgressCallback = Callable[..., None]

logger = logging.getLogger(__name__)

# Browser-like headers to bypass bot detection
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}


async def search_duckduckgo(
    query: str, max_results: int = 10, progress_callback=None
) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo and return top N results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of dicts with keys: 'url', 'title', 'snippet'
        Returns empty list on failure
    """
    logger.info(f"Searching DuckDuckGo for: {query}")
    if progress_callback:
        progress_callback(f"Searching DuckDuckGo for: {query}")

    try:
        # DuckDuckGo search with exponential backoff
        for attempt in range(3):
            try:
                ddgs = DDGS()
                # Force English region to avoid non-English results
                # Use safesearch='off' for better technical results
                # timelimit='y' prioritizes recent results (helpful for tech queries)
                results = list(
                    ddgs.text(
                        query,
                        region="us-en",
                        safesearch="moderate",
                        timelimit=None,
                        max_results=max_results,
                    )
                )

                # Format results
                formatted = []
                for r in results:
                    formatted.append(
                        {
                            "url": r.get("href", r.get("link", "")),
                            "title": r.get("title", "Untitled"),
                            "snippet": r.get("body", r.get("snippet", "")),
                        }
                    )

                logger.info(f"Found {len(formatted)} search results")
                if progress_callback:
                    progress_callback(f"Found {len(formatted)} search results")
                return formatted

            except Exception as e:
                if attempt < 2:
                    wait_time = 2**attempt  # 1s, 2s
                    logger.warning(
                        f"DuckDuckGo search failed (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

        # Should not reach here, but return empty list as fallback
        return []

    except Exception as e:
        logger.error(f"DuckDuckGo search failed after retries: {e}")
        return []


def clean_html_for_content(soup: BeautifulSoup | Tag) -> BeautifulSoup | Tag:
    """
    Aggressively clean HTML to extract main content.

    Removes scripts, styles, navigation, ads, and other noise.
    Based on doxygen.py:clean_doxygen_html pattern.

    Args:
        soup: BeautifulSoup object

    Returns:
        Cleaned BeautifulSoup object
    """
    # 1. Remove scripts, styles, and visual-only elements
    for tag in soup.find_all(
        ["script", "style", "iframe", "img", "svg", "noscript", "link", "meta"]
    ):
        tag.decompose()

    # 2. Remove navigation and UI elements
    for tag in soup.find_all(["nav", "header", "footer", "aside", "form", "button"]):
        tag.decompose()

    # 3. Remove common noise classes (ads, social, cookies, modals)
    noise_classes = [
        "advertisement",
        "ad",
        "ads",
        "social",
        "share",
        "cookie",
        "modal",
        "popup",
        "sidebar",
        "navigation",
        "nav",
        "menu",
        "footer",
        "header",
        "comment",
        "comments",
        "related",
        "recommendations",
    ]

    for cls in noise_classes:
        for tag in soup.find_all(class_=lambda x: x and cls in x.lower()):
            tag.decompose()

    # 4. Remove by common noise IDs
    noise_ids = [
        "sidebar",
        "footer",
        "header",
        "nav",
        "menu",
        "comments",
        "cookie",
        "modal",
    ]

    for noise_id in noise_ids:
        for tag in soup.find_all(id=lambda x: x and noise_id in x.lower()):
            tag.decompose()

    # 5. Remove empty paragraphs and divs
    for tag in soup.find_all(["p", "div", "span"]):
        if not tag.get_text(strip=True):
            tag.decompose()

    return soup


async def fetch_generic_html(
    url: str, session: aiohttp.ClientSession, timeout: int = 10
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Fetch a generic HTML page and convert to markdown (fallback handler).

    This is the default handler when no domain-specific handler matches.

    Args:
        url: URL to fetch
        session: aiohttp ClientSession
        timeout: Timeout in seconds

    Returns:
        Tuple of (markdown_content, status, error_message)
        - markdown_content: Markdown string or None on failure
        - status: "success", "http_error", "timeout", "parse_error", "too_short"
        - error_message: Human-readable error description or None
    """
    logger.info(f"Fetching generic HTML: {url}")

    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with session.get(
            url, headers=BROWSER_HEADERS, timeout=timeout_obj
        ) as response:
            if response.status != 200:
                error_msg = f"HTTP {response.status}"
                logger.warning(f"{error_msg} for {url}")
                return None, "http_error", error_msg

            html = await response.text()

    except asyncio.TimeoutError:
        error_msg = "Timeout"
        logger.warning(f"{error_msg} fetching {url}")
        return None, "timeout", error_msg
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Error fetching {url}: {e}")
        return None, "network_error", error_msg

    try:
        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Extract main content (try multiple selectors)
        content = None
        for selector in ["main", "article", '[role="main"]', ".content", "#content"]:
            content = soup.select_one(selector)
            if content:
                break

        # Fallback to body if no main content found
        if not content:
            content = soup.find("body")

        if not content:
            error_msg = "No content found"
            logger.warning(f"{error_msg} in {url}")
            return None, "parse_error", error_msg

        # Clean HTML
        content = clean_html_for_content(content)

        # Convert to markdown
        markdown = md(str(content), heading_style="ATX", code_language="python")

        # Add source URL as metadata
        markdown = f"<!-- Source: {url} -->\n\n{markdown}"

        # Basic quality check
        if len(markdown.strip()) < 100:
            error_msg = "Content too short"
            logger.warning(f"{error_msg} for {url}")
            return None, "too_short", error_msg

        logger.info(f"✅ Fetched {url} ({len(markdown)} chars)")
        return markdown, "success", None

    except Exception as e:
        error_msg = f"Parse error: {str(e)}"
        logger.warning(f"Error processing {url}: {e}")
        return None, "parse_error", error_msg


async def fetch_page_as_markdown(
    url: str, session: aiohttp.ClientSession, timeout: int = 10
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Fetch a web page and convert to clean markdown.

    Uses domain-specific handlers for special sites (Wikipedia, GitHub, etc.)
    and falls back to generic HTML scraping for other sites.

    Args:
        url: URL to fetch
        session: aiohttp ClientSession
        timeout: Timeout in seconds

    Returns:
        Tuple of (markdown_content, status, error_message)
        - markdown_content: Markdown string or None on failure
        - status: "success", "http_error", "timeout", "parse_error", "too_short"
        - error_message: Human-readable error description or None
    """
    logger.info(f"Fetching: {url}")

    # Check for domain-specific handler
    handler = get_handler_for_url(url)
    if handler:
        logger.info(f"Using {handler.name} handler for {url}")
        return await handler.fetch(url, session, timeout)

    # Fallback to generic HTML scraping
    return await fetch_generic_html(url, session, timeout)


async def fetch_pages_parallel(
    results: List[Dict[str, str]], max_pages: int = 5, progress_callback=None
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str, str | None]]]:
    """
    Fetch multiple pages with "look forward" strategy.

    Tries to fetch max_pages successful pages by continuing through the
    search results list when some pages fail. Returns both successful
    pages and all attempted fetches with their status.

    Args:
        results: List of search results with 'url' and 'title'
        max_pages: Maximum number of successful pages to fetch

    Returns:
        Tuple of:
        - successful: List of (url, title, markdown_content)
        - all_attempts: List of (url, title, status, error_msg)
    """
    if not results:
        return [], []

    logger.info(f"Fetching up to {max_pages} pages with look-forward strategy...")
    if progress_callback:
        progress_callback(f"Fetching up to {max_pages} pages")

    successful: list[tuple[str, str, str]] = []
    all_attempts: list[tuple[str, str, str, str | None]] = []
    current_idx = 0

    async with aiohttp.ClientSession() as session:
        # Keep trying until we get max_pages successes or run out of results
        while len(successful) < max_pages and current_idx < len(results):
            # Calculate how many more we need
            needed = max_pages - len(successful)
            # Calculate how many to try in this batch (up to remaining results)
            batch_size = min(
                needed + 2, len(results) - current_idx
            )  # +2 for redundancy

            batch = results[current_idx : current_idx + batch_size]
            logger.info(
                f"Trying batch of {len(batch)} pages (need {needed} more successes)..."
            )
            if progress_callback:
                progress_callback(f"Trying {len(batch)} pages, need {needed} more")

            # Fetch batch in parallel
            tasks = [fetch_page_as_markdown(r["url"], session) for r in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for search_result, result in zip(batch, batch_results):
                url = search_result["url"]
                title = search_result["title"]

                if isinstance(result, BaseException):
                    exception_msg = str(result)
                    all_attempts.append((url, title, "exception", exception_msg))
                    logger.warning(f"Exception fetching {url}: {result}")
                    if progress_callback:
                        progress_callback(f"Failed: {title} - {exception_msg}")
                else:
                    # Tuple unpacking from Union[Tuple, Exception] (mypy false positive)
                    markdown, status, error_msg = result  # type: ignore[assignment]

                    if status == "success" and markdown:
                        successful.append((url, title, markdown))
                        all_attempts.append((url, title, "success", None))
                        if progress_callback:
                            progress_callback(
                                f"Fetched: {title} ({len(successful)}/{max_pages})"
                            )
                    else:
                        all_attempts.append((url, title, status, error_msg))
                        if progress_callback:
                            progress_callback(f"Skipped: {title} - {error_msg}")

                # Stop if we have enough successful pages
                if len(successful) >= max_pages:
                    break

            current_idx += batch_size

    logger.info(
        f"Successfully fetched {len(successful)} pages after {len(all_attempts)} attempts"
    )
    return successful, all_attempts


async def get_model_context_window(
    model_name: str, ollama_url: str, default: int = 16384
) -> int:
    """
    Get the context window size for the specified Ollama model.

    Args:
        model_name: Name of the Ollama model
        ollama_url: Ollama API base URL
        default: Default context window if retrieval fails

    Returns:
        Context window size in tokens
    """
    try:
        # Query Ollama API for model info
        timeout_obj = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ollama_url}/api/show",
                json={"name": model_name},
                timeout=timeout_obj,
            ) as response:
                if response.status == 200:
                    model_info = await response.json()
                    # Try to get num_ctx from model parameters
                    if "parameters" in model_info:
                        params_str = model_info.get("parameters", "")
                        # Parse parameters string for num_ctx
                        for line in params_str.split("\n"):
                            if "num_ctx" in line.lower():
                                try:
                                    # Extract number from line like "num_ctx 8192"
                                    parts = line.split()
                                    for i, part in enumerate(parts):
                                        if "num_ctx" in part.lower() and i + 1 < len(
                                            parts
                                        ):
                                            ctx_size = int(parts[i + 1])
                                            logger.info(
                                                f"Model {model_name} context window: {ctx_size}"
                                            )
                                            return ctx_size
                                except (ValueError, IndexError) as e:
                                    logger.debug(
                                        f"Failed to parse num_ctx from line '{line}': {e}"
                                    )
                                    continue

                    # Try modelfile for num_ctx
                    if "modelfile" in model_info:
                        modelfile = model_info.get("modelfile", "")
                        for line in modelfile.split("\n"):
                            if "num_ctx" in line.lower():
                                try:
                                    parts = line.split()
                                    for i, part in enumerate(parts):
                                        if part.isdigit():
                                            ctx_size = int(part)
                                            logger.info(
                                                f"Model {model_name} context window: {ctx_size}"
                                            )
                                            return ctx_size
                                except ValueError as e:
                                    logger.debug(
                                        f"Failed to parse num_ctx from modelfile line '{line}': {e}"
                                    )
                                    continue

        logger.info(
            f"Could not determine context window for {model_name}, using default: {default}"
        )
        return default

    except Exception as e:
        logger.warning(
            f"Failed to get model context window: {e}, using default: {default}"
        )
        return default


async def summarize_with_llm(
    query: str,
    pages: List[Tuple[str, str, str]],
    model_name: str,
    ollama_url: str,
    progress_callback=None,
    context_window: Optional[int] = None,
    custom_instructions: Optional[str] = None,
) -> str:
    """
    Use Ollama LLM to summarize web search findings.

    Args:
        query: Original search query
        pages: List of (url, title, markdown_content) tuples
        model_name: Ollama model name (from session)
        ollama_url: Ollama API URL

    Returns:
        Formatted markdown summary
    """
    logger.info(f"Summarizing {len(pages)} pages with {model_name}...")
    if progress_callback:
        progress_callback(f"Generating summary with {model_name}")

    if not pages:
        return "**No pages could be fetched.** Please try a different query."

    # Get model's actual context window (use provided or fetch from model)
    if context_window is None:
        context_window = await get_model_context_window(
            model_name, ollama_url, default=16384
        )
    # If provided but still using default, warn
    elif context_window == 8192:
        logger.warning(
            f"Using context_window={context_window}. "
            f"Consider setting higher for web search (e.g., 16384)"
        )

    # Calculate max chars based on context window
    # Use ~60% of context for input (leaving room for prompt structure and output)
    # Rough estimate: 1 token ≈ 4 chars
    max_input_tokens = int(context_window * 0.6)
    max_total_chars = max_input_tokens * 4
    max_per_page = min(
        max_total_chars // len(pages), 4000
    )  # Distribute across pages, cap at 4k per page

    logger.info(
        (
            f"Using context window: {context_window}, "
            f"max_total_chars: {max_total_chars}, "
            f"max_per_page: {max_per_page}"
        )
    )

    # Build source list with numbered markdown links
    sources_list = []
    combined = []

    for idx, (url, title, content) in enumerate(pages, 1):
        # Add to sources list as markdown link
        sources_list.append(f"{idx}. [{title}]({url})")

        # Truncate individual pages based on dynamic limit
        truncated = content[:max_per_page]
        if len(content) > max_per_page:
            truncated += "\n\n[Content truncated...]"

        # Format with source number and markdown link
        combined.append(f"### Source {idx}: [{title}]({url})\n\n{truncated}\n\n---\n")

    sources_text = "\n".join(sources_list)
    combined_text = "\n".join(combined)

    # Truncate total to prevent token overflow
    if len(combined_text) > max_total_chars:
        combined_text = combined_text[:max_total_chars]
        combined_text += "\n\n[Additional content truncated for length...]"

    # Calculate appropriate response length based on content
    # Use ~40% of context window for output
    max_output_tokens = int(context_window * 0.4)
    target_words = max_output_tokens  # Rough estimate: 1 token ≈ 1 word

    # Build prompt with dynamic length guidance
    custom_instruction_text = ""
    if custom_instructions:
        custom_instruction_text = (
            f"\n\n**Additional Instructions:** {custom_instructions}"
        )

    prompt = f"""You are a research assistant. User asked: "{query}"

## Available Sources
{sources_text}

## Content from Sources
{combined_text}

CRITICAL CITATION RULES - READ CAREFULLY:
1. **ALWAYS cite using markdown hyperlinks**, NEVER use plain [1] or [2] style
2. **Correct citation format**: "According to [Source Title](url), the key point is..."
3. **Example**: "The [YOLO algorithm](https://pjreddie.com/darknet/yolo/) \
detects objects..."
4. **WRONG**: "According to [1], the algorithm..." (incorrect)
5. **RIGHT**: "According to [YOLO Documentation]\
(https://pjreddie.com/darknet/yolo/), the algorithm..." (correct)
6. **Preserve ALL existing hyperlinks** from the source content
7. **Link technical terms** to their definitions when sources provide them

Provide a comprehensive summary ({target_words} words approx.) \
answering the question.{custom_instruction_text}

### Summary
[Thorough overview with inline hyperlinked citations. Example: \
"[Source 1]({pages[0][0]}) explains that..." or \
"The main approach involves [technique name]({pages[0][0]})..."]

### Detailed Findings
[Organize by topic. ALWAYS use hyperlinked citations like "[title](url)" - never bare [1] numbers. \
Include relevant details and preserve all markdown links from sources.]

### Key Takeaways
[Important points with inline hyperlinked sources where appropriate]

Begin your response:"""

    try:
        thinking_enabled = check_thinking_support(model_name)

        # Create Ollama LLM (reuses already-loaded model in VRAM)
        llm = Ollama(
            model=model_name,
            base_url=ollama_url,
            request_timeout=120.0,
            temperature=0.3,  # Low temperature for factual summarization
            context_window=context_window,
            num_ctx=context_window,  # Ollama-specific parameter
            thinking=thinking_enabled,
        )

        # Generate summary
        response = await llm.acomplete(prompt)
        summary = response.text.strip()

        logger.info("Summary generated successfully")
        return summary

    except Exception as e:
        logger.error(f"LLM summarization failed: {e}")
        return f"**Summarization failed:** {str(e)}\n\nPlease try again."


async def web_search_stream(
    query: str,
    model_name: str,
    ollama_url: str,
    max_results: int = 10,
    max_pages: int = 5,
    context_window: Optional[int] = None,
    custom_instructions: Optional[str] = None,
    reranker_model: Optional[str] = None,
    reranker_device: str = "cuda",
    # Threshold and context fitting params (from config)
    rerank_title_threshold: float = 0.1,
    rerank_content_threshold: float = 0.1,
    max_source_context_pct: float = 0.15,
    input_context_pct: float = 0.6,
) -> AsyncGenerator[WebSearchChunk, None]:
    """
    Streaming web search that yields chunks like RAG does.

    This generator yields WebSearchChunk objects for each phase:
    1. Search phase: agent_progress with search status
    2. Ranking phase (optional): agent_progress when reranking results
       - Applies title threshold to reject low-relevance results
    3. Fetch phase: agent_progress per page attempt (in ranked order if reranker enabled)
    4. Ranking phase (optional): agent_progress when reranking fetched content
       - Applies content threshold to reject low-relevance pages
    5. Context fitting: Trims sources to fit context window (fill-from-top)
    6. Zero-source handling: LLM explains why no sources passed (if applicable)
    7. Summarize phase: status='generating' then token chunks
    8. Complete: sources and is_complete=True

    Args:
        query: Search query
        model_name: Ollama model name
        ollama_url: Ollama API base URL
        max_results: Max search results to fetch from DDG
        max_pages: Max pages to download and process
        context_window: Optional context window size
        custom_instructions: Optional custom instructions for LLM
        reranker_model: Optional reranker model name (None = disabled)
        reranker_device: Device for reranker ("cuda", "cpu", "mps")
        rerank_title_threshold: Min score after title/snippet reranking (0.0-1.0)
        rerank_content_threshold: Min score after content reranking (0.0-1.0)
        max_source_context_pct: Max % of context per source
        input_context_pct: % of context window for input (rest for output)

    Yields:
        WebSearchChunk: Streaming chunks with progress, tokens, or sources
    """
    # Phase 1: Search - yield agent_progress
    yield WebSearchChunk(
        agent_progress={
            "agent": "web_search",
            "phase": "searching",
            "message": f"Searching for: {query}",
            "search_query": query,
            "search_hits": None,
        }
    )

    search_results = await search_duckduckgo(query, max_results)

    yield WebSearchChunk(
        agent_progress={
            "agent": "web_search",
            "phase": "searching",
            "message": f"Found {len(search_results)} results",
            "search_query": query,
            "search_hits": len(search_results),
        }
    )

    if not search_results:
        # No results - yield error message as token and complete
        error_msg = (
            f'## Web Search: "{query}"\n\n'
            f"**No results found.**\n\n"
            f"This could be due to:\n"
            f"- Network connectivity issues\n"
            f"- DuckDuckGo rate limiting\n"
            f"- Very specific or unusual query\n\n"
            f"Please try rephrasing your query or try again later."
        )
        yield WebSearchChunk(status="generating")
        yield WebSearchChunk(token=error_msg)
        yield WebSearchChunk(sources=[], is_complete=True)
        return

    # Phase 2: Rerank search results (if reranker enabled)
    reranker = None
    rejected_at_title: List[Tuple[str, float]] = (
        []
    )  # (title, score) for no-source explanation
    if reranker_model:
        # Check if model needs loading (first use detection)
        from tensortruth.services.model_manager import ModelManager

        manager = ModelManager.get_instance()
        model_needs_loading = not manager.is_reranker_loaded(reranker_model)

        if model_needs_loading:
            yield WebSearchChunk(
                agent_progress={
                    "agent": "web_search",
                    "phase": "loading_model",
                    "message": "Loading reranker model (first use may take 30-60s)...",
                    "model_name": reranker_model,
                }
            )

        yield WebSearchChunk(
            agent_progress={
                "agent": "web_search",
                "phase": "ranking_titles",
                "message": "Ranking search results by title...",
            }
        )
        reranker = get_reranker_for_web(reranker_model, reranker_device)
        ranked_results = rerank_search_results(
            query, search_results, max_results, reranker
        )

        # Apply title threshold filtering
        passing_results, rejected_results = filter_by_threshold(
            ranked_results, rerank_title_threshold
        )

        # Track rejected for potential no-source explanation
        rejected_at_title = [
            (r.get("title", "Untitled"), score) for r, score in rejected_results
        ]

        if not passing_results:
            # All results rejected at title stage - will handle after fetch phase
            search_results_ordered = []
            yield WebSearchChunk(
                agent_progress={
                    "agent": "web_search",
                    "phase": "ranking_titles",
                    "message": f"All {len(rejected_results)} results below relevance threshold",
                }
            )
        else:
            # Extract just the results in ranked order
            search_results_ordered = [r for r, _ in passing_results]
            yield WebSearchChunk(
                agent_progress={
                    "agent": "web_search",
                    "phase": "ranking_titles",
                    "message": (
                        f"{len(passing_results)} results passed threshold "
                        f"({len(rejected_results)} rejected)"
                    ),
                }
            )
    else:
        search_results_ordered = search_results

    # Get context window for pipeline
    if context_window is None:
        context_window = await get_model_context_window(
            model_name, ollama_url, default=16384
        )

    # Phase 3-6: Fetch, rerank, and fit pages using unified pipeline
    from tensortruth.core.source_pipeline import SourceFetchPipeline

    # Collect progress chunks to yield after pipeline execution
    progress_chunks: List[WebSearchChunk] = []

    # Define allowed detail keys for progress data
    ALLOWED_DETAIL_KEYS = {
        "pages_target",
        "pages_fetched",
        "pages_failed",
        "page_count",
        "passed",
        "rejected",
        "fitted",
        "total",
        "model",
    }

    def collect_progress(phase: str, message: str, details: dict):
        """Collect pipeline progress as WebSearchChunk."""
        progress_data: Dict[str, Any] = {
            "agent": "web_search",
            "phase": phase,
            "message": message,
        }
        # Add relevant details
        for key, value in details.items():
            if key in ALLOWED_DETAIL_KEYS:
                progress_data[key] = value
        progress_chunks.append(WebSearchChunk(agent_progress=progress_data))

    # Create pipeline instance with progress callback
    pipeline = SourceFetchPipeline(
        query=query,
        max_pages=max_pages,
        context_window=context_window,
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        rerank_content_threshold=rerank_content_threshold,
        max_source_context_pct=max_source_context_pct,
        input_context_pct=input_context_pct,
        custom_instructions=custom_instructions,
        progress_callback=collect_progress,
    )

    # Execute pipeline
    try:
        fitted_pages, source_nodes, allocations = await pipeline.execute(
            search_results_ordered
        )

        # Yield all collected progress chunks
        for chunk in progress_chunks:
            yield chunk
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        yield WebSearchChunk(
            agent_progress={
                "agent": "web_search",
                "phase": "error",
                "message": f"Pipeline failed: {str(e)}",
            }
        )
        yield WebSearchChunk(sources=[], is_complete=True)
        return

    # Convert SourceNode to WebSearchSource for compatibility
    sources: List[WebSearchSource] = []
    for node in source_nodes:
        sources.append(
            WebSearchSource(
                url=node.url,
                title=node.title,
                status=node.status,
                error=node.error,
                snippet=node.snippet,
                content=node.content,
                content_chars=node.content_chars,
                relevance_score=node.relevance_score,
            )
        )

    # Handle no pages case
    if not fitted_pages:
        # Check if we had any fetches that failed
        rejected_at_content = [
            (s.title, s.relevance_score or 0.0)
            for s in sources
            if s.status == "skipped" and s.relevance_score is not None
        ]

        if not sources:
            # No pages fetched at all - show snippets
            snippets = "\n\n".join(
                [
                    f"**{i+1}. [{r['title']}]({r['url']})**\n{r.get('snippet', '')}"
                    for i, r in enumerate(search_results[:max_pages])
                ]
            )
            fallback_msg = (
                f'## Web Search: "{query}"\n\n'
                f"**Found results but couldn't fetch full pages.**\n\n"
                f"Here are the search snippets:\n\n{snippets}\n\n"
                f"---\n*Page fetching failed. Try visiting the links directly.*"
            )
            yield WebSearchChunk(status="generating")
            yield WebSearchChunk(token=fallback_msg)
            yield WebSearchChunk(sources=sources, is_complete=True)
            return

        # Pages rejected after reranking - explain why
        yield WebSearchChunk(
            agent_progress={
                "agent": "web_search",
                "phase": "summarizing",
                "message": "Explaining search results...",
                "model_name": model_name,
            }
        )
        yield WebSearchChunk(status="generating")

        async for token in generate_no_sources_explanation(
            query=query,
            rejected_titles=rejected_at_title,
            rejected_content=rejected_at_content,
            title_threshold=rerank_title_threshold,
            content_threshold=rerank_content_threshold,
            model_name=model_name,
            ollama_url=ollama_url,
        ):
            yield WebSearchChunk(token=token)

        yield WebSearchChunk(sources=sources, is_complete=True)
        return

    # Use fitted pages for summarization
    pages = fitted_pages

    # Build source_scores dict from SourceNode results for synthesis config
    source_scores: Dict[str, float] = {}
    for node in source_nodes:
        if node.url and node.relevance_score is not None:
            source_scores[node.url] = node.relevance_score

    # Phase 7: Summarize - yield status then tokens
    yield WebSearchChunk(
        agent_progress={
            "agent": "web_search",
            "phase": "summarizing",
            "message": f"Generating summary with {model_name}",
            "model_name": model_name,
        }
    )
    yield WebSearchChunk(status="generating")

    # Convert tuple format to dict format for synthesis engine
    pages_dict = [
        {"url": url, "title": title, "content": content, "status": "success"}
        for url, title, content in pages
    ]

    # Debug: Log titles being passed to synthesis
    logger.info(f"Passing {len(pages_dict)} pages to synthesis:")
    for i, p in enumerate(pages_dict, 1):
        logger.info(f"  {i}. title='{p['title']}' url={p['url'][:50]}...")

    # Build synthesis config
    synthesis_config = SynthesisConfig(
        query=query,
        context_window=context_window,
        citation_style=CitationStyle.HYPERLINK,
        custom_instructions=custom_instructions,
        source_scores=source_scores if source_scores else None,
        input_context_pct=input_context_pct,
        max_source_context_pct=max_source_context_pct,
    )

    # Initialize LLM for synthesis
    thinking_enabled = check_thinking_support(model_name)
    llm = Ollama(
        model=model_name,
        base_url=ollama_url,
        request_timeout=120.0,
        temperature=0.3,
        context_window=context_window,
        num_ctx=context_window,
        thinking=thinking_enabled,
    )

    # Use core synthesis engine
    async for token in core_synthesize(llm, synthesis_config, pages_dict):
        yield WebSearchChunk(token=token)

    # Final: yield sources and complete
    yield WebSearchChunk(sources=sources, is_complete=True)


async def web_search_async(
    query: str,
    model_name: str,
    ollama_url: str,
    max_results: int = 10,
    max_pages: int = 5,
    progress_callback: Optional[ProgressCallback] = None,
    context_window: Optional[int] = None,
    custom_instructions: Optional[str] = None,
) -> Tuple[str, List[WebSearchSource]]:
    """
    Complete web search pipeline: search -> fetch -> summarize.

    Args:
        query: Search query
        model_name: Ollama model name
        ollama_url: Ollama API base URL
        max_results: Max search results to fetch (default: 10)
        max_pages: Max pages to download and process (default: 5)
        progress_callback: Optional callback for progress updates
        context_window: Optional context window size. If None, fetches from model.
        custom_instructions: Optional custom instructions for LLM summarization

    Returns:
        Tuple of (formatted markdown response, list of WebSearchSource)
    """
    # Step 1: Search DuckDuckGo
    search_results = await search_duckduckgo(query, max_results, progress_callback)

    if not search_results:
        return (
            f'## Web Search: "{query}"\n\n'
            f"**No results found.**\n\n"
            f"This could be due to:\n"
            f"- Network connectivity issues\n"
            f"- DuckDuckGo rate limiting\n"
            f"- Very specific or unusual query\n\n"
            f"Please try rephrasing your query or try again later.",
            [],
        )

    # Step 2: Fetch pages in parallel with look-forward strategy
    pages, all_attempts = await fetch_pages_parallel(
        search_results, max_pages, progress_callback
    )

    # Build sources list from all_attempts
    sources: List[WebSearchSource] = []
    # Create a mapping of url -> snippet from search results
    snippet_map = {r["url"]: r.get("snippet", "") for r in search_results}

    for url, title, status, error_msg in all_attempts:
        if status == "success":
            sources.append(
                WebSearchSource(
                    url=url,
                    title=title,
                    status="success",
                    error=None,
                    snippet=snippet_map.get(url),
                )
            )
        else:
            # Map various failure statuses to "failed" or "skipped"
            source_status: Literal["success", "failed", "skipped"] = (
                "skipped" if status in ("too_short", "parse_error") else "failed"
            )
            sources.append(
                WebSearchSource(
                    url=url,
                    title=title,
                    status=source_status,
                    error=error_msg,
                    snippet=snippet_map.get(url),
                )
            )

    if not pages:
        # Fallback: show search snippets if ALL page fetches fail
        snippets = "\n\n".join(
            [
                f"**{i+1}. [{r['title']}]({r['url']})**\n{r['snippet']}"
                for i, r in enumerate(search_results[:max_pages])
            ]
        )
        return (
            f'## Web Search: "{query}"\n\n'
            f"**Found results but couldn't fetch full pages.**\n\n"
            f"Here are the search snippets:\n\n{snippets}\n\n"
            f"---\n*Page fetching failed. Try visiting the links directly.*",
            sources,
        )

    # Step 3: LLM Summarization
    summary = await summarize_with_llm(
        query,
        pages,
        model_name,
        ollama_url,
        progress_callback,
        context_window,
        custom_instructions,
    )

    # Step 4: Format final response (sources are returned separately now)
    return f"{summary}\n\n", sources


def web_search(
    query: str,
    model_name: str,
    ollama_url: str,
    max_results: int = 10,
    max_pages: int = 5,
    progress_callback: Optional[ProgressCallback] = None,
    context_window: Optional[int] = None,
    custom_instructions: Optional[str] = None,
) -> Tuple[str, List[WebSearchSource]]:
    """
    Sync wrapper for web search (Streamlit compatible).

    Args:
        query: Search query
        model_name: Ollama model name
        ollama_url: Ollama API base URL
        max_results: Max search results to fetch (default: 10)
        max_pages: Max pages to download and process (default: 5)
        progress_callback: Optional callback for progress updates
        context_window: Optional context window size. If None, fetches from model.
        custom_instructions: Optional custom instructions for LLM summarization

    Returns:
        Tuple of (formatted markdown response, list of WebSearchSource)
    """
    return asyncio.run(
        web_search_async(
            query,
            model_name,
            ollama_url,
            max_results,
            max_pages,
            progress_callback,
            context_window,
            custom_instructions,
        )
    )
