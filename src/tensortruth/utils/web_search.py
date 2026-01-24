"""Web search utilities using DuckDuckGo with LLM-based summarization."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional, Tuple

import aiohttp
from bs4 import BeautifulSoup, Tag
from ddgs import DDGS
from llama_index.llms.ollama import Ollama
from markdownify import markdownify as md

from tensortruth.core.ollama import check_thinking_support

# Import handlers to register them (must be after this module is defined)
# These are imported at module load time to register handlers via decorators
from . import arxiv_handler  # noqa: F401, E402
from . import github_handler  # noqa: F401, E402
from . import wikipedia_handler  # noqa: F401, E402
from . import youtube_handler  # noqa: F401, E402
from .domain_handlers import get_handler_for_url  # noqa: E402

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


def web_source_to_source_node(source: WebSearchSource) -> Dict[str, any]:
    """Convert WebSearchSource to SourceNode format for unified UI.

    This enables web search sources to be displayed using the same SourcesList
    component used for RAG sources, providing a consistent user experience.

    Args:
        source: WebSearchSource from web search pipeline

    Returns:
        Dict matching SourceNode schema with web-specific metadata
    """
    return {
        "text": source.snippet or "",
        "score": 1.0 if source.status == "success" else 0.0,
        "metadata": {
            "source_url": source.url,
            "display_name": source.title,
            "doc_type": "web",
            "fetch_status": source.status,
            "fetch_error": source.error,
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
                    markdown, status, error_msg = result

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


async def summarize_with_llm_stream(
    query: str,
    pages: List[Tuple[str, str, str]],
    model_name: str,
    ollama_url: str,
    context_window: Optional[int] = None,
    custom_instructions: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from LLM summarization instead of blocking.

    Args:
        query: Original search query
        pages: List of (url, title, markdown_content) tuples
        model_name: Ollama model name
        ollama_url: Ollama API URL
        context_window: Optional context window size
        custom_instructions: Optional custom instructions for LLM

    Yields:
        str: Individual tokens from the LLM response
    """
    logger.info(f"Streaming summary of {len(pages)} pages with {model_name}...")

    if not pages:
        yield "**No pages could be fetched.** Please try a different query."
        return

    # Get model's actual context window (use provided or fetch from model)
    if context_window is None:
        context_window = await get_model_context_window(
            model_name, ollama_url, default=16384
        )

    # Calculate max chars based on context window
    max_input_tokens = int(context_window * 0.6)
    max_total_chars = max_input_tokens * 4
    max_per_page = min(max_total_chars // len(pages), 4000)

    # Build source list with numbered markdown links
    sources_list = []
    combined = []

    for idx, (url, title, content) in enumerate(pages, 1):
        sources_list.append(f"{idx}. [{title}]({url})")
        truncated = content[:max_per_page]
        if len(content) > max_per_page:
            truncated += "\n\n[Content truncated...]"
        combined.append(f"### Source {idx}: [{title}]({url})\n\n{truncated}\n\n---\n")

    sources_text = "\n".join(sources_list)
    combined_text = "\n".join(combined)

    if len(combined_text) > max_total_chars:
        combined_text = combined_text[:max_total_chars]
        combined_text += "\n\n[Additional content truncated for length...]"

    max_output_tokens = int(context_window * 0.4)
    target_words = max_output_tokens

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

        llm = Ollama(
            model=model_name,
            base_url=ollama_url,
            request_timeout=120.0,
            temperature=0.3,
            context_window=context_window,
            num_ctx=context_window,
            thinking=thinking_enabled,
        )

        # Use astream_complete for streaming
        async for chunk in await llm.astream_complete(prompt):
            if chunk.delta:
                yield chunk.delta

        logger.info("Streaming summary completed successfully")

    except Exception as e:
        logger.error(f"LLM streaming summarization failed: {e}")
        yield f"**Summarization failed:** {str(e)}\n\nPlease try again."


async def web_search_stream(
    query: str,
    model_name: str,
    ollama_url: str,
    max_results: int = 10,
    max_pages: int = 5,
    context_window: Optional[int] = None,
    custom_instructions: Optional[str] = None,
) -> AsyncGenerator[WebSearchChunk, None]:
    """
    Streaming web search that yields chunks like RAG does.

    This generator yields WebSearchChunk objects for each phase:
    1. Search phase: agent_progress with search status
    2. Fetch phase: agent_progress per page attempt
    3. Summarize phase: status='generating' then token chunks
    4. Complete: sources and is_complete=True

    Args:
        query: Search query
        model_name: Ollama model name
        ollama_url: Ollama API base URL
        max_results: Max search results to fetch
        max_pages: Max pages to download and process
        context_window: Optional context window size
        custom_instructions: Optional custom instructions for LLM

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

    # Phase 2: Fetch pages - yield agent_progress per attempt
    snippet_map = {r["url"]: r.get("snippet", "") for r in search_results}
    pages: List[Tuple[str, str, str]] = []
    sources: List[WebSearchSource] = []
    current_idx = 0
    pages_fetched = 0
    pages_failed = 0

    async with aiohttp.ClientSession() as session:
        while len(pages) < max_pages and current_idx < len(search_results):
            needed = max_pages - len(pages)
            batch_size = min(needed + 2, len(search_results) - current_idx)
            batch = search_results[current_idx : current_idx + batch_size]

            # Fetch batch
            tasks = [fetch_page_as_markdown(r["url"], session) for r in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for search_result, result in zip(batch, batch_results):
                url = search_result["url"]
                title = search_result["title"]

                if isinstance(result, BaseException):
                    pages_failed += 1
                    sources.append(
                        WebSearchSource(
                            url=url,
                            title=title,
                            status="failed",
                            error=str(result),
                            snippet=snippet_map.get(url),
                        )
                    )
                    yield WebSearchChunk(
                        agent_progress={
                            "agent": "web_search",
                            "phase": "fetching",
                            "message": f"Failed: {title}",
                            "pages_target": max_pages,
                            "pages_fetched": pages_fetched,
                            "pages_failed": pages_failed,
                            "current_page": {
                                "url": url,
                                "title": title,
                                "status": "failed",
                                "error": str(result),
                            },
                        }
                    )
                else:
                    markdown, status, error_msg = result
                    if status == "success" and markdown:
                        pages.append((url, title, markdown))
                        pages_fetched += 1
                        sources.append(
                            WebSearchSource(
                                url=url,
                                title=title,
                                status="success",
                                error=None,
                                snippet=snippet_map.get(url),
                            )
                        )
                        yield WebSearchChunk(
                            agent_progress={
                                "agent": "web_search",
                                "phase": "fetching",
                                "message": f"Fetched: {title}",
                                "pages_target": max_pages,
                                "pages_fetched": pages_fetched,
                                "pages_failed": pages_failed,
                                "current_page": {
                                    "url": url,
                                    "title": title,
                                    "status": "success",
                                    "error": None,
                                },
                            }
                        )
                    else:
                        source_status: Literal["success", "failed", "skipped"] = (
                            "skipped"
                            if status in ("too_short", "parse_error")
                            else "failed"
                        )
                        pages_failed += 1
                        sources.append(
                            WebSearchSource(
                                url=url,
                                title=title,
                                status=source_status,
                                error=error_msg,
                                snippet=snippet_map.get(url),
                            )
                        )
                        yield WebSearchChunk(
                            agent_progress={
                                "agent": "web_search",
                                "phase": "fetching",
                                "message": f"{source_status.title()}: {title}",
                                "pages_target": max_pages,
                                "pages_fetched": pages_fetched,
                                "pages_failed": pages_failed,
                                "current_page": {
                                    "url": url,
                                    "title": title,
                                    "status": source_status,
                                    "error": error_msg,
                                },
                            }
                        )

                if len(pages) >= max_pages:
                    break

            current_idx += batch_size

    # Handle no pages fetched
    if not pages:
        snippets = "\n\n".join(
            [
                f"**{i+1}. [{r['title']}]({r['url']})**\n{r['snippet']}"
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

    # Phase 3: Summarize - yield status then tokens
    yield WebSearchChunk(
        agent_progress={
            "agent": "web_search",
            "phase": "summarizing",
            "message": f"Generating summary with {model_name}",
            "model_name": model_name,
        }
    )
    yield WebSearchChunk(status="generating")

    async for token in summarize_with_llm_stream(
        query,
        pages,
        model_name,
        ollama_url,
        context_window,
        custom_instructions,
    ):
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
