"""FunctionTool wrappers for the orchestrator's built-in tools.

Wraps RAGService retrieval, ToolService-managed tools (web_search, fetch_page,
fetch_pages_batch) as FunctionTool instances that emit ToolProgress during
execution. Each factory function returns a FunctionTool via the closure
pattern: dependencies and progress_emitter are captured at construction time,
not passed per-call.

MCP tools from ToolService.tools are already FunctionTool instances and are
passed through directly to the orchestrator's tool set — no wrapping needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from llama_index.core.tools import FunctionTool
from llama_index.tools.mcp import BasicMCPClient
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, Field

from tensortruth.agents.tool_output import extract_tool_text
from tensortruth.services.models import ToolProgress
from tensortruth.services.tool_service import ToolService

if TYPE_CHECKING:
    from tensortruth.core.source import SourceNode
    from tensortruth.services.mcp_server_service import MCPServerService
    from tensortruth.services.models import RAGRetrievalResult
    from tensortruth.services.rag_service import RAGService
    from tensortruth.services.tool_confirmation_service import ToolConfirmationService

logger = logging.getLogger(__name__)

# Type alias for the progress emitter callback.
# Accepts a ToolProgress and returns None. May be sync or async;
# async variants are awaited inside the tool wrappers.
ProgressEmitter = Callable[[ToolProgress], Any]

# Type alias for a callback that receives the raw RAGRetrievalResult.
# Used by the stream translator to extract proper sources and metrics
# without relying on text parsing of the tool output.
RAGResultCallback = Callable[["RAGRetrievalResult"], None]

# Type alias for a callback that receives web SourceNodes from fetch_pages_batch.
WebResultCallback = Callable[[List["SourceNode"]], None]

# Type alias for a callback that receives full tool output for the synthesizer.
# Args: (tool_name, full_output_text).
FullOutputCallback = Callable[[str, str], None]


# --- Pydantic schemas for FunctionTool parameter validation ---


class AddArxivPaperInput(BaseModel):
    """Input schema for add_arxiv_paper orchestrator tool."""

    arxiv_id: str = Field(
        description=(
            'An arXiv paper identifier. Accepts new format ("2301.12345"), '
            'old format ("hep-th/9901001"), or a full arXiv URL.'
        )
    )
    scope: str = Field(
        default="session",
        description=(
            'Where to add the paper: "session" (default) adds to the '
            'current chat session, "project" adds to the parent project.'
        ),
    )


class RAGQueryInput(BaseModel):
    """Input schema for rag_query orchestrator tool."""

    query: str = Field(
        description="A search query to look up in the indexed knowledge base."
    )


class WebSearchInput(BaseModel):
    """Input schema for web_search orchestrator tool."""

    query: str = Field(description="A search query string to look up on the web.")


class FetchPageInput(BaseModel):
    """Input schema for fetch_page orchestrator tool."""

    url: str = Field(description="The URL of the web page to fetch.")


class FetchPagesBatchInput(BaseModel):
    """Input schema for fetch_pages_batch orchestrator tool."""

    urls: List[str] = Field(
        description=(
            "List of URLs to fetch in parallel (recommended: 3-5 URLs). "
            "All pages are fetched simultaneously for efficiency."
        )
    )


# --- Helper ---


async def _emit(emitter: ProgressEmitter, progress: ToolProgress) -> None:
    """Call the progress emitter, awaiting if it returns a coroutine."""
    result = emitter(progress)
    if asyncio.iscoroutine(result):
        await result


async def request_confirmation(
    confirmation_service: "ToolConfirmationService",
    progress_emitter: ProgressEmitter,
    tool_name: str,
    action_type: str,
    title: str,
    summary: str,
    details: Dict[str, Any],
    session_id: str,
    timeout: float = 300.0,
) -> str:
    """Request user confirmation, block until resolved.

    Creates a confirmation request, emits a WebSocket event for the frontend
    to render an inline card, then blocks until the user approves/rejects
    or the timeout expires.

    Returns:
        "approved", "rejected", or "expired".
    """
    confirmation = confirmation_service.create_confirmation(
        tool_name=tool_name,
        action_type=action_type,
        title=title,
        summary=summary,
        details=details,
        session_id=session_id,
    )

    # Emit confirmation_request event for the frontend
    logger.info(
        "request_confirmation: emitting confirmation_request for %s",
        confirmation.confirmation_id,
    )
    await _emit(
        progress_emitter,
        ToolProgress(
            tool_id=tool_name,
            phase="confirmation_request",
            message=f"Awaiting user approval: {title}",
            metadata={
                "confirmation_id": confirmation.confirmation_id,
                "tool_name": tool_name,
                "action_type": action_type,
                "title": title,
                "summary": summary,
                "details": details,
            },
        ),
    )
    logger.info(
        "request_confirmation: _emit returned, now awaiting resolution for %s",
        confirmation.confirmation_id,
    )

    # Block until resolved
    return await confirmation_service.await_resolution(
        confirmation.confirmation_id, timeout=timeout
    )


# --- Factory functions ---


def create_rag_tool(
    rag_service: RAGService,
    progress_emitter: ProgressEmitter,
    session_params: Optional[Dict[str, object]] = None,
    session_messages: Optional[List[Dict[str, object]]] = None,
    rag_result_callback: Optional[RAGResultCallback] = None,
    full_output_callback: Optional[FullOutputCallback] = None,
) -> FunctionTool:
    """Create a rag_query FunctionTool for the orchestrator.

    The returned tool performs retrieval + reranking + confidence scoring
    against the loaded vector indexes, but does NOT invoke the LLM for
    synthesis. The orchestrator synthesizes the final response itself.

    The tool function is a closure that captures the rag_service and
    progress_emitter at construction time. Session-level context
    (params, messages) is also captured so that query condensation can
    use chat history.

    Args:
        rag_service: RAGService instance with engine already loaded.
        progress_emitter: Callable that receives ToolProgress updates.
        session_params: Session engine parameters for retrieval config.
        session_messages: Chat history for query condensation.
        rag_result_callback: Optional callback to receive the raw
            RAGRetrievalResult for source extraction by the stream translator.

    Returns:
        A FunctionTool wrapping RAGService.retrieve().
    """
    # Capture mutable references so the tool always sees current state
    captured_params = session_params
    captured_messages = session_messages

    async def rag_query(query: str) -> str:
        """Search the indexed knowledge base for information relevant to the query.

        Returns source excerpts with relevance scores and confidence assessment
        for topics covered by the loaded document modules (papers, documentation,
        books, uploaded PDFs).
        """
        # Progress is emitted via RAGService.retrieve()'s progress_callback,
        # but we also emit a top-level phase for the orchestrator
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="rag",
                phase="retrieving",
                message=f'Searching knowledge base for "{query}"...',
                metadata={"query": query},
            ),
        )

        # Run synchronous retrieve() in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rag_service.retrieve(
                query=query,
                params=captured_params,
                session_messages=captured_messages,
                progress_callback=lambda tp: None,  # Suppress nested progress
            ),
        )

        # Forward raw result for source extraction by the stream translator
        if rag_result_callback is not None:
            rag_result_callback(result)

        # Full output for the synthesizer (via side-channel callback)
        full_output = _format_retrieval_result(result)
        if full_output_callback is not None:
            full_output_callback("rag_query", full_output)

        # Return compact summary for orchestrator scratchpad
        return _summarize_rag_result(result)

    return FunctionTool.from_defaults(
        async_fn=rag_query,
        name="rag_query",
        description=(
            "Search the indexed knowledge base (documents, papers, books, "
            "uploaded PDFs) for information relevant to a query. Returns "
            "source excerpts with relevance scores and a confidence level. "
            "Best for questions about topics covered by the loaded document "
            "modules. Not suitable for current events or live information."
        ),
        fn_schema=RAGQueryInput,
    )


def _format_retrieval_result(result: "RAGRetrievalResult") -> str:
    """Format a RAGRetrievalResult as a structured string for LLM consumption.

    Produces a clear text summary of sources with scores, confidence level,
    and key metrics. The orchestrator LLM reads this to synthesize its response.

    Args:
        result: RAGRetrievalResult from RAGService.retrieve().

    Returns:
        Formatted string with source summaries and metadata.
    """
    if not result.source_nodes:
        return (
            f"No relevant sources found in the knowledge base.\n"
            f"Confidence: {result.confidence_level}\n"
            f"Query used: {result.condensed_query}"
        )

    lines: List[str] = []
    lines.append(
        f"Found {result.num_sources} sources (confidence: {result.confidence_level})"
    )
    lines.append(f"Query used: {result.condensed_query}")
    lines.append("")

    for i, node in enumerate(result.source_nodes, 1):
        # Extract metadata
        inner_node = getattr(node, "node", node)
        metadata = {}
        if hasattr(inner_node, "metadata") and inner_node.metadata:
            metadata = inner_node.metadata
        elif hasattr(node, "metadata") and node.metadata:
            metadata = node.metadata

        # Extract display info
        title = (
            metadata.get("display_name")
            or metadata.get("title")
            or metadata.get("file_name")
            or "Untitled"
        )
        score = node.score if hasattr(node, "score") else None
        score_str = f"{score:.4f}" if score is not None else "N/A"

        # Extract text content (truncated for LLM context efficiency)
        if hasattr(inner_node, "get_content"):
            text = inner_node.get_content()
        elif hasattr(node, "text"):
            text = node.text
        else:
            text = str(node)

        # Truncate long content to keep tool output manageable
        max_chars = 1500
        if len(text) > max_chars:
            text = text[:max_chars] + "... [truncated]"

        lines.append(f"--- Source {i}: {title} (score: {score_str}) ---")
        lines.append(text)
        lines.append("")

    # Append key metrics summary
    if result.metrics:
        score_dist = result.metrics.get("score_distribution", {})
        quality = result.metrics.get("quality", {})
        coverage = result.metrics.get("coverage", {})

        metrics_parts = []
        if score_dist.get("mean") is not None:
            metrics_parts.append(f"avg_score={score_dist['mean']:.4f}")
        if score_dist.get("max") is not None:
            metrics_parts.append(f"max_score={score_dist['max']:.4f}")
        if quality.get("high_confidence_ratio") is not None:
            metrics_parts.append(
                f"high_conf_ratio={quality['high_confidence_ratio']:.2f}"
            )
        if coverage.get("total_chunks") is not None:
            metrics_parts.append(f"chunks={coverage['total_chunks']}")

        if metrics_parts:
            lines.append(f"Metrics: {', '.join(metrics_parts)}")

    return "\n".join(lines)


def _summarize_rag_result(result: "RAGRetrievalResult") -> str:
    """Produce a compact summary of a RAG result for the orchestrator scratchpad.

    Includes source count, confidence level, and titles+scores — but no chunk
    text. The full formatted output is sent separately via the callback.
    """
    if not result.source_nodes:
        return f"No relevant sources found. " f"Confidence: {result.confidence_level}"

    lines = [
        f"Found {result.num_sources} source(s) "
        f"(confidence: {result.confidence_level})"
    ]
    for i, node in enumerate(result.source_nodes, 1):
        inner_node = getattr(node, "node", node)
        metadata = {}
        if hasattr(inner_node, "metadata") and inner_node.metadata:
            metadata = inner_node.metadata
        elif hasattr(node, "metadata") and node.metadata:
            metadata = node.metadata
        title = (
            metadata.get("display_name")
            or metadata.get("title")
            or metadata.get("file_name")
            or "Untitled"
        )
        score = node.score if hasattr(node, "score") else None
        score_str = f"{score:.4f}" if score is not None else "N/A"
        lines.append(f"  {i}. {title} (score: {score_str})")
    return "\n".join(lines)


def _summarize_fetch_page(url: str, title: str, content_len: int) -> str:
    """Produce a compact summary of a fetched page for the orchestrator scratchpad."""
    return f'Fetched: "{title}" ({url}) — {content_len} chars'


def _summarize_fetch_pages_batch(
    fitted_pages: list,
    source_nodes: list,
    total: int,
) -> str:
    """Produce a compact summary of batch-fetched pages for the scratchpad.

    Lists titles, URLs, relevance scores, and char counts — no body content.
    """
    if not fitted_pages:
        return "No pages could be fetched or all were below relevance threshold."

    # Build score lookup from source_nodes
    score_by_url: Dict[str, float] = {}
    for sn in source_nodes:
        if sn.url and sn.score is not None:
            score_by_url[sn.url] = sn.score

    lines = [f"Fetched {len(fitted_pages)}/{total} pages:"]
    for page_url, title, content in fitted_pages:
        score = score_by_url.get(page_url)
        score_str = f", relevance: {score:.2f}" if score is not None else ""
        lines.append(f'  - "{title}" ({page_url}) — {len(content)} chars{score_str}')
    return "\n".join(lines)


def create_web_search_tool(
    tool_service: ToolService,
    progress_emitter: ProgressEmitter,
    reranker_model: Optional[str] = None,
    reranker_device: str = "cpu",
    title_threshold: float = 0.1,
    web_query_setter: Optional[Callable[[str], None]] = None,
    web_search_results_setter: Optional[Callable[[List[Dict]], None]] = None,
) -> FunctionTool:
    """Create a web_search FunctionTool for the orchestrator.

    The returned tool searches the web using a single query string, applies
    title-stage reranking when a reranker model is configured, and emits
    progress via the captured emitter.

    Args:
        tool_service: ToolService instance with loaded tools.
        progress_emitter: Callable that receives ToolProgress updates.
        reranker_model: Optional reranker model for title-stage reranking.
        reranker_device: Device for reranker model.
        title_threshold: Minimum score for title-stage reranking.
        web_query_setter: Callback to store the query for fetch_pages_batch.
        web_search_results_setter: Callback to store search results for
            fetch_pages_batch to access metadata.

    Returns:
        A FunctionTool that wraps ToolService's search_web tool.
    """

    async def web_search(query: str) -> str:
        """Search the web for information using a query string.

        Use this tool when the user asks about current events, recent
        developments, or topics not covered by the indexed knowledge base.
        Returns search results as JSON with titles, URLs, and snippets,
        ranked by relevance when a reranker model is available.
        """
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="web_search",
                phase="searching",
                message=f'Searching the web for "{query}"...',
                metadata={"query": query},
            ),
        )

        result = await tool_service.execute_tool("search_web", {"queries": query})

        if not result.get("success"):
            error_msg = result.get("error", "Unknown search error")
            logger.warning(f"web_search failed: {error_msg}")
            return f"Error: {error_msg}"

        raw_text = extract_tool_text(result["data"])

        # Parse search results for reranking
        search_results: List[Dict[str, Any]] = []
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                search_results = parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Title-stage reranking (reuses utils/web_search.py)
        if reranker_model and search_results:
            await _emit(
                progress_emitter,
                ToolProgress(
                    tool_id="web_search",
                    phase="ranking",
                    message="Ranking results by relevance...",
                    metadata={"count": len(search_results)},
                ),
            )

            from tensortruth.utils.web_search import (
                filter_by_threshold,
                get_reranker_for_web,
                rerank_search_results,
            )

            loop = asyncio.get_event_loop()
            reranker = await loop.run_in_executor(
                None,
                lambda: get_reranker_for_web(
                    reranker_model, reranker_device, top_n=len(search_results)
                ),
            )
            ranked = await loop.run_in_executor(
                None,
                lambda: rerank_search_results(
                    query, search_results, len(search_results), reranker
                ),
            )
            passing, rejected = filter_by_threshold(ranked, title_threshold)

            search_results = []
            for r, score in passing:
                r["relevance_score"] = round(score, 4)
                search_results.append(r)

            logger.info(
                "Title reranking: %d passed, %d rejected (threshold=%.2f)",
                len(passing),
                len(rejected),
                title_threshold,
            )

        # Store query and full results (with snippets) for fetch_pages_batch
        if web_query_setter:
            web_query_setter(query)
        if web_search_results_setter:
            web_search_results_setter(search_results)

        # Strip snippets — LLM sees only urls, titles, scores
        llm_results = []
        for r in search_results:
            entry = {"url": r.get("url", ""), "title": r.get("title", "")}
            if "relevance_score" in r:
                entry["relevance_score"] = r["relevance_score"]
            llm_results.append(entry)

        return json.dumps(llm_results, indent=2)

    return FunctionTool.from_defaults(
        async_fn=web_search,
        name="web_search",
        description=(
            "Search the web using DuckDuckGo for current information, recent "
            "events, or topics not in the indexed knowledge base. Returns a "
            "JSON array of results with titles, URLs, and relevance scores — "
            "NO page content or snippets. You MUST call fetch_pages_batch "
            "with the relevant URLs after this tool to retrieve actual content."
        ),
        fn_schema=WebSearchInput,
    )


def create_fetch_page_tool(
    tool_service: ToolService,
    progress_emitter: ProgressEmitter,
    fetched_urls_getter: Optional[Callable[[], set]] = None,
    fetched_urls_updater: Optional[Callable[[List[str]], None]] = None,
    full_output_callback: Optional[FullOutputCallback] = None,
) -> FunctionTool:
    """Create a fetch_page FunctionTool for the orchestrator.

    The returned tool fetches a single web page and converts it to markdown.

    Args:
        tool_service: ToolService instance with loaded tools.
        progress_emitter: Callable that receives ToolProgress updates.

    Returns:
        A FunctionTool that wraps ToolService's fetch_page tool.
    """

    async def fetch_page(url: str) -> str:
        """Fetch a single web page by URL and extract its content as markdown.

        Accepts one URL. Returns the full page content as clean markdown text,
        or an error message on failure. Supports Wikipedia, GitHub, arXiv,
        YouTube, and general websites.
        """
        domain = urlparse(url).netloc
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="fetch_page",
                phase="fetching",
                message=f"Fetching {domain}...",
                metadata={"url": url},
            ),
        )

        result = await tool_service.execute_tool("fetch_page", {"url": url})

        if not result.get("success"):
            error_msg = result.get("error", "Unknown fetch error")
            logger.warning(f"fetch_page failed for {url}: {error_msg}")
            return f"Error: {error_msg}"

        content = extract_tool_text(result["data"])

        # Track fetched URL
        if fetched_urls_updater:
            fetched_urls_updater([url])

        # Discover links in fetched content
        discovered_section = await _discover_links(
            [(url, content)],
            progress_emitter,
            fetched_urls_getter,
        )

        if discovered_section:
            content += "\n\n" + discovered_section

        # Full output for synthesizer (via side-channel callback)
        if full_output_callback is not None:
            full_output_callback("fetch_page", content)

        # Extract title: first heading or domain fallback
        title = urlparse(url).netloc
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

        return _summarize_fetch_page(url, title, len(content))

    return FunctionTool.from_defaults(
        async_fn=fetch_page,
        name="fetch_page",
        description=(
            "Fetch a single web page by URL and extract its content as clean "
            "markdown text. Accepts one URL. Returns the full page content as "
            "markdown, or an error message on failure. Supports Wikipedia, "
            "GitHub, arXiv, YouTube, and general websites."
        ),
        fn_schema=FetchPageInput,
    )


def create_fetch_pages_batch_tool(
    tool_service: ToolService,
    progress_emitter: ProgressEmitter,
    reranker_model: Optional[str] = None,
    reranker_device: str = "cpu",
    content_threshold: float = 0.1,
    context_window: int = 16384,
    custom_instructions: Optional[str] = None,
    web_query_getter: Optional[Callable[[], Optional[str]]] = None,
    web_search_results_getter: Optional[Callable[[], Optional[List[Dict]]]] = None,
    web_result_callback: Optional[WebResultCallback] = None,
    fetched_urls_getter: Optional[Callable[[], set]] = None,
    fetched_urls_updater: Optional[Callable[[List[str]], None]] = None,
    full_output_callback: Optional[FullOutputCallback] = None,
) -> FunctionTool:
    """Create a fetch_pages_batch FunctionTool for the orchestrator.

    The returned tool uses ``SourceFetchPipeline`` to fetch, content-rerank,
    filter, and fit pages to the context window. Results include relevance
    scores from the reranker.

    Args:
        tool_service: ToolService instance (unused — pipeline fetches directly).
        progress_emitter: Callable that receives ToolProgress updates.
        reranker_model: Optional reranker model for content-stage reranking.
        reranker_device: Device for reranker model.
        content_threshold: Minimum score for content-stage reranking.
        context_window: Context window size in tokens for fitting.
        custom_instructions: Optional custom instructions for reranking.
        web_query_getter: Getter for the stored query from web_search tool.
        web_search_results_getter: Getter for stored search results metadata.
        web_result_callback: Callback to emit SourceNode objects for the
            stream translator to use for source cards.

    Returns:
        A FunctionTool that fetches, reranks, and fits web pages.
    """

    def _build_search_results(
        urls: List[str],
        stored_results: Optional[List[Dict]],
    ) -> List[Dict]:
        """Build search result dicts from URLs + stored web_search metadata.

        Matches each URL to its stored search result for title/snippet.
        URLs not found in stored results get a bare entry.
        """
        stored_map: Dict[str, Dict] = {}
        if stored_results:
            for r in stored_results:
                stored_map[r.get("url", "")] = r

        results = []
        for url in urls:
            if url in stored_map:
                results.append(stored_map[url])
            else:
                domain = urlparse(url).netloc
                results.append({"url": url, "title": domain, "snippet": ""})
        return results

    async def fetch_pages_batch(urls: List[str]) -> str:
        """Fetch multiple web pages in parallel given a list of URLs.

        Pages are fetched simultaneously, ranked by content relevance to the
        current query, and fitted to the context window. Accepts a list of
        URLs (recommended 3-5). Returns full page content as formatted
        markdown sections ready for synthesis.
        """
        total = len(urls)
        domains = list(dict.fromkeys(urlparse(u).netloc for u in urls))
        preview = ", ".join(domains[:3])
        suffix = ", ..." if len(domains) > 3 else ""
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="fetch_pages_batch",
                phase="fetching",
                message=f"Fetching {total} pages ({preview}{suffix})...",
                metadata={"urls": urls, "total": total},
            ),
        )

        # Retrieve stored state from web_search tool
        stored_query = web_query_getter() if web_query_getter else None
        stored_results = (
            web_search_results_getter() if web_search_results_getter else None
        )

        # Build search_results dicts from URLs + stored metadata
        search_results = _build_search_results(urls, stored_results)

        # Use SourceFetchPipeline for fetch + content rerank + fit
        from tensortruth.core.source_pipeline import SourceFetchPipeline

        def _pipeline_progress(phase: str, message: str, details: dict) -> None:
            _emit_sync(
                progress_emitter,
                ToolProgress(
                    tool_id="fetch_pages_batch",
                    phase=phase,
                    message=message,
                    metadata=details,
                ),
            )

        pipeline = SourceFetchPipeline(
            query=stored_query or "",
            max_pages=len(urls),
            context_window=context_window,
            reranker_model=reranker_model,
            reranker_device=reranker_device,
            rerank_content_threshold=content_threshold,
            custom_instructions=custom_instructions,
            progress_callback=_pipeline_progress,
        )

        fitted_pages, source_nodes, allocations = await pipeline.execute(search_results)

        # Emit SourceNodes via callback for stream translator
        if web_result_callback and source_nodes:
            web_result_callback(source_nodes)

        # Track fetched URLs
        if fetched_urls_updater:
            fetched_urls_updater(urls)

        # Emit completion progress
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="fetch_pages_batch",
                phase="fetched",
                message=f"Fetched {len(fitted_pages)}/{total} pages.",
                metadata={
                    "total": total,
                    "fitted": len(fitted_pages),
                    "sources": len(source_nodes),
                },
            ),
        )

        # Format content for LLM consumption
        if not fitted_pages:
            return "No pages could be fetched or all were below relevance threshold."

        # Build score lookup from source_nodes
        score_by_url: Dict[str, float] = {}
        for sn in source_nodes:
            if sn.url and sn.score is not None:
                score_by_url[sn.url] = sn.score

        parts = []
        for page_url, title, content in fitted_pages:
            score = score_by_url.get(page_url)
            score_str = f" | Relevance: {score:.2f}" if score is not None else ""
            header = f"## {title}\nSource: {page_url}{score_str}\n"
            if len(content) < 500:
                header += f"\n*Note: Limited content ({len(content)} chars)*\n"
            parts.append(f"{header}\n{content}")
        output = "\n\n---\n\n".join(parts)

        # Discover links in fetched pages
        page_contents = [(page_url, content) for page_url, _, content in fitted_pages]
        discovered_section = await _discover_links(
            page_contents,
            progress_emitter,
            fetched_urls_getter,
        )

        if discovered_section:
            output += "\n\n" + discovered_section

        # Full output for synthesizer (via side-channel callback)
        if full_output_callback is not None:
            full_output_callback("fetch_pages_batch", output)

        # Return compact summary for orchestrator scratchpad
        return _summarize_fetch_pages_batch(fitted_pages, source_nodes, total)

    return FunctionTool.from_defaults(
        async_fn=fetch_pages_batch,
        name="fetch_pages_batch",
        description=(
            "Fetch multiple web pages in parallel given a list of URLs. Pages "
            "are fetched simultaneously, ranked by content relevance to the "
            "current query, and fitted to the context window. Accepts a list "
            "of URLs (recommended 3-5). Returns full page content as formatted "
            "markdown sections ready for synthesis."
        ),
        fn_schema=FetchPagesBatchInput,
    )


def _emit_sync(emitter: ProgressEmitter, progress: ToolProgress) -> None:
    """Call the progress emitter synchronously (for pipeline callbacks)."""
    try:
        result = emitter(progress)
        if asyncio.iscoroutine(result):
            # If called from async context, we can't await here.
            # The emitter is expected to be sync from pipeline callbacks.
            pass
    except Exception:
        pass


async def _discover_links(
    page_contents: List[tuple],  # [(url, content), ...]
    progress_emitter: ProgressEmitter,
    fetched_urls_getter: Optional[Callable[[], set]] = None,
) -> str:
    """Extract links from fetched pages and fetch metadata for top candidates.

    Returns a formatted "Discovered links" section string, or empty string
    if no links were found.
    """
    from tensortruth.utils.web_search import (
        extract_links_from_markdown,
        fetch_link_metadata,
    )

    exclude_urls = fetched_urls_getter() if fetched_urls_getter else set()

    # Collect links from all pages, tracking which page they came from
    all_links: List[tuple] = []  # (anchor_text, url, page_title_or_url)
    seen_urls: set = set()
    for page_url, content in page_contents:
        links = extract_links_from_markdown(content, page_url, exclude_urls)
        for anchor, link_url in links:
            if link_url not in seen_urls:
                seen_urls.add(link_url)
                # Use domain as source identifier
                source_domain = urlparse(page_url).netloc
                all_links.append((anchor, link_url, source_domain))

    if not all_links:
        return ""

    # Fetch metadata for top candidates
    await _emit(
        progress_emitter,
        ToolProgress(
            tool_id="fetch_pages_batch",
            phase="inspecting_links",
            message="Inspecting links from fetched pages...",
        ),
    )

    import aiohttp

    links_for_meta = [(anchor, url) for anchor, url, _ in all_links[:8]]
    source_map = {url: source for _, url, source in all_links[:8]}

    async with aiohttp.ClientSession() as session:
        metadata = await fetch_link_metadata(links_for_meta, session)

    # Format discovered links section
    lines = ["### Discovered links in fetched pages"]
    for i, meta in enumerate(metadata, 1):
        title = meta.get("title") or meta["anchor_text"]
        domain = urlparse(meta["url"]).netloc
        desc = meta.get("description", "")
        desc_str = f' — "{desc}"' if desc else " — No description available"
        source = source_map.get(meta["url"], "")
        lines.append(
            f'{i}. "{title}" ({domain}){desc_str}\n'
            f'   URL: {meta["url"]} | Found in: {source}'
        )

    lines.append(
        "\nIf the above content does not fully answer the question, "
        "you may call fetch_pages_batch with relevant URLs from this list."
    )

    return "\n".join(lines)


# --- Pydantic schemas for MCP management tools ---


class ListMCPServersInput(BaseModel):
    """Input schema for list_mcp_servers tool (no params required)."""

    pass


class GetMCPPresetsInput(BaseModel):
    """Input schema for get_mcp_presets tool (no params required)."""

    pass


class ProposeMCPServerInput(BaseModel):
    """Input schema for manage_mcp_server tool."""

    action: str = Field(description='Action to perform: "add", "update", or "remove".')
    name: str = Field(description="Server name to add, update, or remove.")
    summary: str = Field(
        description="Human-readable summary of what this change does, for the user."
    )
    type: str = Field(
        default="stdio",
        description='Server type: "stdio" or "sse". Defaults to "stdio".',
    )
    command: str = Field(
        default="",
        description=(
            "REQUIRED for stdio servers. The executable to run. "
            'E.g. "npx", "node", "python", "docker".'
        ),
    )
    args: List[str] = Field(
        default_factory=list,
        description=(
            'Command arguments. E.g. ["-y", "@modelcontextprotocol/server-name"]. '
            "Required alongside command for stdio servers."
        ),
    )
    url: str = Field(
        default="",
        description='REQUIRED for SSE servers. E.g. "http://localhost:3000/sse".',
    )
    description: str = Field(
        default="",
        description="Human-readable description of the server.",
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables required by the server.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the server should be enabled.",
    )


# --- MCP server verification ---


async def _verify_mcp_server(
    config: Dict[str, Any],
    progress_emitter: ProgressEmitter,
    timeout: int = 15,
) -> tuple[bool, str]:
    """Verify an MCP server by attempting to connect and list its tools.

    Spawns the server process (or connects via SSE), performs the MCP
    initialize handshake, lists available tools, then shuts down.

    Returns:
        (success, message) — on success message includes tool count,
        on failure it contains the error details.
    """
    server_type = config.get("type", "stdio")
    name = config.get("name", "unknown")

    await _emit(
        progress_emitter,
        ToolProgress(
            tool_id="manage_mcp_server",
            phase="verifying",
            message=f"Verifying MCP server '{name}'...",
            metadata={"name": name, "type": server_type},
        ),
    )

    # Capture stderr so we can surface the real error (e.g. missing env vars)
    # instead of the generic "Connection closed" from the MCP protocol layer.
    import tempfile

    stderr_file = tempfile.TemporaryFile()

    try:
        from tensortruth.agents.server_registry import _resolve_env

        if server_type == "stdio":
            resolved_env = _resolve_env(config.get("env"))
            server_params = StdioServerParameters(
                command=config["command"],
                args=config.get("args") or [],
                env=resolved_env,
            )

            async def _verify_stdio() -> tuple[bool, str]:
                async with stdio_client(server_params, errlog=stderr_file) as (
                    read_stream,
                    write_stream,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await session.list_tools()
                        tool_count = len(result.tools) if result.tools else 0
                        tool_names = (
                            [t.name for t in result.tools[:5]] if result.tools else []
                        )
                        preview = ", ".join(tool_names)
                        if tool_count > 5:
                            preview += f", ... (+{tool_count - 5} more)"
                        return (
                            True,
                            f"Server verified: {tool_count} tools available"
                            f" ({preview})",
                        )

            return await asyncio.wait_for(_verify_stdio(), timeout=timeout)

        elif server_type == "sse":
            client = BasicMCPClient(
                command_or_url=config["url"],
                timeout=timeout,
            )
            result = await asyncio.wait_for(client.list_tools(), timeout=timeout)
            tool_count = len(result.tools) if result and result.tools else 0
            tool_names = (
                [t.name for t in result.tools[:5]] if result and result.tools else []
            )
            preview = ", ".join(tool_names)
            if tool_count > 5:
                preview += f", ... (+{tool_count - 5} more)"
            return True, f"Server verified: {tool_count} tools available ({preview})"

        else:
            return False, f"Unknown server type '{server_type}'"

    except asyncio.TimeoutError:
        return False, (
            f"Server '{name}' timed out after {timeout}s. "
            "The command may be wrong, the package may not exist, "
            "or the server may take too long to start."
        )
    except FileNotFoundError as e:
        return False, (
            f"Command not found: {e}. "
            "Check that the command is installed and available on PATH."
        )
    except BaseException as e:
        # ExceptionGroups from asyncio TaskGroups wrap the real error;
        # unwrap recursively to surface the actual cause to the LLM.
        cause = e
        while hasattr(cause, "exceptions") and cause.exceptions:
            cause = cause.exceptions[0]
        logger.warning(
            "MCP server '%s' verification failed: %s", name, cause, exc_info=cause
        )
        # Prefer stderr output — it usually contains the real error message
        # (e.g. "TRELLO_API_KEY and TRELLO_TOKEN environment variables are
        # required") while the Python exception is just "Connection closed".
        stderr_output = ""
        try:
            stderr_file.seek(0)
            stderr_output = stderr_file.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        if stderr_output:
            # Extract last meaningful line from stderr
            lines = [ln.strip() for ln in stderr_output.splitlines() if ln.strip()]
            # Look for lines starting with "Error:" or the last non-empty line
            error_lines = [ln for ln in lines if ln.startswith("Error:")]
            error_msg = error_lines[-1] if error_lines else lines[-1]
        else:
            error_msg = str(cause)
        # Trim overly verbose output
        if len(error_msg) > 300:
            error_msg = error_msg[:300] + "..."
        return False, f"Server verification failed: {error_msg}"
    finally:
        stderr_file.close()


# --- MCP management tool factories ---


def create_list_mcp_servers_tool(
    mcp_server_service: "MCPServerService",
    progress_emitter: ProgressEmitter,
) -> FunctionTool:
    """Create a list_mcp_servers FunctionTool for the orchestrator.

    Read-only tool that returns the current MCP server configuration.
    """

    async def list_mcp_servers() -> str:
        """List all configured MCP servers with their status.

        Returns a JSON list of all MCP servers (built-in and user-configured)
        with their type, command, enabled state, and environment variable status.
        """
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="list_mcp_servers",
                phase="listing",
                message="Listing MCP server configurations...",
            ),
        )
        servers = mcp_server_service.list_all()
        return json.dumps(servers, indent=2)

    return FunctionTool.from_defaults(
        async_fn=list_mcp_servers,
        name="list_mcp_servers",
        description=(
            "List all configured MCP servers with their type, command/URL, "
            "enabled state, and environment variable status. Use this to see "
            "what MCP servers are currently set up."
        ),
        fn_schema=ListMCPServersInput,
    )


def create_get_mcp_presets_tool(
    mcp_server_service: "MCPServerService",
    progress_emitter: ProgressEmitter,
) -> FunctionTool:
    """Create a get_mcp_presets FunctionTool for the orchestrator.

    Read-only tool that returns available preset server templates.
    """

    async def get_mcp_presets() -> str:
        """Get available MCP server preset configurations.

        Returns preset templates for well-known MCP servers (e.g. context7,
        github, huggingface) that can be used with manage_mcp_server.
        """
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="get_mcp_presets",
                phase="listing",
                message="Fetching MCP server presets...",
            ),
        )
        presets = mcp_server_service.get_presets()
        return json.dumps(presets, indent=2)

    return FunctionTool.from_defaults(
        async_fn=get_mcp_presets,
        name="get_mcp_presets",
        description=(
            "Get available MCP server preset configurations for well-known "
            "servers like context7, github, and huggingface. Use this before "
            "proposing a server to check if a preset exists."
        ),
        fn_schema=GetMCPPresetsInput,
    )


def create_manage_mcp_server_tool(
    confirmation_service: "ToolConfirmationService",
    mcp_server_service: "MCPServerService",
    tool_service: ToolService,
    progress_emitter: ProgressEmitter,
    session_id: str,
) -> FunctionTool:
    """Create a manage_mcp_server FunctionTool for the orchestrator.

    Requests user confirmation, blocks until approved/rejected, then applies
    the change directly. The LLM sees the real result ("Successfully added...")
    instead of "Proposal created...".
    """

    # Track recent failed calls to prevent infinite retry loops.
    _last_error_key: Optional[str] = None

    async def manage_mcp_server(
        action: str,
        name: str,
        summary: str,
        type: str = "stdio",
        command: str = "",
        args: Optional[List[str]] = None,
        url: str = "",
        description: str = "",
        env: Optional[Dict[str, str]] = None,
        enabled: bool = True,
    ) -> str:
        """Propose adding, updating, or removing an MCP server configuration.

        This does NOT immediately apply the change. Instead, it creates a
        proposal that the user must approve or reject via an inline card
        in the chat. Do not proceed until the user has approved.

        For 'add': provide full server config (name, type, command/url, etc.).
        For 'update': provide the server name and fields to change.
        For 'remove': provide just the server name.
        """
        nonlocal _last_error_key

        # Coerce args from JSON string to list (LLMs sometimes stringify)
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, list):
                    args = parsed
            except (json.JSONDecodeError, ValueError):
                args = [args] if args.strip() else None
        if isinstance(enabled, str):
            enabled = enabled.lower() in ("true", "1", "yes")

        # Normalize empty sentinels to None for internal logic
        command = command if command else None
        args = args if args else None
        url = url if url else None
        description = description if description else None
        env = env if env else None

        # Prevent identical retry loops — refuse if same args as last failure
        call_key = f"{action}:{name}:{command}:{args}"
        if _last_error_key and call_key == _last_error_key:
            _last_error_key = (
                None  # Reset so a third attempt is allowed after adjustment
            )
            return (
                "Error: This is the same call that just failed. "
                "Do NOT retry with the same arguments. "
                "If command is missing, use web_search and fetch_page to find "
                "the correct command and args first, then try again with "
                "the actual values filled in."
            )

        # Validate action
        if action not in ("add", "update", "remove"):
            return f"Error: action must be 'add', 'update', or 'remove', got '{action}'"

        # Auto-fill from preset if name matches and key fields are missing
        if action == "add" and not command and not url:
            presets = mcp_server_service.get_presets()
            preset = presets.get(name)
            if preset:
                type = type or preset.get("type", "stdio")
                command = command or preset.get("command")
                args = args or preset.get("args")
                url = url or preset.get("url")
                description = description or preset.get("description")
                env = env or preset.get("env")
                if enabled is None or enabled is True:
                    enabled = preset.get("enabled", True)

        # Infer command from args when the LLM forgets to set it.
        # Common pattern: LLM passes args=["-y", "<pkg>"] but command=null.
        if action in ("add", "update") and not command and args:
            # args with "-y" flag strongly imply npx
            if "-y" in args or "--yes" in args:
                command = "npx"
            # args starting with a scoped package (@org/pkg) or containing
            # a path-like string suggest npx as well
            elif args and (args[0].startswith("@") or "/" in args[0]):
                command = "npx"
                args = ["-y"] + list(args)

        # Build config dict
        config: Dict[str, Any] = {"name": name}
        if action in ("add", "update"):
            if type:
                config["type"] = type
            if command:
                config["command"] = command
            if args:
                config["args"] = args
            if url:
                config["url"] = url
            if description:
                config["description"] = description
            if env:
                config["env"] = env
            if enabled is not None:
                config["enabled"] = enabled

        # Validate: for add, check required fields
        if action == "add":
            if type == "stdio" and not command:
                _last_error_key = call_key
                return (
                    "Error: 'command' is required for stdio servers. "
                    "You must provide the executable command (e.g. 'npx') and args "
                    "(e.g. ['-y', '<npm-package>']). Use web_search to find the "
                    "server's npm package or GitHub repo, then fetch_page to read "
                    "the installation instructions. Do NOT retry without the command."
                )
            if type == "sse" and not url:
                _last_error_key = call_key
                return (
                    "Error: 'url' is required for SSE servers. "
                    "Use web_search to find the server's URL. "
                    "Do NOT retry without the url."
                )

        # Validate: for remove, check server exists and is user-configured
        if action == "remove":
            all_servers = mcp_server_service.list_all()
            target = next((s for s in all_servers if s["name"] == name), None)
            if target is None:
                return f"Error: Server '{name}' not found."
            if target.get("builtin"):
                return (
                    f"Error: Server '{name}' is a built-in server and cannot be removed. "
                    "You can disable it instead."
                )

        # Validate config via MCPServerConfig for add/update
        if action in ("add", "update"):
            try:
                from tensortruth.agents.config import MCPServerConfig, MCPServerType

                MCPServerConfig(
                    name=name,
                    type=MCPServerType(config.get("type", "stdio")),
                    command=config.get("command"),
                    args=config.get("args", []),
                    url=config.get("url"),
                    description=config.get("description"),
                    env=config.get("env"),
                    enabled=config.get("enabled", True),
                )
            except Exception as e:
                _last_error_key = call_key
                return f"Error: Invalid server configuration: {e}"

        # Verify the server actually works before proposing
        if action == "add":
            ok, verify_msg = await _verify_mcp_server(config, progress_emitter)
            if not ok:
                _last_error_key = call_key
                return (
                    f"Error: {verify_msg} "
                    "Fix the command/args and try again. Use web_search and "
                    "fetch_page to find the correct installation instructions."
                )
            logger.info("MCP verification passed for '%s': %s", name, verify_msg)

        # Success path — clear error tracking
        _last_error_key = None

        # Request user confirmation (blocks until approved/rejected/expired)
        action_type = f"mcp_{action}"
        title = f"{action.capitalize()} MCP server '{name}'"
        outcome = await request_confirmation(
            confirmation_service=confirmation_service,
            progress_emitter=progress_emitter,
            tool_name="manage_mcp_server",
            action_type=action_type,
            title=title,
            summary=summary,
            details={
                "action": action,
                "config": config,
                "target_name": name,
            },
            session_id=session_id,
        )

        if outcome == "rejected":
            return f"User rejected the proposal to {action} MCP server '{name}'."
        if outcome == "expired":
            return f"Confirmation timed out for {action} MCP server '{name}'."

        # Apply the change directly
        try:
            if action == "add":
                mcp_server_service.add(config)
            elif action == "update":
                mcp_server_service.update(name, config)
            elif action == "remove":
                mcp_server_service.remove(name)

            # Reload tools so the new/changed MCP server is available
            await tool_service.reload()
        except Exception as e:
            logger.error("Failed to apply MCP %s for '%s': %s", action, name, e)
            return f"Error applying {action} for MCP server '{name}': {e}"

        return f"Successfully {action}{'d' if action != 'add' else 'ed'} MCP server '{name}'."

    return FunctionTool.from_defaults(
        async_fn=manage_mcp_server,
        name="manage_mcp_server",
        description=(
            "Add, update, or remove an MCP server configuration. "
            "The user will be prompted to approve the change; the tool blocks "
            "until they respond and applies the change automatically. "
            "Do NOT tell the user to review or approve — it happens inline. "
            "Just report the result returned by this tool. "
            "IMPORTANT: For non-preset servers, you MUST research the correct "
            "command and args (via web_search/fetch_page) BEFORE calling this tool. "
            "Do not call with command=null for stdio servers.\n\n"
            "Example — to add a server found on a README:\n"
            '  action="add", name="my-server", command="npx",\n'
            '  args=["-y", "@org/mcp-server-name"],\n'
            '  env={"API_KEY": "$API_KEY"},\n'
            '  summary="Add My Server for API access"'
        ),
        fn_schema=ProposeMCPServerInput,
    )


def create_add_arxiv_paper_tool(
    progress_emitter: ProgressEmitter,
    confirmation_service: "ToolConfirmationService",
    session_id: str,
    project_id: Optional[str] = None,
) -> FunctionTool:
    """Create an add_arxiv_paper FunctionTool for the orchestrator.

    Downloads an arXiv paper PDF and adds it to the session or project
    document store. Requires user confirmation before downloading.
    Reuses the same download + upload logic as the ``_upload_arxiv``
    REST endpoint in ``documents.py``.
    """

    async def add_arxiv_paper(arxiv_id: str, scope: str = "session") -> str:
        """Download an arXiv paper and add it to the session or project documents.

        Use this tool when the user asks you to add, save, or download an arXiv
        paper to their session or project. The user will be prompted to approve
        the download; the tool blocks until they respond. The paper will appear
        in the documents panel and must be indexed before it can be used for
        RAG queries.

        Args:
            arxiv_id: arXiv identifier (e.g. "2301.12345") or full URL.
            scope: "session" (default) or "project".

        Returns:
            Success message with paper details, or an error string.
        """
        import tempfile
        from pathlib import Path

        import arxiv as arxiv_lib

        from tensortruth.api.deps import get_document_service
        from tensortruth.utils.validation import validate_arxiv_id

        # Validate scope
        if scope not in ("session", "project"):
            return f"Error: scope must be 'session' or 'project', got '{scope}'"

        if scope == "project" and not project_id:
            return (
                "Error: cannot add to project — this session is not part of a "
                "project. Use scope='session' instead."
            )

        scope_id = project_id if scope == "project" else session_id
        scope_type = scope

        # Validate arXiv ID
        normalized = validate_arxiv_id(arxiv_id)
        if normalized is None:
            return f"Error: invalid arXiv ID '{arxiv_id}'"

        # Request user confirmation before downloading
        outcome = await request_confirmation(
            confirmation_service=confirmation_service,
            progress_emitter=progress_emitter,
            tool_name="add_arxiv_paper",
            action_type="add_document",
            title=f"Add arXiv paper {normalized}",
            summary=f"Download arXiv:{normalized} and add it to {scope} documents.",
            details={
                "arxiv_id": normalized,
                "url": f"https://arxiv.org/abs/{normalized}",
                "scope": scope,
            },
            session_id=session_id,
        )

        if outcome == "rejected":
            return "User rejected the proposal to add the arXiv paper."
        if outcome == "expired":
            return "Confirmation timed out for adding the arXiv paper."

        # Emit progress
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="add_arxiv_paper",
                phase="downloading",
                message=f"Downloading arXiv paper {normalized}...",
            ),
        )

        # Fetch paper and download PDF in executor (blocking I/O)
        loop = asyncio.get_event_loop()

        def _fetch():
            search = arxiv_lib.Search(id_list=[normalized])
            try:
                paper = next(search.results())
            except StopIteration:
                return None, None

            with tempfile.TemporaryDirectory() as tmpdir:
                paper.download_pdf(dirpath=tmpdir, filename="paper.pdf")
                pdf_bytes = Path(tmpdir, "paper.pdf").read_bytes()

            return paper, pdf_bytes

        try:
            paper, pdf_bytes = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch),
                timeout=60,
            )
        except asyncio.TimeoutError:
            return "Error: arXiv download timed out after 60 seconds."
        except Exception as e:
            logger.error("arXiv download failed for %s: %s", arxiv_id, e)
            return f"Error: failed to download paper — {e}"

        if paper is None:
            return f"Error: paper '{normalized}' not found on arXiv"

        # Upload PDF
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="add_arxiv_paper",
                phase="uploading",
                message="Adding paper to documents...",
            ),
        )

        try:
            filename = f"{paper.title}.pdf"
            with get_document_service(scope_id, scope_type) as doc_service:
                metadata = doc_service.upload(pdf_bytes, filename)

            # Store arXiv metadata
            from tensortruth.utils.metadata import format_authors as _fmt_authors

            authors_str = ", ".join(a.name for a in paper.authors)
            formatted = _fmt_authors(authors_str)
            display = f"{paper.title}, {formatted}" if formatted else paper.title
            arxiv_metadata = {
                "title": paper.title,
                "authors": authors_str,
                "display_name": display,
                "source_url": f"https://arxiv.org/abs/{normalized}",
                "doc_type": "arxiv_paper",
            }
            with get_document_service(scope_id, scope_type) as doc_service:
                doc_service.set_metadata(metadata.pdf_id, arxiv_metadata)
        except Exception as e:
            logger.error("Failed to upload arXiv paper %s: %s", arxiv_id, e)
            return f"Error: downloaded paper but failed to upload — {e}"

        authors_short = (
            authors_str[:80] + "..." if len(authors_str) > 80 else authors_str
        )
        return (
            f"Successfully added arXiv paper to {scope} documents.\n"
            f"Title: {paper.title}\n"
            f"Authors: {authors_short}\n"
            f"Doc ID: {metadata.pdf_id}\n"
            f"IMPORTANT: Tell the user they need to click 'Build Index' in the "
            f"documents panel to index this paper before it can be searched via "
            f"RAG queries."
        )

    return FunctionTool.from_defaults(
        async_fn=add_arxiv_paper,
        name="add_arxiv_paper",
        description=(
            "Download an arXiv paper and add it to the session or project "
            "documents. The user will be prompted to approve; the tool blocks "
            "until they respond. Use when the user asks to save/add/download an "
            "arXiv paper they found or mentioned. Accepts arXiv IDs like "
            "'2301.12345' or full URLs. "
            "After success, always remind the user to build the index from the "
            "documents panel so the paper becomes searchable."
        ),
        fn_schema=AddArxivPaperInput,
    )


def create_all_tool_wrappers(
    tool_service: ToolService,
    progress_emitter: ProgressEmitter,
    rag_service: Optional["RAGService"] = None,
    session_params: Optional[Dict[str, object]] = None,
    session_messages: Optional[List[Dict[str, object]]] = None,
    rag_result_callback: Optional[RAGResultCallback] = None,
    title_reranker_model: Optional[str] = None,
    content_reranker_model: Optional[str] = None,
    reranker_device: str = "cpu",
    title_threshold: float = 0.1,
    content_threshold: float = 0.1,
    context_window: int = 16384,
    custom_instructions: Optional[str] = None,
    web_query_setter: Optional[Callable[[str], None]] = None,
    web_query_getter: Optional[Callable[[], Optional[str]]] = None,
    web_search_results_setter: Optional[Callable[[List[Dict]], None]] = None,
    web_search_results_getter: Optional[Callable[[], Optional[List[Dict]]]] = None,
    web_result_callback: Optional[WebResultCallback] = None,
    fetched_urls_getter: Optional[Callable[[], set]] = None,
    fetched_urls_updater: Optional[Callable[[List[str]], None]] = None,
    full_output_callback: Optional[FullOutputCallback] = None,
    confirmation_service: Optional["ToolConfirmationService"] = None,
    mcp_server_service: Optional["MCPServerService"] = None,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> List[FunctionTool]:
    """Create all orchestrator tool wrappers at once.

    Convenience function that builds the full set of wrapped tools for
    the orchestrator, including the RAG query tool when a RAGService with
    a loaded engine is available, and MCP management tools when the
    confirmation and server services are provided.

    Returns:
        List of FunctionTool wrappers. Always includes web_search, fetch_page,
        fetch_pages_batch. Includes rag_query when rag_service is loaded.
        Includes MCP management tools when services are provided.
    """
    tools: List[FunctionTool] = []

    # RAG tool (only when engine is loaded with indexed modules)
    if rag_service is not None and rag_service.is_loaded():
        tools.append(
            create_rag_tool(
                rag_service,
                progress_emitter,
                session_params=session_params,
                session_messages=session_messages,
                rag_result_callback=rag_result_callback,
                full_output_callback=full_output_callback,
            )
        )

    # Web tools (with reranking pipeline integration)
    tools.extend(
        [
            create_web_search_tool(
                tool_service,
                progress_emitter,
                reranker_model=title_reranker_model,
                reranker_device=reranker_device,
                title_threshold=title_threshold,
                web_query_setter=web_query_setter,
                web_search_results_setter=web_search_results_setter,
            ),
            create_fetch_page_tool(
                tool_service,
                progress_emitter,
                fetched_urls_getter=fetched_urls_getter,
                fetched_urls_updater=fetched_urls_updater,
                full_output_callback=full_output_callback,
            ),
            create_fetch_pages_batch_tool(
                tool_service,
                progress_emitter,
                reranker_model=content_reranker_model,
                reranker_device=reranker_device,
                content_threshold=content_threshold,
                context_window=context_window,
                custom_instructions=custom_instructions,
                web_query_getter=web_query_getter,
                web_search_results_getter=web_search_results_getter,
                web_result_callback=web_result_callback,
                fetched_urls_getter=fetched_urls_getter,
                fetched_urls_updater=fetched_urls_updater,
                full_output_callback=full_output_callback,
            ),
        ]
    )

    # MCP management tools (when services are provided)
    if mcp_server_service is not None:
        tools.append(create_list_mcp_servers_tool(mcp_server_service, progress_emitter))
        tools.append(create_get_mcp_presets_tool(mcp_server_service, progress_emitter))
        if confirmation_service is not None and session_id is not None:
            tools.append(
                create_manage_mcp_server_tool(
                    confirmation_service,
                    mcp_server_service,
                    tool_service,
                    progress_emitter,
                    session_id,
                )
            )

    # ArXiv paper tool (needs session_id + confirmation_service)
    if session_id is not None and confirmation_service is not None:
        tools.append(
            create_add_arxiv_paper_tool(
                progress_emitter, confirmation_service, session_id, project_id
            )
        )

    return tools
