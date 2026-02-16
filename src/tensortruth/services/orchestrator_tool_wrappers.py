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
from pydantic import BaseModel, Field

from tensortruth.agents.tool_output import extract_tool_text
from tensortruth.services.models import ToolProgress
from tensortruth.services.tool_service import ToolService

if TYPE_CHECKING:
    from tensortruth.core.source import SourceNode
    from tensortruth.services.models import RAGRetrievalResult
    from tensortruth.services.rag_service import RAGService

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


# --- Pydantic schemas for FunctionTool parameter validation ---


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


# --- Factory functions ---


def create_rag_tool(
    rag_service: RAGService,
    progress_emitter: ProgressEmitter,
    session_params: Optional[Dict[str, object]] = None,
    session_messages: Optional[List[Dict[str, object]]] = None,
    rag_result_callback: Optional[RAGResultCallback] = None,
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

        # Format result as a structured string for the LLM
        return _format_retrieval_result(result)

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

        # Store query and results for fetch_pages_batch to access
        if web_query_setter:
            web_query_setter(query)
        if web_search_results_setter:
            web_search_results_setter(search_results)

        return json.dumps(search_results, indent=2)

    return FunctionTool.from_defaults(
        async_fn=web_search,
        name="web_search",
        description=(
            "Search the web using DuckDuckGo for current information, recent "
            "events, or topics not in the indexed knowledge base. Returns a "
            "JSON array of search results, each containing a title, URL, and "
            "short snippet. Output contains only metadata and snippets — not "
            "full page content."
        ),
        fn_schema=WebSearchInput,
    )


def create_fetch_page_tool(
    tool_service: ToolService,
    progress_emitter: ProgressEmitter,
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

        return extract_tool_text(result["data"])

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

        parts = []
        for url, title, content in fitted_pages:
            parts.append(f"## {title}\nSource: {url}\n\n{content}")
        return "\n\n---\n\n".join(parts)

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


def create_all_tool_wrappers(
    tool_service: ToolService,
    progress_emitter: ProgressEmitter,
    rag_service: Optional["RAGService"] = None,
    session_params: Optional[Dict[str, object]] = None,
    session_messages: Optional[List[Dict[str, object]]] = None,
    rag_result_callback: Optional[RAGResultCallback] = None,
    reranker_model: Optional[str] = None,
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
) -> List[FunctionTool]:
    """Create all orchestrator tool wrappers at once.

    Convenience function that builds the full set of wrapped tools for
    the orchestrator, including the RAG query tool when a RAGService with
    a loaded engine is available.

    Args:
        tool_service: ToolService instance with loaded tools.
        progress_emitter: Callable that receives ToolProgress updates.
        rag_service: Optional RAGService. When provided and loaded, a
            rag_query tool is included in the returned set.
        session_params: Session engine parameters (forwarded to rag_query).
        session_messages: Chat history (forwarded to rag_query for condensation).
        rag_result_callback: Optional callback to receive the raw
            RAGRetrievalResult for source extraction by the stream translator.
        reranker_model: Optional reranker model for title/content reranking.
        reranker_device: Device for reranker model.
        title_threshold: Minimum score for title-stage reranking.
        content_threshold: Minimum score for content-stage reranking.
        context_window: Context window size in tokens.
        custom_instructions: Optional custom instructions for reranking.
        web_query_setter: Callback to store web search query.
        web_query_getter: Callback to retrieve stored web search query.
        web_search_results_setter: Callback to store web search results.
        web_search_results_getter: Callback to retrieve stored web search results.
        web_result_callback: Callback to emit SourceNode objects.

    Returns:
        List of FunctionTool wrappers. Always includes web_search, fetch_page,
        fetch_pages_batch. Includes rag_query when rag_service is loaded.
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
            )
        )

    # Web tools (with reranking pipeline integration)
    tools.extend(
        [
            create_web_search_tool(
                tool_service,
                progress_emitter,
                reranker_model=reranker_model,
                reranker_device=reranker_device,
                title_threshold=title_threshold,
                web_query_setter=web_query_setter,
                web_search_results_setter=web_search_results_setter,
            ),
            create_fetch_page_tool(tool_service, progress_emitter),
            create_fetch_pages_batch_tool(
                tool_service,
                progress_emitter,
                reranker_model=reranker_model,
                reranker_device=reranker_device,
                content_threshold=content_threshold,
                context_window=context_window,
                custom_instructions=custom_instructions,
                web_query_getter=web_query_getter,
                web_search_results_getter=web_search_results_getter,
                web_result_callback=web_result_callback,
            ),
        ]
    )

    return tools
