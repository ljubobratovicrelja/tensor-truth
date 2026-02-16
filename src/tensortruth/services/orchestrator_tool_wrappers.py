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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field

from tensortruth.agents.tool_output import extract_tool_text
from tensortruth.services.models import ToolProgress
from tensortruth.services.tool_service import ToolService

if TYPE_CHECKING:
    from tensortruth.services.models import RAGRetrievalResult
    from tensortruth.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# Type alias for the progress emitter callback.
# Accepts a ToolProgress and returns None. May be sync or async;
# async variants are awaited inside the tool wrappers.
ProgressEmitter = Callable[[ToolProgress], None]

# Type alias for a callback that receives the raw RAGRetrievalResult.
# Used by the stream translator to extract proper sources and metrics
# without relying on text parsing of the tool output.
RAGResultCallback = Callable[["RAGRetrievalResult"], None]


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

        Use this tool when the user's question likely relates to topics covered
        by the loaded document modules (papers, documentation, books, uploaded
        PDFs). Returns source excerpts with relevance scores and confidence
        assessment. Does NOT generate a response — you must synthesize the
        answer from the returned sources yourself.
        """
        # Progress is emitted via RAGService.retrieve()'s progress_callback,
        # but we also emit a top-level phase for the orchestrator
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="rag",
                phase="retrieving",
                message="Searching knowledge base...",
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
            "Use this when the question relates to topics covered by the "
            "loaded modules. For current events or live information, use "
            "web_search instead."
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
) -> FunctionTool:
    """Create a web_search FunctionTool for the orchestrator.

    The returned tool searches the web using a single query string and emits
    progress via the captured emitter.

    Args:
        tool_service: ToolService instance with loaded tools.
        progress_emitter: Callable that receives ToolProgress updates.

    Returns:
        A FunctionTool that wraps ToolService's search_web tool.
    """

    async def web_search(query: str) -> str:
        """Search the web for information using a query string.

        Use this tool when the user asks about current events, recent
        developments, or topics not covered by the indexed knowledge base.
        Returns search results as JSON with titles, URLs, and snippets.
        """
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="web_search",
                phase="searching",
                message="Searching the web...",
                metadata={"query": query},
            ),
        )

        result = await tool_service.execute_tool("search_web", {"queries": query})

        if not result.get("success"):
            error_msg = result.get("error", "Unknown search error")
            logger.warning(f"web_search failed: {error_msg}")
            return f"Error: {error_msg}"

        return extract_tool_text(result["data"])

    return FunctionTool.from_defaults(
        async_fn=web_search,
        name="web_search",
        description=(
            "Search the web for current information, recent events, or topics "
            "not in the indexed knowledge base. Returns JSON search results "
            "with titles, URLs, and snippets. Use this before fetch_page to "
            "find relevant URLs."
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
        """Fetch a web page and convert it to clean markdown.

        Use this tool to read the full content of a web page. Works with
        Wikipedia, GitHub, arXiv, YouTube, and general websites. Pass a URL
        obtained from web_search results or provided by the user.
        """
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="fetch_page",
                phase="fetching",
                message="Fetching page...",
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
            "Fetch a single web page and convert it to clean markdown text. "
            "Supports Wikipedia, GitHub, arXiv, YouTube, and general sites. "
            "Returns the page content as markdown, or an error message on "
            "failure. For multiple pages, use fetch_pages_batch instead."
        ),
        fn_schema=FetchPageInput,
    )


def create_fetch_pages_batch_tool(
    tool_service: ToolService,
    progress_emitter: ProgressEmitter,
) -> FunctionTool:
    """Create a fetch_pages_batch FunctionTool for the orchestrator.

    The returned tool fetches multiple URLs in parallel and emits per-page
    progress updates.

    Args:
        tool_service: ToolService instance with loaded tools.
        progress_emitter: Callable that receives ToolProgress updates.

    Returns:
        A FunctionTool that wraps ToolService's fetch_pages_batch tool.
    """

    async def fetch_pages_batch(urls: List[str]) -> str:
        """Fetch multiple web pages in parallel for efficient research.

        Use this tool instead of calling fetch_page multiple times. All pages
        are fetched simultaneously. Returns a JSON object with results for
        each URL including status (success/failed/skipped) and content.
        Recommended: pass 3-5 URLs at a time.
        """
        total = len(urls)
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="fetch_pages_batch",
                phase="fetching",
                message=f"Fetching {total} pages...",
                metadata={"urls": urls, "total": total},
            ),
        )

        result = await tool_service.execute_tool("fetch_pages_batch", {"urls": urls})

        if not result.get("success"):
            error_msg = result.get("error", "Unknown batch fetch error")
            logger.warning(f"fetch_pages_batch failed: {error_msg}")
            return f"Error: {error_msg}"

        raw_text = extract_tool_text(result["data"])

        # Emit completion progress with page-level summary
        try:
            parsed = json.loads(raw_text)
            pages = parsed.get("pages", [])
            success_count = sum(1 for p in pages if p.get("status") == "success")
            await _emit(
                progress_emitter,
                ToolProgress(
                    tool_id="fetch_pages_batch",
                    phase="fetched",
                    message=f"Fetched {success_count}/{total} pages.",
                    metadata={
                        "total": total,
                        "success": success_count,
                        "failed": total - success_count,
                    },
                ),
            )
        except (json.JSONDecodeError, AttributeError):
            pass

        return raw_text

    return FunctionTool.from_defaults(
        async_fn=fetch_pages_batch,
        name="fetch_pages_batch",
        description=(
            "Fetch multiple web pages in parallel (much faster than calling "
            "fetch_page multiple times). Returns a JSON object with results "
            "for each URL, including status and content. Recommended for "
            "research tasks requiring 3-5 pages."
        ),
        fn_schema=FetchPagesBatchInput,
    )


def create_all_tool_wrappers(
    tool_service: ToolService,
    progress_emitter: ProgressEmitter,
    rag_service: Optional[RAGService] = None,
    session_params: Optional[Dict[str, object]] = None,
    session_messages: Optional[List[Dict[str, object]]] = None,
    rag_result_callback: Optional[RAGResultCallback] = None,
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

    # Web tools
    tools.extend(
        [
            create_web_search_tool(tool_service, progress_emitter),
            create_fetch_page_tool(tool_service, progress_emitter),
            create_fetch_pages_batch_tool(tool_service, progress_emitter),
        ]
    )

    return tools
