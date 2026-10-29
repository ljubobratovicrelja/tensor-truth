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
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional
from urllib.parse import urlparse

from llama_index.core.tools import FunctionTool
from llama_index.tools.mcp import BasicMCPClient
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, Field

from tensortruth.agents.tool_output import extract_tool_text
from tensortruth.services.context_documents import CHARS_PER_TOKEN
from tensortruth.services.models import ToolProgress
from tensortruth.services.tool_service import ToolService

if TYPE_CHECKING:
    from pathlib import Path

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

# Type alias for a callback that receives full tool output (used by the
# stream translator / UI). Args: (tool_name, full_output_text).
FullOutputCallback = Callable[[str, str], None]

# --- Tool content budgets -------------------------------------------------
# Tools must hand their *content* back to the calling LLM, not just a summary:
# the orchestrator agent has no separate synthesizer, so whatever the tool
# returns is the only material the model can read. Output is capped as a share
# of the context window so a single call cannot crowd out the conversation.
RAG_CONTENT_PCT = 0.12
FETCH_PAGE_CONTENT_PCT = 0.15
FETCH_BATCH_CONTENT_PCT = 0.25
REPO_CONTENT_PCT = 0.15
MIN_TOOL_CONTENT_CHARS = 2000
MAX_TOOL_CONTENT_CHARS = 48000
# Hard cap per RAG source excerpt, regardless of how much budget is left.
RAG_SOURCE_MAX_CHARS = 1500


def _tool_content_budget(context_window: int, pct: float) -> int:
    """Character budget for one tool result, as a share of the context window."""
    total_chars = max(1, int(context_window or 16384)) * CHARS_PER_TOKEN
    return max(
        MIN_TOOL_CONTENT_CHARS,
        min(MAX_TOOL_CONTENT_CHARS, int(total_chars * pct)),
    )


def _fit_tool_text(text: str, budget: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``budget`` chars, appending an explicit marker."""
    if budget <= 0 or len(text) <= budget:
        return text, False
    marker = "\n\n...[truncated to fit the context budget]"
    keep = max(0, budget - len(marker))
    return text[:keep].rstrip() + marker, True


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
            'Where to add the paper: "session" (default) adds it to the '
            'current chat session, "project" adds it to the parent project.'
        ),
    )
    destination: str = Field(
        default="index",
        description=(
            'How the paper becomes usable: "index" (default) indexes it for '
            'rag_query; "context" attaches its full text to this session\'s '
            "conversation context instead (no index build, needs a large "
            "context window). The user can also pick the other option from the "
            "approval card."
        ),
    )


class BuildIndexInput(BaseModel):
    """Input schema for build_index orchestrator tool."""

    scope: str = Field(
        default="session",
        description=(
            'Which document collection to index: "session" (default) for the '
            'current chat session, "project" for the parent project.'
        ),
    )


class LoadModulesInput(BaseModel):
    """Input schema for load_modules orchestrator tool."""

    modules: List[str] = Field(
        description=(
            "Module names to load into the knowledge base, exactly as listed in "
            "your system prompt (1 to 6 per call)."
        )
    )


class RAGQueryInput(BaseModel):
    """Input schema for rag_query orchestrator tool."""

    query: str = Field(
        description="A search query to look up in the indexed knowledge base."
    )


class ChunkRefInput(BaseModel):
    """Input schema for tools that take a chunk reference from rag_query."""

    node_id: str = Field(
        description=(
            "Chunk reference from a rag_query source header, e.g. "
            "'chunk node_id=1f3a9c02'. The 8-character prefix is enough."
        )
    )


class ReadNeighboursInput(ChunkRefInput):
    """Input schema for read_neighbours."""

    before: int = Field(
        default=2, description="How many preceding chunks to include (0-10)."
    )
    after: int = Field(
        default=2, description="How many following chunks to include (0-10)."
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


class InspectInput(BaseModel):
    """Input schema for the ``inspect`` repository tool.

    ``action`` is a closed enum: anything outside the literal set is rejected
    by pydantic before the tool body runs, so there is no passthrough to a
    shell. Every other field is validated again inside the tool.
    """

    action: Literal["list", "clone", "tree", "read", "grep", "git", "drop"] = Field(
        description=(
            "list: repositories cloned into this session. "
            "clone: clone a git repository (needs url). "
            "tree: list a directory (path, depth). "
            "read: read a text file (path, start_line, max_lines). "
            "grep: search file contents (pattern, path, case_sensitive). "
            "git: run log/show/blame (git_command, rev, path). "
            "drop: delete a cloned repository."
        )
    )
    url: Optional[str] = Field(
        default=None,
        description="action='clone': repository URL, e.g. https://github.com/owner/repo",
    )
    repo: Optional[str] = Field(
        default=None,
        description=(
            "Repository selector as 'owner/repo'. Omit it when exactly one "
            "repository is cloned into the session."
        ),
    )
    path: Optional[str] = Field(
        default=None,
        description="Path relative to the repository root (directories allowed).",
    )
    pattern: Optional[str] = Field(
        default=None, description="action='grep': Python regular expression."
    )
    git_command: Optional[Literal["log", "show", "blame"]] = Field(
        default=None, description="action='git': which read-only git command to run."
    )
    rev: Optional[str] = Field(
        default=None,
        description="action='git': revision for show/blame (default HEAD).",
    )
    depth: int = Field(default=3, description="action='tree': depth, 1-6.")
    start_line: int = Field(
        default=1, description="action='read': first line to return (1-based)."
    )
    max_lines: int = Field(
        default=200, description="action='read'/'git': maximum lines to return."
    )
    case_sensitive: bool = Field(
        default=False, description="action='grep': case-sensitive matching."
    )
    full_history: bool = Field(
        default=False,
        description="action='clone': fetch full history instead of a shallow clone.",
    )


# --- Helper ---


async def _emit(emitter: ProgressEmitter, progress: ToolProgress) -> None:
    """Call the progress emitter, awaiting if it returns a coroutine."""
    result = emitter(progress)
    if asyncio.iscoroutine(result):
        await result


async def _queue_index_build(scope_id: str, scope_type: str) -> Optional[str]:
    """Submit an incremental index build for a scope.

    Used after adding a document so the default "index" destination needs no
    user click. Returns the task id, or None when the TaskRunner is not running
    (the agent can then fall back to the build_index tool).
    """
    try:
        from tensortruth.api.deps import get_document_service, get_task_runner

        runner = get_task_runner()
        if getattr(runner, "_worker_task", None) is None:
            logger.info("TaskRunner not started; skipping automatic index build")
            return None

        def _do_build(progress_callback):
            with get_document_service(scope_id, scope_type) as doc_service:
                doc_service.build_index(progress_callback=progress_callback)

        return await runner.submit(
            "build-index",
            _do_build,
            metadata={
                "scope_id": scope_id,
                "scope_type": scope_type,
                "source": "add_arxiv_paper",
            },
        )
    except Exception as e:
        logger.warning("Could not queue index build for %s: %s", scope_id, e)
        return None


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


_BUILD_DOC_STAGE_RE = re.compile(r"^(Converting|Loading) (.+?) \((\d+)/(\d+)\)$")

_BUILD_INDEX_POLL_INTERVAL = 0.25
_BUILD_INDEX_MAX_WAIT_SECONDS = 1800.0


def _format_build_stage(stage: str) -> tuple[str, str]:
    """Turn a raw build-index stage into (display message, phase name).

    The indexer reports stages like ``"Converting pdf_ab12cd_Paper.pdf (1/2)"``.
    The UI wants ``"Converting 1/2: Paper.pdf"`` so the reader sees position
    and document title separately from the internal file name.
    """
    text = (stage or "").strip()
    if not text:
        return "Preparing index build...", "preparing"

    match = _BUILD_DOC_STAGE_RE.match(text)
    if match:
        verb, filename, current, total = match.groups()
        try:
            from tensortruth.api.routes.documents import strip_doc_prefix

            title = strip_doc_prefix(filename.rsplit("/", 1)[-1])
        except Exception:  # pragma: no cover - defensive
            title = filename.rsplit("/", 1)[-1]
        return f"{verb} {current}/{total}: {title}", verb.lower()

    words = text.split()
    phase = words[0].lower().rstrip(":") if words else ""
    return text, phase if phase.isalpha() else "building"


# --- Factory functions ---


def _context_service_for(
    rag_service: Optional["RAGService"], session_id: Optional[str]
) -> Optional[Any]:
    """Build a ChunkContextService for a session, or None when unusable.

    Called per tool invocation rather than at construction: tools are created
    before load_modules runs, and the session's retriever (hence its docstores)
    can change or disappear between calls.
    """
    from tensortruth.services.chunk_context import build_for_session

    try:
        return build_for_session(rag_service, session_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Chunk context service unavailable: %s", exc)
        return None


def create_rag_tool(
    rag_service: RAGService,
    progress_emitter: ProgressEmitter,
    session_params: Optional[Dict[str, object]] = None,
    session_messages: Optional[List[Dict[str, object]]] = None,
    rag_result_callback: Optional[RAGResultCallback] = None,
    full_output_callback: Optional[FullOutputCallback] = None,
    context_window: int = 16384,
    catalog_names: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    module_mode: str = "auto",
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
        context_window: Model context window in tokens, used to cap the size
            of the excerpts returned to the agent.
        session_id: Session whose retrieval set this tool queries.
        module_mode: ``"auto"`` or ``"manual"``. In manual mode there is no
            load_modules tool, so the no-engine guidance must not suggest it.

    Returns:
        A FunctionTool wrapping RAGService.retrieve().
    """
    # Capture mutable references so the tool always sees current state
    captured_params = session_params
    captured_messages = session_messages
    # Owner-scoped calls only pass the keyword when the session is known, so
    # legacy single-engine callers keep working unchanged.
    owner_kwargs: Dict[str, Any] = {"owner": session_id} if session_id else {}

    async def rag_query(query: str) -> str:
        """Search the indexed knowledge base for information relevant to the query.

        Returns source excerpts with relevance scores and confidence assessment
        for topics covered by the loaded document modules (papers, documentation,
        books, uploaded PDFs).
        """
        # The tool is registered even with no engine loaded so the agent always
        # sees it; without modules there is nothing to search, only guidance.
        if not rag_service.is_loaded(**owner_kwargs):
            if module_mode == "manual":
                return (
                    "No knowledge modules are selected for this session. The "
                    "user has not selected any knowledge modules, and the "
                    "selection is fixed for this session; answer without "
                    "searching the knowledge base."
                )
            if catalog_names:
                available = ", ".join(catalog_names)
            else:
                available = "the modules listed in your system prompt"
            return (
                "No knowledge modules are loaded. Call load_modules with one of: "
                f"{available}."
            )

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
                **owner_kwargs,
            ),
        )

        # Forward raw result for source extraction by the stream translator
        if rag_result_callback is not None:
            rag_result_callback(result)

        # Format source excerpts (+ scores, confidence, metrics) for the agent.
        # The orchestrator has no separate synthesizer: this text IS what the
        # model reads, so it must carry the chunk content, budget-capped.
        # The context service is only for headers (node id prefix + section
        # size); it never changes how the excerpt budget is spent.
        output = _format_retrieval_result(
            result,
            max_chars=_tool_content_budget(context_window, RAG_CONTENT_PCT),
            context_service=_context_service_for(rag_service, session_id),
        )
        if full_output_callback is not None:
            full_output_callback("rag_query", output)

        return output

    return FunctionTool.from_defaults(
        async_fn=rag_query,
        name="rag_query",
        description=(
            "Search the indexed knowledge base (documents, papers, books, "
            "uploaded PDFs) for information relevant to a query. Returns "
            "source excerpts (actual text) with relevance scores and a "
            "confidence level. Always call this when the answer may be in the "
            "attached documents — do not answer from memory. Not suitable for "
            "current events or live information."
        ),
        fn_schema=RAGQueryInput,
    )


MAX_MODULES_PER_LOAD = 6

# Expansion output cap. A section larger than this is not worth pasting whole:
# the agent already has the chunk it asked about, and a bigger blob would push
# the rest of the retrieval out of the context window.
MAX_CONTEXT_CHARS = 6000


def _truncate_context(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 40)].rstrip() + "\n... [truncated, section too long]"


def create_chunk_context_tools(
    rag_service: Optional["RAGService"] = None,
    session_id: Optional[str] = None,
    context_service: Optional[Any] = None,
) -> List[FunctionTool]:
    """Create the chunk-context expansion tools for the orchestrator.

    rag_query prints a short node id per source; these tools spend one cheap
    docstore walk (no embedding, no reindex) to hand back the enclosing section,
    the neighbour chunks, or the document outline. They exist because the
    retrieved ~500-char leaves routinely start mid-sentence.

    Args:
        rag_service: RAGService whose session retriever supplies the docstores.
        session_id: Session whose retrieval set to inspect.
        context_service: Pre-built ChunkContextService; when given, the tools
            use it instead of resolving one per call (used by tests).

    Returns:
        expand_section, read_neighbours and document_outline FunctionTools.
    """

    def _service() -> Optional[Any]:
        if context_service is not None:
            return context_service
        return _context_service_for(rag_service, session_id)

    no_documents = (
        "No documents are loaded for this session, so chunk ids cannot be "
        "resolved. Call load_modules first, then rag_query."
    )

    def _not_found(node_id: str) -> str:
        return (
            f"Chunk id '{node_id}' not found (it may be mistyped, ambiguous, or "
            "belong to documents that are no longer loaded). Chunk ids come "
            "from the 'chunk node_id=' field in rag_query source headers; run "
            "rag_query again if unsure."
        )

    def _document_label(service: Any, node: Any) -> str:
        name = service.document_name(node) or "unknown document"
        return f'"{name}"'

    async def expand_section(node_id: str) -> str:
        """Return the whole section that contains a retrieved chunk.

        Use when a rag_query excerpt looks relevant but is cut off, starts
        mid-sentence, or stops before the explanation finishes: this returns
        the enclosing section (up to 6000 chars) without another search.
        """
        service = _service()
        if service is None:
            return no_documents
        node = service.resolve(node_id)
        if node is None:
            return _not_found(node_id)

        section, _ancestors = service.section(node.node_id, max_chars=MAX_CONTEXT_CHARS)
        # Sections bigger than the cap are not an error: the chunk's own text
        # is still the best bounded answer we have.
        target = section if section is not None else node
        text = _truncate_context(target.get_content() or "")
        if not text.strip():
            return f"Chunk id '{node_id}' resolved but its section has no text."

        included = service.child_refs(target) or [target.node_id[:8]]
        header = [
            f"Section: {service.heading_path(target)}",
            f"Document: {_document_label(service, target)}",
            f"Chunks included: {', '.join(included)}",
            (
                "Scope: enclosing section"
                if section is not None
                else "Scope: the chunk itself (its section exceeds 6000 chars)"
            ),
        ]
        return "\n".join(header) + "\n\n" + text

    async def read_neighbours(node_id: str, before: int = 2, after: int = 2) -> str:
        """Return the chunks immediately before and after a retrieved chunk.

        Use when the discussion clearly continues on either side of an excerpt
        (a definition split across chunks, a list whose items are missing) but
        the section as a whole is too large to read.
        """
        service = _service()
        if service is None:
            return no_documents
        node = service.resolve(node_id)
        if node is None:
            return _not_found(node_id)

        before = max(0, min(int(before), 10))
        after = max(0, min(int(after), 10))
        pairs = service.neighbours(node.node_id, before=before, after=after)
        if not pairs:
            return (
                f"No adjacent chunks found for node_id={node.node_id[:8]} "
                "(it is the only chunk of its document, or the edges are not "
                "linked in this index)."
            )

        previous = [n for label, n in pairs if label == "previous"]
        following = [n for label, n in pairs if label == "next"]

        body: List[str] = [
            f"Neighbouring chunks of node_id={node.node_id[:8]} "
            f"in {_document_label(service, node)} "
            f"({len(previous)} before, {len(following)} after):",
            "",
        ]
        for n in previous:
            body.append(f"--- previous: node_id={n.node_id[:8]} ---")
            body.append(n.get_content() or "")
            body.append("")
        body.append(f"--- current: node_id={node.node_id[:8]} ---")
        body.append(node.get_content() or "")
        body.append("")
        for n in following:
            body.append(f"--- next: node_id={n.node_id[:8]} ---")
            body.append(n.get_content() or "")
            body.append("")
        return _truncate_context("\n".join(body).rstrip())

    async def document_outline(node_id: str) -> str:
        """List the sections of the document a retrieved chunk belongs to.

        Use to find other sections of the same document (a later chapter, the
        results after the methods) when the retrieved excerpt covers only part
        of what the question needs. Then call expand_section on a listed id.
        """
        service = _service()
        if service is None:
            return no_documents
        node = service.resolve(node_id)
        if node is None:
            return _not_found(node_id)

        entries = service.outline_entries(node.node_id, max_entries=40)
        if not entries:
            return (
                f"The document of node_id={node.node_id[:8]} has no section "
                "structure (indexed before the structure parser); only "
                "read_neighbours is available for it."
            )

        from tensortruth.services.chunk_context import format_char_size

        lines = [
            f"Outline of {_document_label(service, node)} "
            f"({len(entries)} sections, in document order):"
        ]
        for index, title, section_id, size in entries:
            lines.append(
                f"{index}. {title} | node_id={section_id[:8]} | "
                f"{format_char_size(size)}"
            )
        lines.append("")
        lines.append(
            "Call expand_section with a section's node_id to read it, or "
            "rag_query for content-level search."
        )
        return _truncate_context("\n".join(lines))

    return [
        FunctionTool.from_defaults(
            async_fn=expand_section,
            name="expand_section",
            description=(
                "Expand a retrieved chunk to its enclosing section. Call this "
                "when a rag_query excerpt looks relevant but is truncated, "
                "starts mid-sentence, or stops before the explanation ends. "
                "Takes the 'chunk node_id' prefix printed in the rag_query "
                "source header and returns the section text (up to 6000 "
                "chars) with its document and heading path. No new search is "
                "performed, so it is cheap."
            ),
            fn_schema=ChunkRefInput,
        ),
        FunctionTool.from_defaults(
            async_fn=read_neighbours,
            name="read_neighbours",
            description=(
                "Read the chunks immediately before and after a retrieved "
                "chunk. Call this when the surrounding discussion is what is "
                "missing (a definition split across chunks, a list with items "
                "cut off) but the whole section is too large. Takes the 'chunk "
                "node_id' prefix from a rag_query source header; before/after "
                "default to 2 chunks each, capped at 6000 chars total."
            ),
            fn_schema=ReadNeighboursInput,
        ),
        FunctionTool.from_defaults(
            async_fn=document_outline,
            name="document_outline",
            description=(
                "List the sections of the document a retrieved chunk belongs "
                "to, with a node id per section. Call this to find other "
                "sections of the same document (later chapters, results after "
                "methods) when one excerpt does not cover the question, then "
                "expand_section on a listed node id."
            ),
            fn_schema=ChunkRefInput,
        ),
    ]


def create_load_modules_tool(
    rag_service: "RAGService",
    progress_emitter: ProgressEmitter,
    module_catalog: Optional[List[Any]] = None,
    loaded_modules: Optional[List[str]] = None,
    session_params: Optional[Dict[str, object]] = None,
    additional_index_paths: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    session_service: Optional[Any] = None,
) -> FunctionTool:
    """Create a load_modules FunctionTool for the orchestrator.

    The agent decides which index modules a question needs and loads them; the
    engine is rebuilt with the merged module set (existing modules + the new
    ones) while preserving the session params and index paths captured here.

    ``loaded_modules`` is a shared mutable list owned by the orchestrator: the
    tool appends to it so two loads in the same turn see each other, and the
    merged set is persisted to the session so the next turn's reload check and
    the UI agree with the engine.

    Args:
        rag_service: RAGService instance (engine may not be loaded yet).
        progress_emitter: Callable that receives ToolProgress updates.
        module_catalog: Every indexed module (ModuleDescription) available to load.
        loaded_modules: Live list of currently loaded module names.
        session_params: Engine params to preserve across the reload.
        additional_index_paths: Session/project index paths to preserve.
        session_id: Session whose ``modules`` list is updated after a load.
        session_service: Optional SessionService override (tests).
    """
    catalog = list(module_catalog or [])
    catalog_by_name = {entry.name: entry for entry in catalog}
    catalog_names = [entry.name for entry in catalog]
    loaded = loaded_modules if loaded_modules is not None else []
    captured_params = session_params or {}
    captured_paths = list(additional_index_paths or [])

    def _catalog_suffix() -> str:
        if not catalog_names:
            return ""
        return " Valid modules: " + ", ".join(catalog_names) + "."

    def _persist() -> None:
        if not session_id:
            return
        try:
            service = session_service
            if service is None:
                from tensortruth.api.deps import get_session_service

                service = get_session_service()
            data = service.load()
            data = service.update_modules(session_id, list(loaded), data)
            service.save(data)
        except Exception:
            logger.exception(
                "Failed to persist loaded modules for session %s", session_id
            )

    async def load_modules(modules: List[str]) -> str:
        """Load knowledge modules into the searchable index set.

        Call this before rag_query when the question needs a module that is not
        yet loaded. Returns the modules that are now loaded.
        """
        names: List[str] = []
        for raw in modules:
            name = str(raw).strip()
            if name and name not in names:
                names.append(name)

        if not names:
            return "Error: provide at least one module name." + _catalog_suffix()
        if len(names) > MAX_MODULES_PER_LOAD:
            return (
                f"Error: at most {MAX_MODULES_PER_LOAD} modules can be loaded per "
                f"call ({len(names)} requested). Call load_modules again for the rest."
            )

        unknown = [name for name in names if name not in catalog_by_name]
        if unknown:
            return (
                "Error: unknown module(s): "
                + ", ".join(unknown)
                + "."
                + _catalog_suffix()
            )

        to_load = [name for name in names if name not in loaded]
        already_loaded = [name for name in names if name in loaded]

        if to_load:
            merged = list(loaded) + to_load
            total = len(to_load)
            for index, name in enumerate(to_load, start=1):
                entry = catalog_by_name[name]
                await _emit(
                    progress_emitter,
                    ToolProgress(
                        tool_id="modules",
                        phase="loading_module",
                        message=f"Loading {entry.display_name}...",
                        metadata={
                            "module": name,
                            "percent": int(100 * index / total),
                        },
                    ),
                )

            try:
                # Reloads only this session's set; other sessions keep theirs.
                # Never block the one event loop the whole app shares.
                await asyncio.to_thread(
                    rag_service.load_engine,
                    merged,
                    captured_params,
                    captured_paths,
                    **({"owner": session_id} if session_id else {}),
                )
            except Exception as exc:
                logger.exception("load_modules failed for %s", to_load)
                return (
                    f"Error: failed to load {', '.join(to_load)}: {exc}. "
                    f"Currently loaded: {', '.join(loaded) or 'none'}."
                )

            loaded[:] = merged
            _persist()

        lines: List[str] = []
        if to_load:
            lines.append(
                "Loaded: "
                + ", ".join(
                    f"{name} ({catalog_by_name[name].doc_count} docs)"
                    for name in to_load
                )
            )
        if already_loaded:
            lines.append("Already loaded: " + ", ".join(already_loaded))
        lines.append("Currently loaded: " + (", ".join(loaded) or "none"))
        lines.append("Use rag_query to search these modules.")
        return "\n".join(lines)

    return FunctionTool.from_defaults(
        async_fn=load_modules,
        name="load_modules",
        description=(
            "Load knowledge modules into the searchable set so rag_query can "
            "search them. Call this when the question relates to a module that "
            "is listed in your system prompt but not marked [loaded]. Accepts 1 "
            f"to {MAX_MODULES_PER_LOAD} module names per call; unknown names are "
            "rejected with the list of valid modules."
        ),
        fn_schema=LoadModulesInput,
    )


def _format_source_header(
    index: int,
    node: Any,
    title: str,
    score_str: str,
    context_service: Optional[Any] = None,
) -> str:
    """Build one source header line, with chunk pointers when available.

    ``[s2] Book > Chapter 1 > Methods | chunk node_id=1f3a9c02 | section
    ~4.1k chars available | score: 0.7391``

    The node id prefix is what expand_section/read_neighbours accept; the
    section size tells the agent whether expanding is worth a tool call. Both
    are omitted when no context service is wired (LLM-only or no engine), so
    the header degrades to the old title + score.
    """
    header = f"[s{index}] {title}"
    node_id = getattr(node, "node_id", None)

    if context_service is not None and node_id:
        try:
            heading = context_service.heading_path(node)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("heading_path failed for %s: %s", node_id, exc)
            heading = ""
        if heading and heading != title:
            header += f" > {heading}"
        header += f" | chunk node_id={node_id[:8]}"
        try:
            section_node, _ancestors = context_service.section(node_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("section lookup failed for %s: %s", node_id, exc)
            section_node = None
        if section_node is not None:
            from tensortruth.services.chunk_context import format_char_size

            size = len(section_node.get_content() or "")
            header += f" | section {format_char_size(size)} available"

    header += f" | score: {score_str}"
    return header


def _format_retrieval_result(
    result: "RAGRetrievalResult",
    max_chars: Optional[int] = None,
    context_service: Optional[Any] = None,
) -> str:
    """Format a RAGRetrievalResult as a structured string for LLM consumption.

    Includes the confidence level, per-source titles/scores and the actual
    chunk text, which is what the orchestrator agent reads. Excerpts are
    clipped to a per-source share of ``max_chars`` so a single retrieval
    cannot exhaust the context window.

    Args:
        result: RAGRetrievalResult from RAGService.retrieve().
        max_chars: Total character budget for all excerpts. None keeps the
            legacy per-source cap only.
        context_service: Optional ChunkContextService. When present each
            source header carries the chunk's node id prefix and the size of
            the enclosing section, which is how the agent decides to call
            expand_section instead of re-running the query.

    Returns:
        Formatted string with source excerpts and metadata.
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

    num_sources = max(1, len(result.source_nodes))
    per_source_cap = RAG_SOURCE_MAX_CHARS
    if max_chars is not None:
        per_source_cap = min(RAG_SOURCE_MAX_CHARS, max(max_chars // num_sources, 0))
    any_truncated = False

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
        if len(text) > per_source_cap:
            text = text[: max(0, per_source_cap - 16)] + "... [truncated]"
            any_truncated = True

        lines.append(
            _format_source_header(i, inner_node, title, score_str, context_service)
        )
        lines.append(text)
        lines.append("")

    if any_truncated:
        lines.append(
            "Note: one or more excerpts were truncated to fit the context "
            "budget. Run a narrower query if you need the missing text."
        )
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
    context_window: int = 16384,
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

        # Return the page text itself — a summary would leave the agent with
        # nothing to read. Budget-capped so one page cannot flood the window.
        body, truncated = _fit_tool_text(
            content, _tool_content_budget(context_window, FETCH_PAGE_CONTENT_PCT)
        )
        header = (
            f'Fetched: "{title}" ({url}) — {len(content)} chars'
            f"{' (truncated)' if truncated else ''}\n\n"
        )
        return header + body

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

        if full_output_callback is not None:
            full_output_callback("fetch_pages_batch", output)

        # Return the fitted page content itself (already sized to the context
        # window by the pipeline) rather than a titles-only summary.
        body, truncated = _fit_tool_text(
            output, _tool_content_budget(context_window, FETCH_BATCH_CONTENT_PCT)
        )
        summary = _summarize_fetch_pages_batch(fitted_pages, source_nodes, total)
        if truncated:
            summary += "\n(content truncated to fit the context budget)"
        return f"{summary}\n\n{body}"

    return FunctionTool.from_defaults(
        async_fn=fetch_pages_batch,
        name="fetch_pages_batch",
        description=(
            "Fetch multiple web pages in parallel given a list of URLs. Pages "
            "are fetched simultaneously, ranked by content relevance to the "
            "current query, and fitted to the context window. Accepts a list "
            "of URLs (recommended 3-5). Returns the full page content as "
            "formatted markdown sections plus per-page relevance scores."
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

    # Collect links from all pages, tracking which page they came from.
    # Regex over multi-KB page text is CPU work — run it on a worker thread.
    def _extract_all_links() -> List[tuple]:
        links_out: List[tuple] = []  # (anchor_text, url, page_title_or_url)
        seen: set = set()
        for page_url, content in page_contents:
            links = extract_links_from_markdown(content, page_url, exclude_urls)
            for anchor, link_url in links:
                if link_url not in seen:
                    seen.add(link_url)
                    # Use domain as source identifier
                    links_out.append((anchor, link_url, urlparse(page_url).netloc))
        return links_out

    all_links = await asyncio.to_thread(_extract_all_links)

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

    stderr_file = tempfile.TemporaryFile(mode="w+")

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
            stderr_output = stderr_file.read().strip()
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
        # Reads the MCP config from disk — off the event loop.
        servers = await asyncio.to_thread(mcp_server_service.list_all)
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
        presets = await asyncio.to_thread(mcp_server_service.get_presets)
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
        command: Optional[str] = "",
        args: Optional[List[str]] = None,
        url: Optional[str] = "",
        description: Optional[str] = "",
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
            presets = await asyncio.to_thread(mcp_server_service.get_presets)
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
            all_servers = await asyncio.to_thread(mcp_server_service.list_all)
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
        def _apply_config_change() -> None:
            # YAML config read-modify-write — blocking filesystem I/O.
            if action == "add":
                mcp_server_service.add(config)
            elif action == "update":
                mcp_server_service.update(name, config)
            elif action == "remove":
                mcp_server_service.remove(name)

        try:
            await asyncio.to_thread(_apply_config_change)

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

    async def add_arxiv_paper(
        arxiv_id: str, scope: str = "session", destination: str = "index"
    ) -> str:
        """Download an arXiv paper and add it to the session or project documents.

        Use this tool when the user asks you to add, save, or download an arXiv
        paper to their session or project. The user will be prompted to approve
        the download; the tool blocks until they respond.

        Indexing is the default: the paper is stored and an index build is
        queued automatically, so it becomes searchable through rag_query
        without the user clicking anything. The approval card also offers
        "add directly to context", which pastes the paper's text into the
        conversation instead (no index; suits large-context models).

        Args:
            arxiv_id: arXiv identifier (e.g. "2301.12345") or full URL.
            scope: "session" (default) or "project".
            destination: "index" (default) or "context". The user's choice on
                the approval card overrides this.

        Returns:
            Success message with paper details, or an error string.
        """
        import tempfile
        from pathlib import Path

        import arxiv as arxiv_lib

        from tensortruth.api.deps import get_document_service
        from tensortruth.utils.validation import validate_arxiv_id

        # Legacy alias: scope="context" used to mean session + attach to context
        if scope == "context":
            scope = "session"
            destination = "context"

        # Validate scope / destination
        if scope not in ("session", "project"):
            return f"Error: scope must be 'session' or 'project', got '{scope}'"
        if destination not in ("index", "context"):
            return (
                "Error: destination must be 'index' or 'context', "
                f"got '{destination}'"
            )

        if scope == "project" and not project_id:
            return (
                "Error: cannot add to project — this session is not part of a "
                "project. Use scope='session' instead."
            )

        assert (scope != "project") or (project_id is not None)
        scope_id: str = project_id if scope == "project" else session_id  # type: ignore[assignment]
        scope_type = "project" if scope == "project" else "session"
        # Indexing is the default; context is an opt-in alternative (either
        # requested by the agent or chosen by the user on the approval card).
        add_to_context = destination == "context"

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
            summary=(
                f"Download arXiv:{normalized} and attach it directly to the "
                "conversation context (no index)."
                if add_to_context
                else (
                    f"Download arXiv:{normalized}, add it to {scope} documents "
                    "and index it for search."
                )
            ),
            details={
                "arxiv_id": normalized,
                "url": f"https://arxiv.org/abs/{normalized}",
                "scope": scope,
                "in_context": add_to_context,
                # Extra choice rendered as a third button on the approval card
                "alternatives": (
                    []
                    if add_to_context
                    else [
                        {
                            "id": "context",
                            "label": "Add directly to context",
                            "description": (
                                "Skip indexing; paste the full text into this "
                                "conversation instead (large-context models)."
                            ),
                        }
                    ]
                ),
            },
            session_id=session_id,
        )

        if outcome == "rejected":
            return "User rejected the proposal to add the arXiv paper."
        if outcome == "expired":
            return "Confirmation timed out for adding the arXiv paper."
        if outcome == "context":
            # User explicitly chose the context buffer over indexing.
            add_to_context = True

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

        def _store_paper():
            """Upload + persist metadata on a worker thread.

            DocumentService is created and closed here so its stores keep
            thread affinity; the whole block is blocking filesystem I/O.
            """
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
            return metadata, authors_str

        try:
            metadata, authors_str = await asyncio.to_thread(_store_paper)
        except Exception as e:
            logger.error("Failed to upload arXiv paper %s: %s", arxiv_id, e)
            return f"Error: downloaded paper but failed to upload — {e}"

        context_chars: int | None = None
        context_error: str | None = None
        if add_to_context:
            await _emit(
                progress_emitter,
                ToolProgress(
                    tool_id="add_arxiv_paper",
                    phase="extracting",
                    message="Extracting paper text for context...",
                ),
            )

            def _attach_to_context() -> int | None:
                # set_in_context may run pymupdf4llm over the whole paper
                # (~16s for 50 pages). Service is created and closed on the
                # worker thread so its stores keep thread affinity.
                with get_document_service(scope_id, scope_type) as doc_service:
                    return doc_service.set_in_context(metadata.pdf_id, True)

            try:
                context_chars = await asyncio.to_thread(_attach_to_context)
            except Exception as e:
                # The PDF is stored; only the text extraction failed. Keep the
                # add and report it instead of pretending the paper is readable.
                context_error = str(e)
                logger.error(
                    "Failed to attach arXiv paper %s to context: %s", arxiv_id, e
                )

        authors_short = (
            authors_str[:80] + "..." if len(authors_str) > 80 else authors_str
        )

        if add_to_context and context_chars is not None:
            from tensortruth.services.context_documents import CHARS_PER_TOKEN

            tokens = (context_chars or 0) // CHARS_PER_TOKEN
            return (
                "Successfully added arXiv paper and attached it to this "
                "conversation's context.\n"
                f"Title: {paper.title}\n"
                f"Authors: {authors_short}\n"
                f"Doc ID: {metadata.pdf_id}\n"
                f"Attached to context: ~{tokens:,} tokens.\n"
                "The full text is included in the prompt on this and subsequent "
                "turns, so answer directly from it — no RAG query and no index "
                "build are needed. Do NOT ask the user to build an index just to "
                "read or discuss this paper. If it exceeds the context budget it "
                "is truncated automatically and the truncation is reported in "
                "your prompt."
            )

        if context_error is not None:
            return (
                f"Successfully added arXiv paper to {scope} documents, but its "
                f"text could not be extracted for direct context ({context_error}).\n"
                f"Title: {paper.title}\n"
                f"Authors: {authors_short}\n"
                f"Doc ID: {metadata.pdf_id}\n"
                "You cannot quote its body yet. Say so plainly; the user can "
                "build the index from the documents panel if they want RAG search."
            )

        # Default destination: index. Queue the build so the paper becomes
        # searchable without the user clicking anything.
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="add_arxiv_paper",
                phase="indexing",
                message="Queuing index build...",
            ),
        )
        task_id = await _queue_index_build(scope_id, scope_type)

        if task_id:
            return (
                f"Successfully added arXiv paper to {scope} documents and "
                f"queued an index build (task {task_id}).\n"
                f"Title: {paper.title}\n"
                f"Authors: {authors_short}\n"
                f"Doc ID: {metadata.pdf_id}\n"
                "Indexing runs in the background (a few minutes for a long "
                "paper). Until it finishes you cannot quote the paper's body, "
                "so do not claim to have read it; once it is indexed, rag_query "
                "will find it. Do NOT ask the user to click 'Build Index' — no "
                "action is needed from them."
            )

        return (
            f"Successfully added arXiv paper to {scope} documents.\n"
            f"Title: {paper.title}\n"
            f"Authors: {authors_short}\n"
            f"Doc ID: {metadata.pdf_id}\n"
            "It is not indexed yet and not attached to the context, so you "
            "cannot quote its body. You can start indexing yourself with the "
            "build_index tool; do not ask the user to click anything."
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
            "Indexing is the default destination: the paper is stored and an "
            "index build is queued automatically, so never ask the user to "
            "click 'Build Index'. The approval card also offers 'Add directly "
            "to context', which pastes the paper's text into the conversation "
            "instead of indexing it (large-context models only). Use "
            "destination='context' when the user explicitly asks for that."
        ),
        fn_schema=AddArxivPaperInput,
    )


def create_build_index_tool(
    progress_emitter: ProgressEmitter,
    confirmation_service: "ToolConfirmationService",
    session_id: str,
    project_id: Optional[str] = None,
) -> FunctionTool:
    """Create a build_index FunctionTool for the orchestrator.

    Builds the vector index for the session's (or project's) uploaded
    documents — the same incremental job the "Build Index" button in the
    Documents panel triggers. It reuses ``_build_index`` from the documents
    REST route, so config resolution, task submission and RAG cache eviction
    are shared with the UI path.

    The tool asks for confirmation first (it loads the embedding model and can
    run for minutes), then blocks while polling the TaskRunner and emits
    machine-readable progress for the frontend progress bar.
    """

    async def build_index(scope: str = "session") -> str:
        """Build the vector index for uploaded session or project documents.

        Use this when the user asks to index, re-index, or make their uploaded
        documents searchable. The user is asked to approve first; the tool
        blocks until the build finishes and reports what was indexed.

        Args:
            scope: "session" (default) or "project".

        Returns:
            Summary of the build, or an error string.
        """
        import time

        from fastapi import HTTPException

        from tensortruth.api.deps import (
            get_document_service,
            get_project_service,
            get_task_runner,
        )
        from tensortruth.api.routes.documents import _build_index
        from tensortruth.services.task_runner import TaskStatus

        # Validate scope
        if scope not in ("session", "project"):
            return f"Error: scope must be 'session' or 'project', got '{scope}'"

        if scope == "project" and not project_id:
            return (
                "Error: cannot build a project index — this session is not part "
                "of a project. Use scope='session' instead."
            )

        assert (scope != "project") or (project_id is not None)
        scope_id: str = project_id if scope == "project" else session_id  # type: ignore[assignment]
        scope_type = scope

        # Pre-flight: surface "nothing to do" cases as friendly strings before
        # asking the user to approve anything.
        try:

            def _preflight():
                # Directory walks + metadata reads on a worker thread.
                with get_document_service(scope_id, scope_type) as doc_service:
                    return (
                        doc_service.get_pdf_count(),
                        len(doc_service.get_all_markdown_files()),
                        doc_service.get_unindexed_doc_ids(),
                    )

            pdf_count, md_count, unindexed = await asyncio.to_thread(_preflight)
        except HTTPException as e:
            return f"Could not inspect documents: {e.detail}"
        except Exception as e:
            logger.error("build_index: failed to inspect %s scope: %s", scope, e)
            return f"Error: could not inspect {scope} documents — {e}"

        if pdf_count == 0 and md_count == 0:
            return (
                f"There are no documents in this {scope} to index. "
                "Add a PDF, text, or URL document first."
            )

        if not unindexed:
            return (
                f"All {pdf_count + md_count} document(s) in this {scope} are "
                "already indexed — nothing to build. New uploads are picked up "
                "by the next build."
            )

        doc_word = "document" if len(unindexed) == 1 else "documents"
        outcome = await request_confirmation(
            confirmation_service=confirmation_service,
            progress_emitter=progress_emitter,
            tool_name="build_index",
            action_type="build_index",
            title=f"Build index for {len(unindexed)} {scope} {doc_word}",
            summary=(
                f"Convert and embed {len(unindexed)} {doc_word} in the {scope} "
                "so they can be searched. This loads the embedding model and "
                "may take several minutes."
            ),
            details={
                "scope": scope,
                "scope_id": scope_id,
                "document_count": len(unindexed),
            },
            session_id=session_id,
        )

        if outcome == "rejected":
            return "User rejected the proposal to build the index."
        if outcome == "expired":
            return "Confirmation timed out for building the index."

        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="build_index",
                phase="starting",
                message="Starting index build...",
                metadata={
                    "percent": 0,
                    "message": "Starting index build...",
                    "stage": "",
                    "scope": scope,
                    "tool": "build_index",
                },
            ),
        )

        project_service = get_project_service() if scope == "project" else None
        task_runner = get_task_runner()

        # Clear the RAG engine cache once the task settles so subsequent
        # rag_query calls see the newly built index.
        try:
            from tensortruth.api.routes.index_management import _make_on_complete

            on_complete = _make_on_complete()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("build_index: no RAG cache eviction callback: %s", e)
            on_complete = None

        try:
            response = await _build_index(
                scope_id,
                scope_type,
                task_runner,
                project_service,
                on_complete=on_complete,
            )
        except HTTPException as e:
            return f"Could not start the index build: {e.detail}"
        except Exception as e:
            logger.error("build_index: failed to submit build: %s", e)
            return f"Error: failed to start the index build — {e}"

        task_id = response.task_id
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        deadline = started + _BUILD_INDEX_MAX_WAIT_SECONDS

        last_stage: Optional[str] = None
        last_percent = -1

        while True:
            task_info = task_runner.get_task(task_id)
            if task_info is None:
                return (
                    f"Error: index build task {task_id} disappeared before "
                    "finishing."
                )

            if task_info.status in (TaskStatus.COMPLETED, TaskStatus.ERROR):
                break

            stage = task_info.stage or ""
            percent = int(task_info.progress or 0)
            if stage != last_stage or percent != last_percent:
                message, phase = _format_build_stage(stage)
                if not stage and percent == 0:
                    phase, message = "preparing", "Preparing index build..."
                await _emit(
                    progress_emitter,
                    ToolProgress(
                        tool_id="build_index",
                        phase=phase,
                        message=message,
                        metadata={
                            "percent": max(0, min(percent, 100)),
                            "message": message,
                            "stage": stage,
                            "scope": scope,
                            "scope_type": scope_type,
                            "task_id": task_id,
                            "tool": "build_index",
                        },
                    ),
                )
                last_stage, last_percent = stage, percent

            if loop.time() >= deadline:
                return (
                    f"Index build is still running in the background "
                    f"(task {task_id}, {last_percent}% — "
                    f"{_format_build_stage(last_stage or '')[0]}). It will keep "
                    "going; ask again later or check the Documents panel for "
                    "status."
                )

            await asyncio.sleep(_BUILD_INDEX_POLL_INTERVAL)

        elapsed = time.monotonic() - started

        if task_info.status == TaskStatus.ERROR:
            await _emit(
                progress_emitter,
                ToolProgress(
                    tool_id="build_index",
                    phase="failed",
                    message="Index build failed.",
                    metadata={
                        "percent": last_percent if last_percent >= 0 else 0,
                        "message": "Index build failed.",
                        "stage": task_info.stage or "",
                        "error": task_info.error,
                        "scope": scope,
                        "tool": "build_index",
                    },
                ),
            )
            return f"Error: index build failed — {task_info.error or 'unknown error'}"

        result = task_info.result or {}
        indexed = result.get("documents_indexed", len(unindexed))
        chunks = result.get("chunks")

        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="build_index",
                phase="complete",
                message=f"Indexed {indexed} document(s).",
                metadata={
                    "percent": 100,
                    "message": f"Indexed {indexed} document(s).",
                    "stage": task_info.stage or "Complete",
                    "scope": scope,
                    "tool": "build_index",
                },
            ),
        )

        summary = (
            f"Index build complete for the {scope}. " f"Documents indexed: {indexed}. "
        )
        if isinstance(chunks, int) and chunks > 0:
            summary += f"Chunks in index: {chunks}. "
        summary += f"Elapsed: {elapsed:.1f}s. The documents are now searchable."
        return summary

    return FunctionTool.from_defaults(
        async_fn=build_index,
        name="build_index",
        description=(
            "Build the vector index for documents uploaded to the current "
            "session or its project. Use when the user asks to index, "
            "re-index, or make uploaded documents searchable via rag_query. "
            "The user is asked to approve first (it loads the embedding model); "
            "the tool blocks until the build finishes. "
            "scope='session' (default) or scope='project'."
        ),
        fn_schema=BuildIndexInput,
    )


_INSPECT_GIT_COMMANDS = ("log", "show", "blame")

_INSPECT_DESCRIPTION = (
    "Inspect a git repository cloned into this session. Read-only. This is NOT "
    "a shell: no pipes, redirects, shell expansion or arbitrary commands are "
    "accepted — only the fixed actions below. Actions: clone(url) · list · "
    "tree(repo, path, depth) · read(repo, path, start_line, max_lines) · "
    "grep(repo, pattern, path) · git(repo, git_command=log|show|blame, rev, "
    "path) · drop(repo)."
)


def create_inspect_tool(
    progress_emitter: ProgressEmitter,
    session_id: str,
    scope_dir: Optional["Path"] = None,
    confirmation_service: Optional["ToolConfirmationService"] = None,
    context_window: int = 16384,
    full_output_callback: Optional[FullOutputCallback] = None,
) -> FunctionTool:
    """Create the ``inspect`` repository-inspection FunctionTool.

    The tool exposes a *closed* set of read-only actions over git
    repositories cloned into the session's workspace directory. It is not a
    shell: ``action`` is a ``Literal`` enum, no argument is ever interpolated
    into a command string, and the workspace module enforces the path jail.
    All blocking work (clone, walk, grep, git subprocess) runs in a worker
    thread so the event loop keeps serving other requests.

    Args:
        progress_emitter: Callable that receives ToolProgress updates.
        session_id: Owning session; also used for confirmation cards.
        scope_dir: Override for the session workspace root (tests).
        confirmation_service: Gates the network+disk ``clone`` action when set.
        context_window: Session context window, used for the output budget.
        full_output_callback: Side-channel carrying the untruncated output.
    """

    async def inspect(
        action: str,
        url: Optional[str] = None,
        repo: Optional[str] = None,
        path: Optional[str] = None,
        pattern: Optional[str] = None,
        git_command: Optional[str] = None,
        rev: Optional[str] = None,
        depth: int = 3,
        start_line: int = 1,
        max_lines: int = 200,
        case_sensitive: bool = False,
        full_history: bool = False,
    ) -> str:
        """Inspect a git repository cloned into this session (read-only).

        ``action`` is one of: list, clone, tree, read, grep, git, drop.
        Returns file listings, file contents, matches or git output as text.
        There is no shell access and no way to write inside the clone.
        """
        from tensortruth.app_utils.paths import get_session_dir
        from tensortruth.services import repo_workspace as rw

        workdir = scope_dir if scope_dir is not None else get_session_dir(session_id)
        budget = _tool_content_budget(context_window, REPO_CONTENT_PCT)

        def _deliver(text: str) -> str:
            if full_output_callback is not None:
                full_output_callback("inspect", text)
            body, _truncated = _fit_tool_text(text, budget)
            return body

        if action == "list":
            try:
                records = rw.list_repos(workdir)
            except rw.RepoError as exc:
                return f"Error: {exc}"
            if not records:
                return (
                    "No repositories are cloned in this session. Clone one first: "
                    'inspect(action="clone", url="https://github.com/owner/repo").'
                )
            lines = [f"{len(records)} repository(ies) cloned in this session:"]
            for rec in records:
                lines.append(
                    f"- {rec.name} @ {rec.short_sha} — branch {rec.branch}, "
                    f"cloned {_format_age(rec.cloned_at)}"
                )
            return _deliver("\n".join(lines))

        if action == "clone":
            if not url or not url.strip():
                return (
                    "Error: action='clone' requires url, "
                    'e.g. url="https://github.com/owner/repo".'
                )
            clean_url = url.strip()
            if confirmation_service is not None and session_id:
                outcome = await request_confirmation(
                    confirmation_service=confirmation_service,
                    progress_emitter=progress_emitter,
                    tool_name="inspect",
                    action_type="clone_repository",
                    title=f"Clone {clean_url}",
                    summary=(
                        f"Clone {clean_url} into this session's workspace and "
                        "inspect it read-only for the life of the session."
                        + (
                            " Full history."
                            if full_history
                            else " Shallow clone (depth 1)."
                        )
                    ),
                    details={
                        "url": clean_url,
                        "full_history": full_history,
                        "scope": "session",
                        "alternatives": (
                            []
                            if full_history
                            else [
                                {
                                    "id": "full_history",
                                    "label": "Clone full history",
                                    "description": (
                                        "Fetch every commit instead of depth 1; "
                                        "needed to inspect history with "
                                        "git log/blame."
                                    ),
                                }
                            ]
                        ),
                    },
                    session_id=session_id,
                )
                if outcome == "rejected":
                    return f"User rejected cloning {clean_url}."
                if outcome == "full_history":
                    full_history = True

            await _emit(
                progress_emitter,
                ToolProgress(
                    tool_id="inspect",
                    phase="cloning",
                    message=f"Cloning {clean_url}...",
                    metadata={"url": clean_url},
                ),
            )
            try:
                record = await asyncio.to_thread(
                    rw.clone_repo, workdir, clean_url, full_history=full_history
                )
            except rw.RepoError as exc:
                return f"Error: {exc}"
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("inspect clone failed for %s", clean_url)
                return f"Error: could not clone {clean_url}: {exc}"
            return _deliver(
                f"Cloned {record.name} @ {record.short_sha} "
                f"(branch {record.branch}) into this session's workspace.\n"
                "It stays available for the life of the session — inspect it "
                'with inspect(action="tree"), inspect(action="read", path=...), '
                'inspect(action="grep", pattern=...) or '
                'inspect(action="git", git_command="log").'
            )

        if action == "drop":
            try:
                record = await asyncio.to_thread(rw.drop_repo, workdir, repo or "")
            except rw.RepoError as exc:
                return f"Error: {exc}"
            return f"Dropped {record.name} {record.short_sha} from this session."

        if action not in ("tree", "read", "grep", "git"):
            # Unreachable through the schema; kept for direct callers.
            return (
                f"Error: unknown action {action!r}. Valid actions: "
                "list, clone, tree, read, grep, git, drop."
            )

        try:
            record, root = await asyncio.to_thread(rw.find_repo, workdir, repo)
        except rw.RepoError as exc:
            return f"Error: {exc}"

        rel_path = (path or "").strip()
        phase = {"tree": "listing", "read": "reading", "grep": "searching"}.get(
            action, "inspecting"
        )
        await _emit(
            progress_emitter,
            ToolProgress(
                tool_id="inspect",
                phase=phase,
                message=f"{phase.capitalize()} {record.name} {rel_path or '/'}...",
                metadata={"repo": record.name, "path": rel_path},
            ),
        )

        try:
            if action == "tree":
                text = await asyncio.to_thread(
                    rw.tree,
                    record,
                    root,
                    rel_path,
                    max(1, min(int(depth or 3), 6)),
                )
            elif action == "read":
                text = await asyncio.to_thread(
                    rw.read_text,
                    record,
                    root,
                    rel_path,
                    max(1, int(start_line or 1)),
                    max(1, min(int(max_lines or 200), 2000)),
                )
            elif action == "grep":
                if not pattern:
                    return "Error: action='grep' requires pattern (a Python regex)."
                text = await asyncio.to_thread(
                    rw.grep,
                    record,
                    root,
                    pattern,
                    rel_path,
                    case_sensitive=bool(case_sensitive),
                )
            else:
                if git_command not in _INSPECT_GIT_COMMANDS:
                    return (
                        "Error: action='git' requires git_command, one of: "
                        + ", ".join(_INSPECT_GIT_COMMANDS)
                        + "."
                    )
                text = await asyncio.to_thread(
                    rw.git,
                    record,
                    root,
                    git_command,
                    path=rel_path,
                    rev=(rev or "HEAD"),
                    max_lines=max(1, min(int(max_lines or 200), 2000)),
                )
        except rw.RepoError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("inspect %s failed for %s", action, record.name)
            return f"Error: {action} failed on {record.name}: {exc}"

        return _deliver(text)

    return FunctionTool.from_defaults(
        async_fn=inspect,
        name="inspect",
        description=_INSPECT_DESCRIPTION,
        fn_schema=InspectInput,
    )


def _format_age(cloned_at: float) -> str:
    """Render a clone timestamp as a coarse age for progress narration."""
    import time

    try:
        seconds = max(0.0, time.time() - float(cloned_at))
    except (TypeError, ValueError):
        return "recently"
    if seconds < 120:
        return "just now"
    if seconds < 7200:
        return f"{int(seconds // 60)} min ago"
    if seconds < 172800:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"


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
    module_catalog: Optional[List[Any]] = None,
    loaded_modules: Optional[List[str]] = None,
    additional_index_paths: Optional[List[str]] = None,
    module_mode: str = "auto",
) -> List[FunctionTool]:
    """Create all orchestrator tool wrappers at once.

    Convenience function that builds the full set of wrapped tools for
    the orchestrator, including the RAG query tool when a RAGService with
    a loaded engine is available, and MCP management tools when the
    confirmation and server services are provided.

    Returns:
        List of FunctionTool wrappers. Always includes web_search, fetch_page,
        fetch_pages_batch. Includes rag_query whenever a rag_service is
        provided (loaded or not). Includes load_modules only in auto mode:
        manual mode pins the session to the user's selection.
        Includes MCP management tools when services are provided.
    """
    tools: List[FunctionTool] = []
    normalized_mode = module_mode or "auto"

    # RAG tools. rag_query is registered whenever a service exists, including
    # before anything is loaded. load_modules only exists in auto mode, where
    # it is exactly what turns an unloaded engine on.
    if rag_service is not None:
        catalog_names = [entry.name for entry in (module_catalog or [])]
        tools.append(
            create_rag_tool(
                rag_service,
                progress_emitter,
                session_params=session_params,
                session_messages=session_messages,
                rag_result_callback=rag_result_callback,
                full_output_callback=full_output_callback,
                context_window=context_window,
                catalog_names=catalog_names,
                session_id=session_id,
                module_mode=normalized_mode,
            )
        )
        # Context expansion tools read the docstore hierarchy that indexing
        # already persisted, so they cost no retrieval. They are registered
        # separately and guarded: a failure here must not cost the session its
        # rag_query tool.
        try:
            tools.extend(
                create_chunk_context_tools(
                    rag_service=rag_service, session_id=session_id
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Chunk context tools unavailable: %s", exc)
        if normalized_mode == "auto":
            tools.append(
                create_load_modules_tool(
                    rag_service,
                    progress_emitter,
                    module_catalog=module_catalog,
                    loaded_modules=loaded_modules,
                    session_params=session_params,
                    additional_index_paths=additional_index_paths,
                    session_id=session_id,
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
                context_window=context_window,
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

    # Build-index tool (needs session_id + confirmation_service)
    if session_id is not None and confirmation_service is not None:
        tools.append(
            create_build_index_tool(
                progress_emitter, confirmation_service, session_id, project_id
            )
        )

    # Repository inspection (needs session_id: the workspace is session-scoped)
    if session_id is not None:
        tools.append(
            create_inspect_tool(
                progress_emitter,
                session_id,
                confirmation_service=confirmation_service,
                context_window=context_window,
                full_output_callback=full_output_callback,
            )
        )

    return tools
