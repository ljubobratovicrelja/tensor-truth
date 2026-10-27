"""Orchestrator service built on LlamaIndex FunctionAgent.

Executes a single-turn agentic loop: creates a FunctionAgent, streams its
output directly as token events, and auto-continues if tools are requested.
No separate synthesis phase — the agent's own output IS the final answer.

The FunctionAgent handles the agentic loop internally (call tool -> inspect
result -> decide -> repeat or produce final answer).

This is the core of the agentic chat loop.  It does NOT maintain conversation
memory -- history is passed per-call from ChatHistoryService.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
)

from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentStream,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.agent.workflow.function_agent import (
    FunctionAgent as LIFunctionAgent,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow.errors import WorkflowRuntimeError

from tensortruth.agents.tool_output import describe_tool_call, extract_tool_text
from tensortruth.core.prompts import current_date_context
from tensortruth.core.providers import (
    get_orchestrator_llm as _providers_get_orchestrator_llm,
)
from tensortruth.core.providers import (
    resolve_model_from_params,
)
from tensortruth.services.context_documents import (
    CONTEXT_BUDGET_PCT,
    MIN_HISTORY_PCT,
    build_context_block,
)
from tensortruth.services.models import RAGRetrievalResult, ToolProgress
from tensortruth.services.orchestrator_tool_wrappers import (
    ProgressEmitter,
    create_all_tool_wrappers,
)

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

    from tensortruth.services.rag_service import RAGService
    from tensortruth.services.tool_service import ToolService

logger = logging.getLogger(__name__)

# Token-to-character conversion constant (same as core/synthesis.py)
CHARS_PER_TOKEN = 4

# Context window budget allocation (percentage of total context window)
SYSTEM_PROMPT_PCT = 0.12
CHAT_HISTORY_PCT = 0.18
USER_PROMPT_PCT = 0.18
RESPONSE_BUFFER_PCT = 0.50  # Must leave room for LLM output + tool results

# Built-in tool names that are already wrapped by orchestrator_tool_wrappers.
# These must be filtered out of ToolService.tools to avoid duplicates.
_WRAPPED_BUILTIN_TOOL_NAMES = frozenset(
    {
        "search_web",
        "fetch_page",
        "fetch_pages_batch",
        "search_focused",
        "add_arxiv_paper",
        "build_index",
    }
)

# Default maximum agentic iterations before the LLM's last response is used.
DEFAULT_MAX_ITERATIONS = 10

# Context-aware messages for the post-tool inference gap.
_TOOL_ANALYZING_MESSAGES: Dict[str, str] = {
    "web_search": "Analyzing search results...",
    "fetch_page": "Analyzing page content...",
    "fetch_pages_batch": "Analyzing fetched pages...",
    "rag_query": "Analyzing knowledge base results...",
    "add_arxiv_paper": "Adding paper to documents...",
}
_DEFAULT_ANALYZING_MESSAGE = "Analyzing results..."


# Module-level agent cache (Phase 2: Stateful Agent Cache).
# Maps identity tuple -> OrchestratorService instance.
_agent_cache: "OrderedDict[Tuple, OrchestratorService]" = OrderedDict()
_AGENT_CACHE_MAX_SIZE = 16

# Simple prompt heuristic keywords (Phase 3 Lever 1).
# Prompts containing any of these are considered likely to need tools.
_TOOL_KEYWORDS = frozenset(
    {
        "search",
        "find",
        "what",
        "when",
        "where",
        "who",
        "lookup",
        "fetch",
        "latest",
        "recent",
        "current",
        "compare",
        "research",
        "analyze",
        "investigate",
        "check",
        "look up",
        "summarize",
        "translate",
        "calculate",
    }
)
_SIMPLE_PROMPT_MAX_LENGTH = 80


def clear_agent_cache() -> None:
    """Clear the agent cache (for testing / reset)."""
    _agent_cache.clear()


def _compute_cache_identity(
    session_id: str,
    tool_service: "ToolService",
    rag_service: "RAGService",
    model: str,
    base_url: str,
    context_window: int,
    session_params: Optional[Dict[str, Any]],
    module_descriptions: Optional[List["ModuleDescription"]],
    custom_instructions: Optional[str],
    project_metadata: Optional[str],
    max_iterations: int,
    mcp_server_service: Optional[Any],
) -> tuple:
    """Compute a hashable identity tuple for the agent cache.

    The identity captures every input that affects the constructed tool set,
    LLM config, and system prompt.  When any of these changes a cache miss
    occurs and a new instance is created.
    """
    # Wrapped built-in tool names depend on RAG availability
    wrapped_names = {"web_search", "fetch_page", "fetch_pages_batch"}
    if rag_service and rag_service.is_loaded():
        wrapped_names.add("rag_query")
    # MCP management tools
    if mcp_server_service is not None:
        wrapped_names.update(
            {"list_mcp_servers", "get_mcp_presets", "manage_mcp_server"}
        )
    # MCP tools from ToolService (filter out wrapped built-ins)
    mcp_names = frozenset(
        t.metadata.name
        for t in tool_service.tools
        if t.metadata.name not in _WRAPPED_BUILTIN_TOOL_NAMES
    )
    all_tool_names = frozenset(wrapped_names) | mcp_names

    # System-prompt-relevant inputs hash
    descs = tuple(
        sorted(
            (m.name, m.display_name, m.doc_type) for m in (module_descriptions or [])
        )
    )
    sp_hash = hash(
        (
            descs,
            custom_instructions,
            project_metadata,
            (session_params or {}).get("provider_id"),
            # Think level changes the LLM (reasoning_effort), so it must
            # invalidate the cached agent instance.
            (session_params or {}).get("thinking"),
            max_iterations,
        )
    )

    return (
        session_id or "",
        model,
        base_url,
        context_window,
        all_tool_names,
        sp_hash,
    )


def _analyzing_message(tool_name: str) -> str:
    """Return a context-aware message for the post-tool inference phase."""
    if tool_name in _TOOL_ANALYZING_MESSAGES:
        return _TOOL_ANALYZING_MESSAGES[tool_name]
    readable = tool_name.replace("_", " ").replace("-", " ")
    return f"Analyzing {readable} results..."


def _is_transient_llm_error(exc: Exception) -> bool:
    """Check if an exception is likely a transient Ollama/LLM error."""
    msg = str(exc).lower()
    transient_indicators = [
        "failed to parse json",
        "connection refused",
        "connection reset",
        "status code: 500",
        "status code: 502",
        "status code: 503",
        "timed out",
        "timeout",
        "server disconnected",
        "broken pipe",
    ]
    return any(indicator in msg for indicator in transient_indicators)


@dataclass
class OrchestratorEvent:
    """A single event emitted by the orchestrator during execution.

    Downstream consumers (WebSocket handler) translate these into protocol
    messages (``token``, ``tool_progress``, ``tool_phase``, etc.).

    Exactly one of the content fields will be set per event.
    """

    # Text token delta from the LLM's final response
    token: Optional[str] = None

    # Thinking/reasoning token delta from the synthesis LLM
    thinking: Optional[str] = None

    # Tool call started
    tool_call: Optional[Dict[str, Any]] = None

    # Tool call completed
    tool_call_result: Optional[Dict[str, Any]] = None

    # Intermediate reasoning delta from the FunctionAgent between tool calls
    reasoning: Optional[str] = None

    # Text the FunctionAgent emitted while deciding to call a tool. This is
    # narration, not the answer: it is streamed live as tokens before the tool
    # call is known, then reclassified here so consumers can keep it out of the
    # final response.
    narration: Optional[str] = None

    # Phase-level progress from a tool wrapper (ToolProgress)
    tool_phase: Optional[ToolProgress] = None


@dataclass
class ModuleDescription:
    """Lightweight description of an indexed knowledge module."""

    name: str
    display_name: str
    doc_type: str


class OrchestratorService:
    """Agentic orchestrator built on LlamaIndex FunctionAgent.

    Creates a FunctionAgent with the orchestrator LLM + tool set + system
    prompt, then streams events during execution. Does NOT maintain
    conversation memory -- history is passed per-call.
    """

    def __init__(
        self,
        tool_service: "ToolService",
        rag_service: "RAGService",
        model: str,
        base_url: str,
        context_window: int,
        session_params: Optional[Dict[str, Any]] = None,
        session_messages: Optional[List[Dict[str, Any]]] = None,
        module_descriptions: Optional[List[ModuleDescription]] = None,
        custom_instructions: Optional[str] = None,
        project_metadata: Optional[str] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        confirmation_service: Optional[Any] = None,
        mcp_server_service: Optional[Any] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """Initialize the orchestrator service.

        Args:
            tool_service: ToolService instance with loaded tools.
            rag_service: RAGService instance (may or may not have engine loaded).
            model: Ollama model name for the orchestrator LLM.
            base_url: Ollama server base URL.
            context_window: Context window size in tokens for the model.
            session_params: Session engine parameters forwarded to RAG tool.
            session_messages: Chat history forwarded to RAG tool for query condensation.
            module_descriptions: Descriptions of active indexed modules for the system prompt.
            custom_instructions: Session-level custom instructions to include in system prompt.
            project_metadata: Project-level metadata string (name, description, instructions).
            max_iterations: Maximum agentic iterations before stopping.
            confirmation_service: Optional ToolConfirmationService for tool confirmations.
            mcp_server_service: Optional MCPServerService for MCP management tools.
            session_id: Session ID for MCP proposal tracking.
            project_id: Project ID for project-scoped operations (e.g. add_arxiv_paper).
        """
        self._tool_service = tool_service
        self._rag_service = rag_service
        self._model = model
        self._base_url = base_url
        self._context_window = context_window
        self._session_params = session_params or {}

        # Resolve model reference once for provider-aware LLM creation
        self._model_ref = resolve_model_from_params(self._session_params, model)
        self._session_messages: List[Dict[str, Any]] = session_messages or []
        self._module_descriptions = module_descriptions or []
        self._custom_instructions = custom_instructions
        self._project_metadata = project_metadata
        self._max_iterations = max_iterations
        self._confirmation_service = confirmation_service
        self._mcp_server_service = mcp_server_service
        self._session_id = session_id
        self._project_id = project_id

        # Built lazily on first execute()
        self._llm: Optional["LLM"] = None
        self._agent: Optional[LIFunctionAgent] = None
        self._tools: Optional[List[FunctionTool]] = None

        # Per-call external progress emitter (updated each execute())
        # Tool closures read this dynamically via the internal emitter.
        self._current_external_emitter: Optional[ProgressEmitter] = None

        # Tracks the last system prompt used to build _agent, so we know
        # when to rebuild (e.g. on date change or config change).
        self._last_system_prompt: Optional[str] = None

        # Accumulator for tool phase events emitted by tool wrappers.
        # Populated during execute() via the progress emitter closure.
        self._pending_phases: List[ToolProgress] = []

        # Last RAGRetrievalResult from rag_query tool execution.
        # Populated via the rag_result_callback during execute().
        # Used by the stream translator for proper source extraction.
        self._last_rag_result: Optional[RAGRetrievalResult] = None

        # Web source state — shared between web_search and fetch_pages_batch
        # tools via closures. web_search stores query + results; fetch_pages_batch
        # reads them for metadata and passes SourceNodes back via callback.
        self._last_web_sources: Optional[List] = None
        self._last_web_query: Optional[str] = None
        self._last_web_search_results: Optional[List[Dict[str, Any]]] = None

        # Tracks all URLs fetched in this execution to exclude from link discovery
        self._fetched_urls: set[str] = set()

        # Accumulator for full tool output (used by built-in tools via callback).
        # The scratchpad receives only summaries; full output is popped here
        # for tool_results_context which feeds the synthesizer.
        self._full_output_queue: List[tuple] = []

    # ------------------------------------------------------------------
    # Cache-aware factory (Phase 2)
    # ------------------------------------------------------------------

    @classmethod
    def get_or_create(
        cls, session_id: str = "", **kwargs: Any
    ) -> "OrchestratorService":
        """Get or create a cached OrchestratorService instance.

        The cache key is a hashable identity tuple capturing session_id,
        model, base_url, context_window, the tool name set, and a hash of
        system-prompt-relevant inputs (module descriptions, custom instructions,
        project metadata, provider_id, max_iterations).

        On cache hit the per-call ``session_messages`` are synchronised to
        the cached instance so that the RAG tool's query condensation has
        the latest conversation history.

        The cache is capped at ``_AGENT_CACHE_MAX_SIZE`` (16) entries by
        LRU-style eviction.
        """
        identity = _compute_cache_identity(
            session_id=session_id,
            tool_service=cast("ToolService", kwargs.get("tool_service")),
            rag_service=cast("RAGService", kwargs.get("rag_service")),
            model=kwargs.get("model") or "",
            base_url=kwargs.get("base_url") or "",
            context_window=kwargs.get("context_window") or 16384,
            session_params=kwargs.get("session_params"),
            module_descriptions=kwargs.get("module_descriptions"),
            custom_instructions=kwargs.get("custom_instructions"),
            project_metadata=kwargs.get("project_metadata"),
            max_iterations=kwargs.get("max_iterations", DEFAULT_MAX_ITERATIONS),
            mcp_server_service=kwargs.get("mcp_server_service"),
        )

        if identity in _agent_cache:
            _agent_cache.move_to_end(identity)
            cached = _agent_cache[identity]
            # Sync per-call mutable state (session messages grow each turn).
            new_msgs = kwargs.get("session_messages") or []
            cached._session_messages.clear()
            cached._session_messages.extend(new_msgs)
            logger.debug("Agent cache HIT for session %s", session_id)
            return cached

        instance = cls(session_id=session_id, **kwargs)
        _agent_cache[identity] = instance
        if len(_agent_cache) > _AGENT_CACHE_MAX_SIZE:
            _agent_cache.popitem(last=False)
        logger.debug(
            "Agent cache MISS for session %s (cache size=%d)",
            session_id,
            len(_agent_cache),
        )
        return instance

    # ------------------------------------------------------------------
    # Tool assembly
    # ------------------------------------------------------------------

    def _make_internal_emitter(self) -> ProgressEmitter:
        """Build the progress emitter handed to every wrapped tool.

        Tool wrappers are built once and cached, so the emitter resolves the
        active external emitter lazily on each call.

        Delivery is mutually exclusive:
        - external emitter attached (WebSocket streaming): forward the phase
          directly. Do NOT also queue it — the queue is drained after the next
          ``ToolCall``/``ToolCallResult`` and yielded as an event, so the
          client would receive the same phase twice. For
          ``confirmation_request`` phases that duplicate is fatal: two approval
          cards share one ``confirmation_id``, the second Approve fails with
          "already approved", and the tool blocks until its 300s timeout.
        - no external emitter: queue the phase so it is drained into the
          orchestrator event stream.
        """

        def _internal_emitter(tp: ToolProgress) -> None:
            if self._current_external_emitter is not None:
                self._current_external_emitter(tp)
                return
            self._pending_phases.append(tp)

        return _internal_emitter

    def _record_static_overhead(self, tools: List[FunctionTool]) -> None:
        """Measure the fixed per-turn cost (system prompt + tool schemas).

        Recorded once per orchestrator build so the context meter has real
        overhead numbers even before the first tool-capable turn.
        """
        try:
            from tensortruth.services.context_accounting import (
                record_static_overhead,
                tool_schema_chars,
            )

            with_modules = self._build_system_prompt(tools)
            without_modules = self._build_system_prompt(tools, include_modules=False)
            modules_chars = max(0, len(with_modules) - len(without_modules))
            record_static_overhead(
                {
                    "system": len(without_modules),
                    "tools": tool_schema_chars(tools),
                    "modules": modules_chars,
                }
            )
        except Exception:  # pragma: no cover - accounting must never break a turn
            logger.debug("Static context accounting failed", exc_info=True)

    def _build_tools(self) -> List[FunctionTool]:
        """Build the complete tool set for the orchestrator.

        Combines:
        1. Wrapped built-in tools (rag_query, web_search, fetch_page, etc.)
        2. MCP tools from ToolService (pass-through, already FunctionTool instances)

        Built-in ToolService tools that are already wrapped are filtered out
        to avoid duplicates.

        The internal emitter either forwards to ``_current_external_emitter``
        (set per execute() call) or, when no external emitter is attached,
        queues phases in ``_pending_phases`` for draining into the event
        stream. Tools can be built once and reused — the emitter reference
        resolves lazily.

        Returns:
            Complete list of FunctionTool instances.
        """

        # Internal emitter — reads self._current_external_emitter dynamically
        # so tool closures work correctly across execute() calls.
        _internal_emitter = self._make_internal_emitter()

        # RAG result callback stores the raw retrieval result for the
        # stream translator to extract proper sources and metrics.
        def _rag_result_cb(result: RAGRetrievalResult) -> None:
            self._last_rag_result = result

        # Web source closures — shared state between web_search and
        # fetch_pages_batch tools.
        def _web_query_setter(q: str) -> None:
            self._last_web_query = q

        def _web_query_getter() -> Optional[str]:
            return self._last_web_query

        def _web_results_setter(r: List[Dict[str, Any]]) -> None:
            self._last_web_search_results = r

        def _web_results_getter() -> Optional[List[Dict[str, Any]]]:
            return self._last_web_search_results

        def _web_result_cb(nodes: list) -> None:
            if self._last_web_sources is None:
                self._last_web_sources = []
            self._last_web_sources.extend(nodes)

        def _fetched_urls_getter() -> set:
            return self._fetched_urls

        def _fetched_urls_updater(urls: List[str]) -> None:
            self._fetched_urls.update(urls)

        # Full output callback — stores full tool output for synthesizer
        def _full_output_cb(tool_name: str, full_output: str) -> None:
            self._full_output_queue.append((tool_name, full_output))

        # Resolve reranker config from session params
        reranker_model = self._session_params.get("reranker_model")
        reranker_device = str(self._session_params.get("rag_device", "cpu"))

        # Load web search config for thresholds and reranking toggles
        try:
            from tensortruth.api.deps import get_config_service

            config = get_config_service().load()
            ws_config = config.web_search
            title_threshold = ws_config.rerank_title_threshold
            content_threshold = ws_config.rerank_content_threshold
            enable_title_reranking = ws_config.enable_title_reranking
            enable_content_reranking = ws_config.enable_content_reranking
        except Exception:
            title_threshold = 0.1
            content_threshold = 0.1
            enable_title_reranking = True
            enable_content_reranking = True

        # Compute per-stage reranker refs (None disables that stage)
        title_reranker = reranker_model if enable_title_reranking else None
        content_reranker = reranker_model if enable_content_reranking else None

        # 1. Create wrapped built-in tools (RAG + web tools with reranking)
        wrapped_tools = create_all_tool_wrappers(
            tool_service=self._tool_service,
            progress_emitter=_internal_emitter,
            rag_service=self._rag_service,
            session_params=self._session_params,
            session_messages=self._session_messages,
            rag_result_callback=_rag_result_cb,
            title_reranker_model=title_reranker,
            content_reranker_model=content_reranker,
            reranker_device=reranker_device,
            title_threshold=title_threshold,
            content_threshold=content_threshold,
            context_window=self._context_window,
            custom_instructions=self._custom_instructions,
            web_query_setter=_web_query_setter,
            web_query_getter=_web_query_getter,
            web_search_results_setter=_web_results_setter,
            web_search_results_getter=_web_results_getter,
            web_result_callback=_web_result_cb,
            fetched_urls_getter=_fetched_urls_getter,
            fetched_urls_updater=_fetched_urls_updater,
            full_output_callback=_full_output_cb,
            confirmation_service=self._confirmation_service,
            mcp_server_service=self._mcp_server_service,
            session_id=self._session_id,
            project_id=self._project_id,
        )

        # 2. Collect MCP tools from ToolService, filtering out already-wrapped built-ins
        mcp_tools = [
            t
            for t in self._tool_service.tools
            if t.metadata.name not in _WRAPPED_BUILTIN_TOOL_NAMES
        ]

        all_tools = wrapped_tools + mcp_tools

        tool_names = [t.metadata.name for t in all_tools]
        logger.info(
            "Orchestrator tool set: %d tools (%d wrapped, %d MCP): %s",
            len(all_tools),
            len(wrapped_tools),
            len(mcp_tools),
            tool_names,
        )

        # Log individual tool descriptions at debug level for diagnostics
        for t in all_tools:
            desc = t.metadata.description or "(no description)"
            logger.debug(
                "  Tool '%s': %s",
                t.metadata.name,
                desc[:120] + ("..." if len(desc) > 120 else ""),
            )

        # Phase 3 Lever 2 — slim_tool_schemas: truncate descriptions in
        # FunctionTool metadata (sent via the ``tools=`` LLM parameter).
        try:
            from tensortruth.api.deps import get_config_service

            _cfg = get_config_service().load()
            if _cfg.agent.slim_tool_schemas:
                slimmed = []
                from dataclasses import replace as _dataclass_replace

                for t in all_tools:
                    new_meta = _dataclass_replace(
                        t.metadata,
                        description=(t.metadata.description or "")[:120],
                    )
                    slimmed.append(
                        FunctionTool(
                            fn=t._fn,
                            async_fn=t._async_fn,
                            metadata=new_meta,
                        )
                    )
                all_tools = slimmed
        except Exception:
            pass

        return all_tools

    # ------------------------------------------------------------------
    # System prompt composition
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        tools: List[FunctionTool],
        in_context_block: str = "",
        include_modules: bool = True,
    ) -> str:
        """Compose the orchestrator system prompt.

        Includes:
        - Role description
        - Active indexed modules with descriptions
        - Tool usage guidance
        - Custom instructions / project metadata
        - Attached (direct-to-context) documents, when present
        - Explicit tool list to prevent hallucination

        Args:
            tools: The full tool set (used to list available tool names).
            in_context_block: Pre-rendered attached-documents block.

        Returns:
            Complete system prompt string.
        """
        sections: List[str] = []

        # --- Current date (temporal grounding) ---
        sections.append(current_date_context())

        # --- Role ---
        sections.append(
            "You are the assistant powering TensorTruth, a local-first RAG "
            "application for technical documentation and research papers. "
            "You help users by answering questions, finding information, and "
            "completing tasks. You have access to tools that let you search "
            "a knowledge base, search the web, fetch web pages, and more. "
            "Decide which tools to use based on the user's question."
        )

        # --- Indexed modules ---
        if include_modules and self._module_descriptions:
            module_lines = []
            for mod in self._module_descriptions:
                module_lines.append(
                    f"- {mod.name}: {mod.display_name} ({mod.doc_type})"
                )
            modules_block = "\n".join(module_lines)
            sections.append(
                "You have access to a knowledge base with the following indexed modules:\n"
                f"{modules_block}\n\n"
                "Use rag_query when the user's question likely relates to topics "
                "covered by these modules."
            )

        # --- Tool routing guidance ---
        sections.append(
            "Tool routing:\n"
            "- For questions about topics covered by the indexed modules, "
            "search the knowledge base. rag_query returns verbatim excerpts "
            "from the indexed documents — quote and cite them instead of "
            "answering from memory.\n"
            "- For current events, recent developments, or topics NOT in the "
            "indexed modules, search the web. CRITICAL: web_search returns "
            "ONLY titles and URLs — NO page content at all. You MUST call "
            "fetch_pages_batch with the relevant URLs after every web_search "
            "to retrieve actual content before answering.\n"
            "- For simple conversational messages (greetings, clarifications, "
            "opinions), respond directly without using any tools.\n"
            "- If a tool returns an error, analyze the cause: if the input was "
            "wrong, correct it and retry; if it is an internal error, report "
            "the issue and continue with other tools if possible.\n"
            "- To fetch multiple web pages, prefer fetch_pages_batch with a list of "
            "URLs over calling fetch_page multiple times.\n"
            '- After fetching pages, review the "Discovered links" section at the end of '
            "the results. If the fetched page content already answers the user's question "
            "well, proceed to your summary — do NOT follow links just because they exist. "
            "Only follow links when the fetched content is clearly insufficient and a "
            "discovered link looks like it would provide critical missing information. "
            "Be mindful of your iteration budget. Do NOT re-fetch URLs already fetched."
        )

        # --- MCP server management guidance ---
        mcp_mgmt_tools = {"list_mcp_servers", "get_mcp_presets", "manage_mcp_server"}
        has_mcp_mgmt = any(t.metadata.name in mcp_mgmt_tools for t in tools)
        if has_mcp_mgmt:
            sections.append(
                "MCP server management:\n"
                "- Use list_mcp_servers to see current MCP server configurations.\n"
                "- Use get_mcp_presets to check for known preset configurations "
                "(presets auto-fill command/args, so you only need name and summary).\n"
                "- For servers NOT in presets, you MUST first research the correct "
                "installation command before calling manage_mcp_server. Use web_search to find the "
                "server's npm package or GitHub repo, then fetch_page to read the "
                "README and find the exact command and args (usually `npx -y <package>` "
                "for npm-based servers). Only call manage_mcp_server once you have "
                "the concrete command and args.\n"
                "- Use manage_mcp_server to add, update, or remove "
                "a server. For 'add' with type 'stdio', you MUST provide `command` "
                "and `args` (e.g. command='npx', args=['-y', '<package>']). "
                "The user is prompted inline to approve; the tool blocks and "
                "applies the change automatically. Just report the result.\n"
                "- NEVER retry manage_mcp_server with the same arguments if it fails. "
                "Fix the issue first (e.g. research the correct command)."
            )

        # --- MCP tool routing ---
        _wrapped_builtin_names = _WRAPPED_BUILTIN_TOOL_NAMES | {
            "rag_query",
            "web_search",
            "fetch_page",
            "fetch_pages_batch",
        }
        mcp_tools = [
            t
            for t in tools
            if t.metadata.name not in _wrapped_builtin_names
            and t.metadata.name not in mcp_mgmt_tools
        ]
        if mcp_tools:
            _names = [t.metadata.name or "?" for t in mcp_tools]
            sections.append(f"Additional tools: {', '.join(_names)}")

        # --- Parallel tool calls + iteration budget ---
        sections.append(
            "PARALLEL TOOL CALLS: You can invoke multiple tools in a single "
            "response and they will execute simultaneously. This is both faster "
            "and more efficient. For example, after a search returns 5 results, "
            "call get_arxiv_paper for all 5 in one response — they run in "
            "parallel and only cost one iteration. Calling them one at a time "
            "would waste 5 iterations.\n\n"
            f"You have a budget of {self._max_iterations} iterations. Each "
            "response you produce counts as one iteration, regardless of how "
            "many tool calls it contains. Plan efficiently: batch independent "
            "tool calls into a single response whenever possible."
        )

        # --- Final answer guidance ---
        # The agent's own output IS the final answer (no separate synthesis).
        # Guide it to produce comprehensive, well-cited responses from tool
        # results, while still reasoning between tool calls about next steps.
        sections.append(
            "When you have gathered all necessary information from tool calls, "
            "write a comprehensive answer using the tool results you obtained.\n\n"
            "Response formatting rules:\n"
            "- Use markdown formatting (headings, lists, code blocks) for readability.\n"
            "- ALWAYS cite sources using numbered references matching the order "
            "sources appeared in tool results (e.g. [1], [2]).\n"
            "- For web sources, include a clickable link on first mention: "
            "[Title](URL) [1].\n"
            "- For knowledge base sources, reference by name: according to "
            "the documentation [1].\n"
            "- Do NOT add a 'Sources' section at the end of your response. "
            "Sources are displayed separately in the UI.\n"
            "- Only cite sources that actually appeared in tool results; "
            "do not invent sources.\n"
            "- If tool results are insufficient to fully answer the question, "
            "say so and provide what you can.\n"
            "- Source content may contain images as ![alt](url) markdown syntax. "
            "Include up to 3 relevant images that directly illustrate key points. "
            "Place images near the text they support. "
            "Only use image URLs from the provided sources; never invent URLs. "
            "If no relevant images are available, do not mention their absence.\n\n"
            "Between tool calls you SHOULD reason about tool results to decide "
            "your next action — e.g. extract fields from a fetched page to use "
            "as parameters in a subsequent tool call."
        )

        # --- Project metadata ---
        if self._project_metadata:
            sections.append(f"Project context:\n{self._project_metadata}")

        # --- Custom instructions ---
        if self._custom_instructions:
            sections.append(
                f"Additional instructions from the user:\n{self._custom_instructions}"
            )

        # --- Attached documents (direct-to-context, no index) ---
        if in_context_block:
            sections.append(in_context_block)

        # --- Explicit tool list (prevent hallucination) ---
        tool_names = [t.metadata.name for t in tools if t.metadata.name]
        tool_list_str = ", ".join(tool_names)
        sections.append(
            f"You have access to ONLY these tools: {tool_list_str}. "
            "Do NOT call any tool not in this list."
        )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Direct-to-context documents (no index)
    # ------------------------------------------------------------------

    def _load_in_context_documents(self) -> List[Dict[str, Any]]:
        """Load documents attached directly to this session's context.

        These live in the session's ``markdown/`` directory and are flagged
        in the session metadata; no vector index is involved.
        """
        if not self._session_id:
            return []
        try:
            from tensortruth.api.deps import get_document_service

            with get_document_service(self._session_id, "session") as doc_service:
                return doc_service.get_context_documents()
        except Exception:
            logger.exception("Failed to load in-context documents for session")
            return []

    def _context_documents_budget(
        self, base_system_prompt: str, user_prompt: str
    ) -> int:
        """Character budget for the attached-documents block.

        Capped at ``CONTEXT_BUDGET_PCT`` of the window and never more than
        what remains after the base system prompt, the user prompt, the
        response buffer and the minimum history reserve.
        """
        total_chars = self._context_window * CHARS_PER_TOKEN
        response_buffer = int(total_chars * RESPONSE_BUFFER_PCT)
        history_reserve = int(total_chars * MIN_HISTORY_PCT)
        available = (
            total_chars
            - len(base_system_prompt)
            - len(user_prompt)
            - response_buffer
            - history_reserve
        )
        cap = int(total_chars * CONTEXT_BUDGET_PCT)
        return max(0, min(cap, available))

    def _build_in_context_block(
        self,
        documents: List[Dict[str, Any]],
        base_system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Render the attached-documents block and log truncation decisions."""
        if not documents:
            return ""

        budget = self._context_documents_budget(base_system_prompt, user_prompt)
        block, report = build_context_block(documents, budget)

        for entry in report:
            if not entry["included"]:
                logger.warning(
                    "In-context document %s (%s) did not fit the budget "
                    "(%d chars, budget=%d) and was omitted from the prompt",
                    entry["doc_id"],
                    entry["filename"],
                    entry["chars"],
                    budget,
                )
            elif entry["truncated"]:
                logger.warning(
                    "In-context document %s (%s) truncated: kept %d of %d chars",
                    entry["doc_id"],
                    entry["filename"],
                    entry["kept_chars"],
                    entry["chars"],
                )

        return block

    # ------------------------------------------------------------------
    # Context window budgeting
    # ------------------------------------------------------------------

    def _budget_history(
        self,
        history_messages: List[Dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict[str, Any]]:
        """Truncate chat history to fit within the context window budget.

        Budget allocation (percentage of context_window):
        - System prompt: ~12% (non-negotiable)
        - Chat history: ~18%
        - User prompt: ~18% (essential)
        - Response buffer: ~50%

        If the system prompt + user prompt already exceed their combined budget,
        history is dropped entirely. History is trimmed by removing oldest
        turns first (each turn = 1 user + 1 assistant message).

        Args:
            history_messages: Session messages in ``[{"role": ..., "content": ...}]`` format.
            system_prompt: The composed system prompt.
            user_prompt: The current user prompt.

        Returns:
            Truncated list of history messages that fits the budget.
        """
        total_chars = self._context_window * CHARS_PER_TOKEN

        system_chars = len(system_prompt)
        user_chars = len(user_prompt)
        history_budget_chars = int(total_chars * CHAT_HISTORY_PCT)

        # If system prompt is very large, still honour history budget
        # but clamp to remaining space after system + user + response buffer
        response_buffer_chars = int(total_chars * RESPONSE_BUFFER_PCT)
        available_for_history = max(
            0, total_chars - system_chars - user_chars - response_buffer_chars
        )
        effective_budget = min(history_budget_chars, available_for_history)

        if effective_budget <= 0 or not history_messages:
            return []

        # Walk backward through messages, accumulating turns
        # A turn = user message + assistant response (2 messages)
        kept: List[Dict[str, Any]] = []
        chars_used = 0

        # Iterate from most recent to oldest
        for msg in reversed(history_messages):
            content = msg.get("content", "")
            msg_chars = len(str(content))
            if chars_used + msg_chars > effective_budget:
                break
            kept.append(msg)
            chars_used += msg_chars

        # Reverse back to chronological order
        kept.reverse()

        # Ensure we keep complete turns (don't orphan an assistant response)
        # If the first kept message is an assistant message, drop it
        if kept and kept[0].get("role") == "assistant":
            kept = kept[1:]

        logger.debug(
            "Context budget: total=%d chars, system=%d, user=%d, "
            "history_budget=%d, history_used=%d (%d messages kept of %d)",
            total_chars,
            system_chars,
            user_chars,
            effective_budget,
            chars_used,
            len(kept),
            len(history_messages),
        )

        return kept

    # ------------------------------------------------------------------
    # Context transform hook (Phase 5)
    # ------------------------------------------------------------------

    def _transform_context(
        self,
        history: List[Dict[str, Any]],
        prompt: str,
        tools: List[FunctionTool],
        in_context_block: str = "",
    ) -> List[Dict[str, Any]]:
        """Transform context before every LLM call.

        Applies, in order:
        1. **Small-history bypass** — if total history chars < 2000, pass
           through with only standard budgeting (skip compaction/floor).
        2. **Tool result compaction** — truncate tool output > 4000 chars
           to first 500 + ``...[truncated]``.
        3. **Budget history** — drop oldest messages when token count
           exceeds the context window budget (already walks newest first).
        4. **Recent-N floor** — always keep the most recent 6 messages
           regardless of budget.

        Returns:
            Truncated list of history messages.
        """
        # 1. Small-history bypass — return as-is (budgeting is unnecessary for
        # short histories and would waste compute).
        total_chars = sum(len(str(m.get("content", ""))) for m in history)
        if total_chars < 2000:
            return list(history)

        # 2. Tool result compaction
        for msg in history:
            if msg.get("role") == "assistant" and msg.get("tool_steps"):
                for step in msg["tool_steps"]:
                    output = step.get("output", "")
                    if len(output) > 4000:
                        step["output"] = output[:500] + "...[truncated]"

        # 3. Budget history
        system_prompt = self._build_system_prompt(
            tools, in_context_block=in_context_block
        )
        budgeted = self._budget_history(history, system_prompt, prompt)

        # 4. Recent-N floor — ensure at least 6 newest messages
        if len(history) >= 6:
            last_6 = history[-6:]
            budgeted_ids = {id(m) for m in budgeted}
            missing = [m for m in last_6 if id(m) not in budgeted_ids]
            if missing:
                # Merge: insert missing messages at their original positions
                result = list(budgeted)
                for m in missing:
                    try:
                        pos = history.index(m)
                    except ValueError:
                        continue
                    insert_idx = 0
                    for i, r in enumerate(result):
                        try:
                            if history.index(r) < pos:
                                insert_idx = i + 1
                        except ValueError:
                            pass
                    result.insert(insert_idx, m)
                budgeted = result
        elif len(history) > 0:
            budgeted = list(history)

        return budgeted

    # ------------------------------------------------------------------
    # No-tools fast path helpers (Phase 3 Lever 1)
    # ------------------------------------------------------------------

    def _is_simple_prompt(self, prompt: str) -> bool:
        """Heuristic: does this prompt look like it needs no tools?

        Returns True when the prompt is short and contains no tool-related
        keywords.  Used to gate the no-tools fast path.
        """
        text = prompt.lower().strip()
        if len(text) > _SIMPLE_PROMPT_MAX_LENGTH:
            return False
        for kw in _TOOL_KEYWORDS:
            if kw in text:
                return False
        return True

    async def _try_fast_path(self, prompt: str) -> Optional[str]:
        """Try the no-tools fast path.

        Calls the LLM with *no tools* and a tiny system prompt.  If the LLM
        responds with exactly ``NEED_TOOLS``, returns ``None`` (caller should
        fall through to the full agent).  Otherwise returns the LLM's text
        response.

        Returns:
            Response text if tools are not needed, or ``None`` if tools are
            needed (caller should proceed with full agent).
        """
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        # Ensure LLM is available
        if self._llm is None:
            self._llm = _providers_get_orchestrator_llm(
                self._model_ref,
                self._context_window,
                thinking=self._session_params.get("thinking"),
            )

        fast_system = (
            "Answer the user's message concisely. "
            "If you genuinely need external tools to answer, "
            "respond with exactly: NEED_TOOLS"
        )
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=fast_system),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        response = await self._llm.achat(messages)
        text = str((response.message.content or "")).strip()

        if text == "NEED_TOOLS":
            return None

        return text

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        progress_emitter: Optional[ProgressEmitter] = None,
        images: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[OrchestratorEvent, None]:
        """Execute the orchestrator for a user prompt.

        Creates (or reuses) the FunctionAgent, feeds it the user prompt
        with budgeted chat history, and yields events as they stream from
        the underlying LlamaIndex workflow.

        Args:
            prompt: The user's current message.
            chat_history: Session messages for context. Truncated to fit
                the context window budget.
            progress_emitter: Optional callback for forwarding ToolProgress
                events to the WebSocket layer. Also captured by tool wrappers.
            images: Optional list of image dicts with "data" (base64) and
                "mimetype" keys for multimodal input.

        Yields:
            OrchestratorEvent instances (tokens, tool calls, tool phases).
        """
        # Reset per-execution state
        self._pending_phases = []
        self._last_rag_result = None
        self._last_web_sources = None
        self._last_web_query = None
        self._last_web_search_results = None
        self._fetched_urls = set()
        self._full_output_queue = []
        self._current_external_emitter = progress_emitter

        # Emit boot phase so the frontend shows immediate feedback while
        # tools and LLM are being initialized.
        boot_phase = ToolProgress(
            tool_id="orchestrator",
            phase="booting",
            message="Starting orchestrator...",
        )
        yield OrchestratorEvent(tool_phase=boot_phase)
        if progress_emitter:
            progress_emitter(boot_phase)

        # Documents attached directly to the session context (no index).
        # Loaded per execution so adding/removing one takes effect next turn.
        in_context_documents = self._load_in_context_documents()

        # ---- Phase 3 Lever 1: No-tools fast path ----
        try:
            from tensortruth.api.deps import get_config_service as _get_cfg

            _cfg = _get_cfg().load()
            _use_fast = _cfg.agent.tool_subset_selection
        except Exception:
            _use_fast = False

        if (
            _use_fast
            and not images
            and not in_context_documents
            and self._is_simple_prompt(prompt)
        ):
            fast_text = await self._try_fast_path(prompt)
            if fast_text is not None:
                # The fast path sends no tools and no history; record what it
                # actually sent so the meter does not claim otherwise.
                try:
                    from tensortruth.services.context_accounting import (
                        get_static_overhead,
                        record_snapshot,
                    )

                    overhead = get_static_overhead()
                    record_snapshot(
                        self._session_id or "",
                        {
                            "system": overhead.get("system", 0),
                            "tools": 0,
                            "modules": overhead.get("modules", 0),
                            "documents": 0,
                            "history": 0,
                            "prompt": len(prompt),
                        },
                        context_window=self._context_window,
                        model_name=getattr(self._model_ref, "model_name", None),
                        messages_included=0,
                        messages_total=len(chat_history or []),
                    )
                except Exception:  # pragma: no cover
                    pass
                logger.info(
                    "Fast path: no-tools response for prompt (len=%d)", len(prompt)
                )
                yield OrchestratorEvent(token=fast_text)
                # Per-execution state is already reset; just return.
                if progress_emitter:
                    progress_emitter(
                        ToolProgress(
                            tool_id="orchestrator",
                            phase="done",
                            message="Complete",
                        )
                    )
                return
            logger.debug("Fast path: NEED_TOOLS, falling through to full agent")

        # --- Build tools and agent (cached across execute() calls) ---
        if self._tools is None:
            self._tools = self._build_tools()
            self._record_static_overhead(self._tools)
        tools = self._tools

        # Get (or reuse) orchestrator LLM singleton
        if self._llm is None:
            self._llm = _providers_get_orchestrator_llm(
                self._model_ref,
                self._context_window,
                thinking=self._session_params.get("thinking"),
            )

        # Compose system prompt; attached documents get their own section
        # under a hard char budget (see _context_documents_budget).
        base_system_prompt = self._build_system_prompt(tools)
        in_context_block = self._build_in_context_block(
            in_context_documents, base_system_prompt, prompt
        )
        system_prompt = self._build_system_prompt(
            tools, in_context_block=in_context_block
        )

        # Phase 5: context transform (budget + compaction + floor)
        budgeted_history = self._transform_context(
            chat_history or [],
            prompt,
            tools,
            in_context_block=in_context_block,
        )

        # Convert budgeted history to LlamaIndex ChatMessage objects
        llama_history = self._to_chat_messages(budgeted_history)

        # Estimate prompt token composition
        system_tokens = len(system_prompt) // CHARS_PER_TOKEN
        history_tokens = (
            sum(len(str(m.get("content", ""))) for m in budgeted_history)
            // CHARS_PER_TOKEN
        )
        user_tokens = len(prompt) // CHARS_PER_TOKEN
        from tensortruth.services.context_accounting import (
            record_snapshot,
        )
        from tensortruth.services.context_accounting import (
            tool_schema_chars as _tool_schema_chars,
        )

        tool_schema_chars = _tool_schema_chars(tools)

        # Per-category breakdown for the status bar and /compact decisions.
        # Measured, not estimated: these are the exact strings we send.
        try:
            prompt_without_modules = self._build_system_prompt(
                tools, in_context_block=in_context_block, include_modules=False
            )
            modules_chars = max(0, len(system_prompt) - len(prompt_without_modules))
            record_snapshot(
                self._session_id or "",
                {
                    "system": max(
                        0,
                        len(system_prompt) - len(in_context_block) - modules_chars,
                    ),
                    "tools": tool_schema_chars,
                    "modules": modules_chars,
                    "documents": len(in_context_block),
                    "history": sum(
                        len(str(m.get("content", ""))) for m in budgeted_history
                    ),
                    "prompt": len(prompt),
                },
                context_window=self._context_window,
                model_name=getattr(self._model_ref, "model_name", None),
                messages_included=len(budgeted_history),
                messages_total=len(chat_history or []),
            )
        except Exception:  # pragma: no cover - accounting must never break a turn
            logger.debug("Context accounting failed", exc_info=True)

        tool_tokens = tool_schema_chars // CHARS_PER_TOKEN
        total_tokens = system_tokens + tool_tokens + history_tokens + user_tokens
        logger.info(
            "Orchestrator prompt: ~%d tokens "
            "(sys=%d, tools=%d, hist=%d, user=%d) — %.0f%% of %d ctx",
            total_tokens,
            system_tokens,
            tool_tokens,
            history_tokens,
            user_tokens,
            total_tokens / self._context_window * 100,
            self._context_window,
        )

        # Create FunctionAgent (cached across calls; rebuild only when
        # system prompt changes e.g. on date rollover).
        if self._agent is None or self._last_system_prompt != system_prompt:
            self._agent = LIFunctionAgent(
                tools=list(tools),  # type: ignore[arg-type]
                llm=self._llm,
                system_prompt=system_prompt,
            )
            self._last_system_prompt = system_prompt

        # Emit thinking phase once agent is ready and about to run.
        thinking_phase = ToolProgress(
            tool_id="orchestrator",
            phase="thinking",
            message="Analyzing your request...",
        )
        yield OrchestratorEvent(tool_phase=thinking_phase)
        if progress_emitter:
            progress_emitter(thinking_phase)

        # --- Agentic loop (single-turn) ---
        tools_called: List[str] = []
        tool_steps: List[Dict[str, Any]] = []
        tool_results_context: List[str] = []
        agent_final_response = ""
        agent_deltas: List[str] = []
        # Text accumulated for the current LLM step. AgentOutput tells us
        # whether that step was narration (it requested a tool) or the final
        # answer, so this buffer is what lets us separate the two.
        step_text = ""

        max_attempts = 2

        for attempt in range(max_attempts):
            step_text = ""
            try:
                # Build user message — multimodal if images provided
                if images:
                    import base64 as _b64

                    from llama_index.core.base.llms.types import (
                        ImageBlock as _ImageBlock,
                    )
                    from llama_index.core.base.llms.types import TextBlock as _TextBlock

                    _blocks: list = [_TextBlock(text=prompt)]
                    for _img in images:
                        _blocks.append(
                            _ImageBlock(
                                image=_b64.b64decode(_img["data"]),
                                image_mimetype=_img["mimetype"],
                            )
                        )
                    user_msg = ChatMessage(role=MessageRole.USER, blocks=_blocks)
                else:
                    user_msg = prompt  # type: ignore[assignment]

                handler = self._agent.run(
                    user_msg=user_msg,
                    chat_history=llama_history if llama_history else None,
                    max_iterations=self._max_iterations,
                )

                async for event in handler.stream_events():
                    if isinstance(event, ToolCall):
                        tools_called.append(event.tool_name)

                        # A tool call proves the text streamed immediately before
                        # it was narration, not an answer. AgentOutput normally
                        # reports this first; this fallback keeps older/mocked
                        # event streams from leaking narration into the answer.
                        if step_text.strip():
                            yield OrchestratorEvent(narration=step_text)
                            step_text = ""

                        # Yield tool call event
                        yield OrchestratorEvent(
                            tool_call={
                                "tool": event.tool_name,
                                "params": event.tool_kwargs,
                                "tool_id": event.tool_id,
                            }
                        )

                        # For MCP tools (not wrapped built-ins), synthesize a
                        # ToolProgress event so users see what's happening
                        # instead of stale "Analyzing your request..."
                        if event.tool_name not in _WRAPPED_BUILTIN_TOOL_NAMES:
                            phase_msg = describe_tool_call(
                                event.tool_name, event.tool_kwargs
                            )
                            yield OrchestratorEvent(
                                tool_phase=ToolProgress(
                                    tool_id=event.tool_name,
                                    phase="tool_call",
                                    message=phase_msg,
                                    metadata={"tool": event.tool_name},
                                )
                            )

                        # Drain any pending tool phase events that were emitted
                        # by the tool wrapper's progress emitter
                        for phase in self._pending_phases:
                            yield OrchestratorEvent(tool_phase=phase)
                        self._pending_phases = []

                    elif isinstance(event, ToolCallResult):
                        # Built-in tools store full output via callback;
                        # the event now contains only a summary for the
                        # scratchpad. Pop matching entry from the queue.
                        full_output = None
                        for i, (name, text) in enumerate(self._full_output_queue):
                            if name == event.tool_name:
                                full_output = self._full_output_queue.pop(i)[1]
                                break
                        if full_output is None:
                            # MCP tools or fallback
                            full_output = extract_tool_text(event.tool_output)
                        output_text = full_output[:2000]
                        is_error = event.tool_output.is_error

                        # tool_steps (saved to DB): truncated for storage
                        tool_steps.append(
                            {
                                "tool": event.tool_name,
                                "params": event.tool_kwargs,
                                "output": output_text,
                                "is_error": is_error,
                            }
                        )

                        # Collect tool results for synthesis context (full output)
                        status = "ERROR" if is_error else "OK"
                        tool_results_context.append(
                            f"[{event.tool_name} ({status})]\n{full_output}"
                        )

                        # Include full_output for source extraction by stream
                        # translator; strip before storing in tool_steps
                        yield OrchestratorEvent(
                            tool_call_result={
                                "tool": event.tool_name,
                                "params": event.tool_kwargs,
                                "output": output_text,
                                "full_output": full_output,
                                "is_error": is_error,
                                "tool_id": event.tool_id,
                            }
                        )

                        # Drain any remaining tool phases
                        for phase in self._pending_phases:
                            yield OrchestratorEvent(tool_phase=phase)
                        self._pending_phases = []

                        # Emit transitional phase to cover the LLM inference
                        # gap between tool completion and next action. Without
                        # this, the last tool phase (e.g. "Fetching arxiv.org...")
                        # stays displayed while the orchestrator LLM processes
                        # the result — misleading because the fetch is already done.
                        yield OrchestratorEvent(
                            tool_phase=ToolProgress(
                                tool_id="orchestrator",
                                phase="analyzing",
                                message=_analyzing_message(event.tool_name),
                            )
                        )

                    elif isinstance(event, AgentOutput):
                        # One LLM step has finished. Its streamed text is either
                        # narration (this step requested tools) or the answer.
                        # Tokens were already sent live; this event lets the
                        # stream translator move the narration out of the final
                        # response and lets the UI render it as separate blocks.
                        step_content = step_text
                        if not step_content and event.response is not None:
                            step_content = str(event.response.content or "")
                        # The final AgentOutput also carries the prior tool
                        # results for run bookkeeping, so "non-empty
                        # tool_calls" alone is not a tool request. Only a
                        # ToolSelection means this step asked to call a tool.
                        requested_tools = any(
                            isinstance(tc, ToolSelection) for tc in event.tool_calls
                        )
                        if requested_tools:
                            if step_content.strip():
                                yield OrchestratorEvent(narration=step_content)
                            # A tool-calling step is never the final answer.
                            agent_final_response = ""
                        else:
                            agent_final_response = step_content
                        step_text = ""

                    elif isinstance(event, AgentStream):
                        # Stream the agent's text directly as the final response
                        # tokens. thinking_delta is forwarded as thinking events.
                        if event.thinking_delta:
                            yield OrchestratorEvent(thinking=event.thinking_delta)
                        if event.delta:
                            step_text += event.delta
                            agent_deltas.append(event.delta)
                            yield OrchestratorEvent(token=event.delta)

                # Await the agent handler to completion
                response = await handler
                if not agent_final_response:
                    agent_final_response = str(response)
                break  # Success — exit retry loop

            except WorkflowRuntimeError:
                # Max iterations reached — agent was cut off mid-loop.
                # All tool results from prior iterations are already in
                # tool_results_context, and any streamed text is in
                # agent_final_response. If the agent already streamed
                # text, let it stand; otherwise emit a graceful message.
                if tool_results_context:
                    if agent_final_response:
                        logger.warning(
                            "Max iterations reached after %d tool calls; "
                            "using streamed text as final response",
                            len(tools_called),
                        )
                        break
                    else:
                        logger.warning(
                            "Max iterations reached after %d tool calls "
                            "with no streamed text; emitting graceful message",
                            len(tools_called),
                        )
                        yield OrchestratorEvent(
                            token="I gathered information but was unable to "
                            "complete the response within the iteration limit. "
                            "Please try again or ask a more specific question."
                        )
                        return
                else:
                    logger.error("Max iterations reached with no tool results")
                    yield OrchestratorEvent(
                        token="I was unable to complete the research within the "
                        "iteration limit. Please try a more specific question."
                    )
                    return

            except Exception as e:
                if (
                    _is_transient_llm_error(e)
                    and not tool_results_context
                    and attempt < max_attempts - 1
                ):
                    logger.warning(
                        "Transient LLM error (attempt %d/%d), retrying: %s",
                        attempt + 1,
                        max_attempts,
                        e,
                    )
                    await asyncio.sleep(2)
                    continue  # Retry

                if _is_transient_llm_error(e) and tool_results_context:
                    if agent_final_response:
                        logger.warning(
                            "Transient LLM error after %d tool calls; "
                            "using streamed text as final response: %s",
                            len(tools_called),
                            e,
                        )
                        break
                    else:
                        logger.warning(
                            "Transient LLM error after %d tool calls "
                            "with no streamed text; emitting graceful message: %s",
                            len(tools_called),
                            e,
                        )
                        yield OrchestratorEvent(
                            token="I gathered information but was unable to "
                            "complete the response within the iteration limit. "
                            "Please try again or ask a more specific question."
                        )
                        return

                # Non-transient or final attempt with no results
                logger.error("Orchestrator execution failed: %s", e, exc_info=True)
                yield OrchestratorEvent(
                    token="I encountered an error while processing your "
                    "request. This may be a temporary issue — please try "
                    "again."
                )
                return

        # --- Check if this was an MCP-management-only interaction ---
        # When the agent's only action was managing MCP servers, skip
        # synthesis and emit the tool result directly as the response.
        _mcp_tools = {"manage_mcp_server", "list_mcp_servers", "get_mcp_presets"}
        if tools_called and all(t in _mcp_tools for t in tools_called):
            if "manage_mcp_server" in tools_called and tool_results_context:
                # Extract the last manage_mcp_server result
                last_result = tool_results_context[-1]
                # Strip the "[manage_mcp_server (OK)]\n" prefix
                result_text = (
                    last_result.split("\n", 1)[-1]
                    if "\n" in last_result
                    else last_result
                )
                yield OrchestratorEvent(token=result_text)
                return

        logger.info(
            "Orchestrator execution complete: %d tools called, prompt_len=%d",
            len(tools_called),
            len(prompt),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_chat_messages(
        history: List[Dict[str, Any]],
    ) -> List[ChatMessage]:
        """Convert budgeted history dicts to LlamaIndex ChatMessage objects.

        Args:
            history: Budgeted session messages as dicts with ``role`` and ``content``.

        Returns:
            List of ChatMessage objects suitable for FunctionAgent.run(chat_history=...).
        """
        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
        }

        messages: List[ChatMessage] = []
        for msg in history:
            role_str = msg.get("role", "user")
            role = role_map.get(role_str, MessageRole.USER)
            content = str(msg.get("content", ""))
            if content:
                # Append tool call trace for assistant messages
                if role_str == "assistant" and msg.get("tool_steps"):
                    content += OrchestratorService._format_tool_trace(msg["tool_steps"])
                messages.append(ChatMessage(role=role, content=content))

        return messages

    @staticmethod
    def _format_tool_trace(
        tool_steps: List[Dict[str, Any]],
        max_chars: int = 300,
        max_param_len: int = 60,
    ) -> str:
        """Build a compact tool call trace string for orchestrator context.

        Args:
            tool_steps: List of tool step dicts with ``tool``, ``params``,
                ``is_error`` keys.
            max_chars: Maximum total length of the trace string.
            max_param_len: Maximum length for the displayed parameter value.

        Returns:
            Formatted trace like ``\\n[Tools: name("val") -> ok]``, or empty
            string if *tool_steps* is empty.
        """
        if not tool_steps:
            return ""

        parts: List[str] = []
        for step in tool_steps:
            name = step.get("tool", "unknown")
            params = step.get("params") or {}
            status = "error" if step.get("is_error") else "ok"

            # Extract key param by priority
            param_display: str
            if "query" in params and isinstance(params["query"], str):
                param_display = params["query"]
            elif "url" in params and isinstance(params["url"], str):
                param_display = params["url"]
            elif "urls" in params and isinstance(params["urls"], (list, tuple)):
                param_display = f"{len(params['urls'])} urls"
            else:
                # First string param
                param_display = ""
                for v in params.values():
                    if isinstance(v, str) and v:
                        param_display = v
                        break

            if param_display:
                if len(param_display) > max_param_len:
                    param_display = param_display[: max_param_len - 3] + "..."
                parts.append(f'{name}("{param_display}") -> {status}')
            else:
                parts.append(f"{name}() -> {status}")

        # Assemble with char cap
        prefix = "\n[Tools: "
        suffix = "]"
        available = max_chars - len(prefix) - len(suffix)

        result_parts: List[str] = []
        used = 0
        for i, part in enumerate(parts):
            separator = ", " if result_parts else ""
            needed = len(separator) + len(part)
            remaining_after = len(parts) - i - 1

            if remaining_after > 0:
                # If not the last part, reserve space for a possible overflow
                overflow_reserve = len(f", ... +{remaining_after} more")
            else:
                overflow_reserve = 0

            if used + needed + overflow_reserve <= available:
                result_parts.append(separator + part)
                used += needed
            else:
                remaining = len(parts) - i
                overflow = (
                    f", ... +{remaining} more"
                    if result_parts
                    else f"... +{remaining} more"
                )
                result_parts.append(overflow)
                break

        return prefix + "".join(result_parts) + suffix

    @property
    def tools(self) -> List[FunctionTool]:
        """Get the current tool set (available after first execute())."""
        return self._tools or []

    def get_tool_names(self) -> List[str]:
        """Return names of all currently registered tools.

        Useful for debugging and for including an explicit tool list in
        the system prompt. Returns an empty list if tools have not been
        built yet (i.e. before the first ``execute()`` call).

        Returns:
            Sorted list of tool name strings.
        """
        return sorted(t.metadata.name for t in (self._tools or []) if t.metadata.name)

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def last_rag_result(self) -> Optional[RAGRetrievalResult]:
        """Get the last RAGRetrievalResult from rag_query execution.

        Available after execute() completes if rag_query was called.
        Used by the stream translator for proper source extraction.
        """
        return self._last_rag_result

    @property
    def last_web_sources(self) -> Optional[list]:
        """Get the last web SourceNode list from fetch_pages_batch execution.

        Available after execute() completes if fetch_pages_batch was called
        with the reranking pipeline. Used by the stream translator for
        proper web source extraction with real relevance scores.
        """
        return self._last_web_sources


def build_source_reference(
    tool_results_context: List[str],
    rag_result: Optional["RAGRetrievalResult"] = None,
    web_sources: Optional[list] = None,
) -> str:
    """Build a structured source reference block for the synthesis prompt.

    Parses RAG and web tool outputs to create a numbered reference list
    that the synthesis LLM can cite. Sources are numbered sequentially
    across both RAG and web results.

    Args:
        tool_results_context: Formatted tool result strings (used as fallback).
        rag_result: Optional RAGRetrievalResult for KB sources.
        web_sources: Optional list of SourceNode objects from web pipeline.

    Returns:
        A formatted source reference string, or empty string if no sources.
    """
    lines: List[str] = []
    idx = 1

    # RAG sources (from RAGRetrievalResult)
    if rag_result and rag_result.source_nodes:
        for node in rag_result.source_nodes:
            inner = getattr(node, "node", node)
            metadata = {}
            if hasattr(inner, "metadata") and inner.metadata:
                metadata = inner.metadata
            elif hasattr(node, "metadata") and node.metadata:
                metadata = node.metadata

            title = (
                metadata.get("display_name")
                or metadata.get("title")
                or metadata.get("file_name")
                or "Untitled"
            )
            score = node.score if hasattr(node, "score") and node.score else None
            score_str = f", score: {score:.2f}" if score is not None else ""

            lines.append(f'[{idx}] "{title}" (knowledge base{score_str})')
            idx += 1

    # Web sources (from SourceFetchPipeline SourceNode objects)
    if web_sources:
        from tensortruth.core.source import SourceStatus

        for node in web_sources:
            if node.status not in (SourceStatus.SUCCESS, SourceStatus.FILTERED):
                continue
            score_str = f", score: {node.score:.2f}" if node.score is not None else ""
            lines.append(f'[{idx}] "{node.title}" (web{score_str}) - {node.url}')
            idx += 1

    if not lines:
        return ""

    header = "--- Source Reference ---\n"
    return header + "\n".join(lines) + "\n--- End Source Reference ---"


def load_module_descriptions(
    modules: List[str],
    config: Any,
) -> List[ModuleDescription]:
    """Load module descriptions from the index catalog.

    Reads ChromaDB metadata for each active module to extract display_name
    and doc_type. Used to populate the orchestrator's system prompt with
    information about what knowledge is available.

    Args:
        modules: List of active module names (from session config).
        config: TensorTruthConfig instance.

    Returns:
        List of ModuleDescription for each module that could be resolved.
    """
    if not modules:
        return []

    descriptions: List[ModuleDescription] = []

    try:
        from tensortruth.app_utils.helpers import get_module_display_name
        from tensortruth.app_utils.paths import get_indexes_dir
        from tensortruth.indexing.metadata import sanitize_model_id

        indexes_dir = get_indexes_dir()
        model_id = sanitize_model_id(config.rag.default_embedding_model)
        model_indexes_dir = indexes_dir / model_id

        for module_name in modules:
            try:
                display_name, doc_type, _, _ = get_module_display_name(
                    model_indexes_dir, module_name
                )
                descriptions.append(
                    ModuleDescription(
                        name=module_name,
                        display_name=display_name,
                        doc_type=doc_type,
                    )
                )
            except Exception:
                # Module metadata unavailable -- include with bare name
                descriptions.append(
                    ModuleDescription(
                        name=module_name,
                        display_name=module_name,
                        doc_type="unknown",
                    )
                )
                logger.debug(
                    "Could not load metadata for module '%s', using bare name",
                    module_name,
                )
    except Exception:
        logger.warning(
            "Failed to load module descriptions, proceeding without them",
            exc_info=True,
        )
        # Fall back to bare names
        for module_name in modules:
            descriptions.append(
                ModuleDescription(
                    name=module_name,
                    display_name=module_name,
                    doc_type="unknown",
                )
            )

    return descriptions
