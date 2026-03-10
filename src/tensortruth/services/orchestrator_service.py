"""Orchestrator service built on LlamaIndex FunctionAgent.

Routes every user prompt through a tool-calling agent that decides whether to
query the knowledge base, search the web, fetch pages, call MCP tools, or
respond directly. The FunctionAgent handles the agentic loop internally
(call tool -> inspect result -> decide -> repeat or produce final answer).

This is the core of the agentic chat loop.  It does NOT maintain conversation
memory -- history is passed per-call from ChatHistoryService.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
)

from llama_index.core.agent.workflow import AgentStream, ToolCall, ToolCallResult
from llama_index.core.agent.workflow.function_agent import (
    FunctionAgent as LIFunctionAgent,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
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
from tensortruth.core.providers import resolve_thinking as _providers_resolve_thinking
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
}
_DEFAULT_ANALYZING_MESSAGE = "Analyzing results..."


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
        mcp_proposal_service: Optional[Any] = None,
        mcp_server_service: Optional[Any] = None,
        session_id: Optional[str] = None,
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
            mcp_proposal_service: Optional MCPProposalService for MCP management tools.
            mcp_server_service: Optional MCPServerService for MCP management tools.
            session_id: Session ID for MCP proposal tracking.
        """
        self._tool_service = tool_service
        self._rag_service = rag_service
        self._model = model
        self._base_url = base_url
        self._context_window = context_window
        self._session_params = session_params or {}

        # Resolve model reference once for provider-aware LLM creation
        self._model_ref = resolve_model_from_params(self._session_params, model)
        self._session_messages = session_messages
        self._module_descriptions = module_descriptions or []
        self._custom_instructions = custom_instructions
        self._project_metadata = project_metadata
        self._max_iterations = max_iterations
        self._mcp_proposal_service = mcp_proposal_service
        self._mcp_server_service = mcp_server_service
        self._session_id = session_id

        # Built lazily on first execute()
        self._llm: Optional["LLM"] = None
        self._agent: Optional[LIFunctionAgent] = None
        self._tools: Optional[List[FunctionTool]] = None

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
    # Tool assembly
    # ------------------------------------------------------------------

    def _build_tools(self, progress_emitter: ProgressEmitter) -> List[FunctionTool]:
        """Build the complete tool set for the orchestrator.

        Combines:
        1. Wrapped built-in tools (rag_query, web_search, fetch_page, etc.)
        2. MCP tools from ToolService (pass-through, already FunctionTool instances)

        Built-in ToolService tools that are already wrapped are filtered out
        to avoid duplicates.

        Args:
            progress_emitter: Callback for tool phase progress reporting.

        Returns:
            Complete list of FunctionTool instances.
        """

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

        # Load web search config for thresholds
        try:
            from tensortruth.api.deps import get_config_service

            config = get_config_service().load()
            ws_config = config.web_search
            title_threshold = ws_config.rerank_title_threshold
            content_threshold = ws_config.rerank_content_threshold
        except Exception:
            title_threshold = 0.1
            content_threshold = 0.1

        # 1. Create wrapped built-in tools (RAG + web tools with reranking)
        wrapped_tools = create_all_tool_wrappers(
            tool_service=self._tool_service,
            progress_emitter=progress_emitter,
            rag_service=self._rag_service,
            session_params=self._session_params,
            session_messages=self._session_messages,
            rag_result_callback=_rag_result_cb,
            reranker_model=reranker_model,
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
            mcp_proposal_service=self._mcp_proposal_service,
            mcp_server_service=self._mcp_server_service,
            session_id=self._session_id,
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

        return all_tools

    # ------------------------------------------------------------------
    # System prompt composition
    # ------------------------------------------------------------------

    def _build_system_prompt(self, tools: List[FunctionTool]) -> str:
        """Compose the orchestrator system prompt.

        Includes:
        - Role description
        - Active indexed modules with descriptions
        - Tool usage guidance
        - Custom instructions / project metadata
        - Explicit tool list to prevent hallucination

        Args:
            tools: The full tool set (used to list available tool names).

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
        if self._module_descriptions:
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
            "search the knowledge base.\n"
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
        mcp_mgmt_tools = {"list_mcp_servers", "get_mcp_presets", "propose_mcp_server"}
        has_mcp_mgmt = any(t.metadata.name in mcp_mgmt_tools for t in tools)
        if has_mcp_mgmt:
            sections.append(
                "MCP server management:\n"
                "- Use list_mcp_servers to see current MCP server configurations.\n"
                "- Use get_mcp_presets to check for known preset configurations "
                "(presets auto-fill command/args, so you only need name and summary).\n"
                "- For servers NOT in presets, you MUST first research the correct "
                "installation command before proposing. Use web_search to find the "
                "server's npm package or GitHub repo, then fetch_page to read the "
                "README and find the exact command and args (usually `npx -y <package>` "
                "for npm-based servers). Only call propose_mcp_server once you have "
                "the concrete command and args.\n"
                "- Use propose_mcp_server to propose adding, updating, or removing "
                "a server. For 'add' with type 'stdio', you MUST provide `command` "
                "and `args` (e.g. command='npx', args=['-y', '<package>']). "
                "The user must approve the proposal before it takes effect.\n"
                "- After a proposal is approved, the tools will be reloaded automatically.\n"
                "- NEVER retry propose_mcp_server with the same arguments if it fails. "
                "Fix the issue first (e.g. research the correct command)."
            )

        # --- MCP tool routing ---
        mcp_tools = [
            t
            for t in tools
            if t.metadata.name not in _WRAPPED_BUILTIN_TOOL_NAMES
            and t.metadata.name != "rag_query"
            and t.metadata.name not in mcp_mgmt_tools
        ]
        if mcp_tools:
            lines = []
            for t in mcp_tools:
                desc = (t.metadata.description or "")[:200]
                if desc:
                    lines.append(f"- {t.metadata.name}: {desc}")
            if lines:
                sections.append("Additional tools:\n" + "\n".join(lines))

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

        # --- Synthesis handoff ---
        # When tools are called, a separate synthesis service generates the
        # final answer.  Instruct the orchestrator to keep its post-tool
        # response minimal so it doesn't waste tokens on text that will be
        # discarded.
        sections.append(
            "CRITICAL: After gathering information from tools, respond with "
            "AT MOST one or two short sentences summarizing what you found "
            "(e.g. 'Found 3 relevant sources about X with high confidence.'). "
            "Do NOT analyze, explain, or answer the question — a separate "
            "synthesis step does that. Any detailed response you write will "
            "be DISCARDED. Keep your inter-tool summary under 200 characters."
        )

        # --- Project metadata ---
        if self._project_metadata:
            sections.append(f"Project context:\n{self._project_metadata}")

        # --- Custom instructions ---
        if self._custom_instructions:
            sections.append(
                f"Additional instructions from the user:\n{self._custom_instructions}"
            )

        # --- Explicit tool list (prevent hallucination) ---
        tool_names = [t.metadata.name for t in tools if t.metadata.name]
        tool_list_str = ", ".join(tool_names)
        sections.append(
            f"You have access to ONLY these tools: {tool_list_str}. "
            "Do NOT call any tool not in this list."
        )

        return "\n\n".join(sections)

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
    # Direct response generation (thinking-enabled, no tools)
    # ------------------------------------------------------------------

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

        # Load configurable reasoning visibility flag
        try:
            from tensortruth.api.deps import get_config_service

            _agent_config = get_config_service().load().agent
            show_reasoning = _agent_config.show_orchestrator_reasoning
        except Exception:
            show_reasoning = False

        # Build a progress emitter that both stores phases locally and
        # forwards to the external emitter (if provided).
        def _combined_emitter(tp: ToolProgress) -> None:
            self._pending_phases.append(tp)
            if progress_emitter:
                progress_emitter(tp)

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

        # --- Build tools and agent ---
        tools = self._build_tools(_combined_emitter)
        self._tools = tools

        # Get (or reuse) orchestrator LLM singleton (provider-agnostic)
        self._llm = _providers_get_orchestrator_llm(
            self._model_ref,
            self._context_window,
        )

        # Compose system prompt
        system_prompt = self._build_system_prompt(tools)

        # Budget chat history
        budgeted_history = self._budget_history(
            chat_history or [],
            system_prompt,
            prompt,
        )

        # Convert budgeted history to LlamaIndex ChatMessage objects
        llama_history = self._to_chat_messages(budgeted_history)

        # Create FunctionAgent for this execution
        agent = LIFunctionAgent(
            tools=list(tools),  # type: ignore[arg-type]
            llm=self._llm,
            system_prompt=system_prompt,
        )

        # Emit thinking phase once agent is ready and about to run.
        thinking_phase = ToolProgress(
            tool_id="orchestrator",
            phase="thinking",
            message="Analyzing your request...",
        )
        yield OrchestratorEvent(tool_phase=thinking_phase)
        if progress_emitter:
            progress_emitter(thinking_phase)

        # --- Phase 1: Tool routing (non-thinking orchestrator LLM) ---
        reasoning_chars = 0
        MAX_REASONING_CHARS = 1000  # ~250 tokens, enough for inter-tool reasoning
        tools_called: List[str] = []
        tool_steps: List[Dict[str, Any]] = []
        tool_results_context: List[str] = []
        agent_final_response = ""
        agent_deltas: List[str] = []

        max_attempts = 2

        for attempt in range(max_attempts):
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

                handler = agent.run(
                    user_msg=user_msg,
                    chat_history=llama_history if llama_history else None,
                    max_iterations=self._max_iterations,
                )

                async for event in handler.stream_events():
                    if isinstance(event, ToolCall):
                        tools_called.append(event.tool_name)

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

                        # Reset reasoning char counter so the next inter-tool
                        # reasoning gap gets its own visibility budget.
                        reasoning_chars = 0

                    elif isinstance(event, AgentStream):
                        # Capture the orchestrator's text output.
                        # If tools have been called, this is ephemeral reasoning
                        # between tool calls — show it in the reasoning box.
                        # If no tools have been called yet, this might be the
                        # final answer (agent responding directly) — DON'T show
                        # it as reasoning unless show_reasoning is enabled.
                        if event.delta:
                            agent_final_response += event.delta
                            agent_deltas.append(event.delta)
                            if tools_called or show_reasoning:
                                reasoning_chars += len(event.delta)
                                if reasoning_chars <= MAX_REASONING_CHARS:
                                    yield OrchestratorEvent(reasoning=event.delta)

                # Await the agent handler to completion
                response = await handler
                if not agent_final_response:
                    agent_final_response = str(response)
                break  # Success — exit retry loop

            except WorkflowRuntimeError:
                # Max iterations reached — agent was cut off mid-loop.
                # All tool results from prior iterations are already in
                # tool_results_context. If we have any, fall through to synthesis.
                if tool_results_context:
                    logger.warning(
                        "Max iterations reached after %d tool calls; "
                        "proceeding to synthesis with available results",
                        len(tools_called),
                    )
                    # Fall through to Phase 2 (synthesis) below
                else:
                    logger.error("Max iterations reached with no tool results")
                    yield OrchestratorEvent(
                        token="I was unable to complete the research within the "
                        "iteration limit. Please try a more specific question."
                    )
                    return
                break

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
                    logger.warning(
                        "Transient LLM error after %d tool calls; "
                        "proceeding to synthesis with available results: %s",
                        len(tools_called),
                        e,
                    )
                    break  # Fall through to Phase 2

                # Non-transient or final attempt with no results
                logger.error("Orchestrator execution failed: %s", e, exc_info=True)
                yield OrchestratorEvent(
                    token="I encountered an error while processing your "
                    "request. This may be a temporary issue — please try "
                    "again."
                )
                return

        # --- Check if this was a proposal-only interaction ---
        # When the agent's only meaningful action was proposing an MCP server
        # change, skip synthesis — the approval card IS the response.
        _proposal_tools = {"propose_mcp_server", "list_mcp_servers", "get_mcp_presets"}
        if tools_called and all(t in _proposal_tools for t in tools_called):
            # Check if a proposal was actually created (not just listed)
            has_proposal = "propose_mcp_server" in tools_called
            if has_proposal:
                yield OrchestratorEvent(
                    token="I've proposed the MCP server configuration change above. "
                    "Please review and approve or reject it."
                )
                return

        # --- Phase 2: Response generation ---
        # Route through the synthesis service for both tools-called and
        # no-tools paths.  The synthesizer handles empty tool_results by
        # switching to a general assistant prompt without citation rules.
        from tensortruth.services.synthesis_service import get_synthesis_service

        # Build structured source reference for citation guidance (only
        # meaningful when tools were called).
        source_reference = None
        if tools_called:
            source_reference = build_source_reference(
                tool_results_context,
                rag_result=self._last_rag_result,
                web_sources=self._last_web_sources,
            )

        thinking_pref = self._session_params.get("thinking")
        resolved_thinking = _providers_resolve_thinking(self._model_ref, thinking_pref)
        synthesis = get_synthesis_service(
            self._model,
            self._base_url,
            self._context_window,
            thinking=resolved_thinking,
            provider_id=self._session_params.get("provider_id"),
        )
        async for synth_event in synthesis.synthesize(
            prompt=prompt,
            chat_history=llama_history,
            tool_results=tool_results_context,
            module_descriptions=self._module_descriptions,
            custom_instructions=self._custom_instructions,
            project_metadata=self._project_metadata,
            progress_emitter=_combined_emitter,
            source_reference=source_reference,
        ):
            yield synth_event

        logger.info(
            "Orchestrator execution complete%s: %d tools called, prompt_len=%d",
            " (direct)" if not tools_called else "",
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
                messages.append(ChatMessage(role=role, content=content))

        return messages

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
