"""Orchestrator service built on LlamaIndex FunctionAgent.

Routes every user prompt through a tool-calling agent that decides whether to
query the knowledge base, search the web, fetch pages, call MCP tools, or
respond directly. The FunctionAgent handles the agentic loop internally
(call tool -> inspect result -> decide -> repeat or produce final answer).

This is the core of the agentic chat loop.  It does NOT maintain conversation
memory -- history is passed per-call from ChatHistoryService.
"""

from __future__ import annotations

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

from tensortruth.agents.tool_output import extract_tool_text
from tensortruth.core.ollama import get_orchestrator_llm
from tensortruth.services.models import RAGRetrievalResult, ToolProgress
from tensortruth.services.orchestrator_tool_wrappers import (
    ProgressEmitter,
    create_all_tool_wrappers,
)

if TYPE_CHECKING:
    from llama_index.llms.ollama import Ollama

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
    return _TOOL_ANALYZING_MESSAGES.get(tool_name, _DEFAULT_ANALYZING_MESSAGE)


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
        """
        self._tool_service = tool_service
        self._rag_service = rag_service
        self._model = model
        self._base_url = base_url
        self._context_window = context_window
        self._session_params = session_params or {}
        self._session_messages = session_messages
        self._module_descriptions = module_descriptions or []
        self._custom_instructions = custom_instructions
        self._project_metadata = project_metadata
        self._max_iterations = max_iterations

        # Built lazily on first execute()
        self._llm: Optional["Ollama"] = None
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
            self._last_web_sources = nodes

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

        # --- Role ---
        sections.append(
            "You are an intelligent assistant that helps users by answering "
            "questions, finding information, and completing tasks. You have "
            "access to tools that let you search a knowledge base, search the "
            "web, fetch web pages, and more. Decide which tools to use based "
            "on the user's question."
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
            "indexed modules, search the web. IMPORTANT: web search returns "
            "only titles, URLs, and short snippets — NOT full page content. "
            "After searching, always fetch the full content of relevant pages "
            "before answering. Search snippets alone are not enough for a "
            "comprehensive answer.\n"
            "- For simple conversational messages (greetings, clarifications, "
            "opinions), respond directly without using any tools.\n"
            "- If a tool returns an error, analyze the cause: if the input was "
            "wrong, correct it and retry; if it is an internal error, report "
            "the issue and continue with other tools if possible."
        )

        # --- Iteration budget ---
        sections.append(
            f"IMPORTANT: You have a budget of {self._max_iterations} iterations "
            "for this request. Each tool call or response counts as one iteration. "
            "Plan your research efficiently: limit web searches to 1-2 focused "
            "queries, then use remaining iterations to fetch and read page content."
        )

        # --- Synthesis handoff ---
        # When tools are called, a separate synthesis service generates the
        # final answer.  Instruct the orchestrator to keep its post-tool
        # response minimal so it doesn't waste tokens on text that will be
        # discarded.
        sections.append(
            "IMPORTANT: After you have called tools and gathered information, "
            "respond with ONLY a very brief one-line summary of what you found "
            "(e.g. 'Found 3 relevant sources about X.'). Do NOT write a "
            "detailed answer — a separate synthesis step will handle that."
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
    # Main execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        progress_emitter: Optional[ProgressEmitter] = None,
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

        Yields:
            OrchestratorEvent instances (tokens, tool calls, tool phases).
        """
        # Reset per-execution state
        self._pending_phases = []
        self._last_rag_result = None
        self._last_web_sources = None
        self._last_web_query = None
        self._last_web_search_results = None

        # Build a progress emitter that both stores phases locally and
        # forwards to the external emitter (if provided).
        def _combined_emitter(tp: ToolProgress) -> None:
            self._pending_phases.append(tp)
            if progress_emitter:
                progress_emitter(tp)

        # Emit orchestrator start phase directly (not deferred via
        # _pending_phases) so it arrives at the frontend before any
        # reasoning events from the FunctionAgent's initial thinking.
        initial_phase = ToolProgress(
            tool_id="orchestrator",
            phase="thinking",
            message="Analyzing your request...",
        )
        yield OrchestratorEvent(tool_phase=initial_phase)
        if progress_emitter:
            progress_emitter(initial_phase)

        # --- Build tools and agent ---
        tools = self._build_tools(_combined_emitter)
        self._tools = tools

        # Get (or reuse) orchestrator LLM singleton
        self._llm = get_orchestrator_llm(
            model=self._model,
            base_url=self._base_url,
            context_window=self._context_window,
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

        # --- Phase 1: Tool routing (non-thinking orchestrator LLM) ---
        tools_called: List[str] = []
        tool_steps: List[Dict[str, Any]] = []
        tool_results_context: List[str] = []
        agent_final_response = ""

        try:
            handler = agent.run(
                user_msg=prompt,
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
                        }
                    )

                    # Drain any pending tool phase events that were emitted
                    # by the tool wrapper's progress emitter
                    for phase in self._pending_phases:
                        yield OrchestratorEvent(tool_phase=phase)
                    self._pending_phases = []

                elif isinstance(event, ToolCallResult):
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

                elif isinstance(event, AgentStream):
                    # Capture but do NOT stream the orchestrator's response.
                    # Synthesis will be done separately with the thinking LLM.
                    if event.delta:
                        agent_final_response += event.delta
                        yield OrchestratorEvent(reasoning=event.delta)

            # Await the agent handler to completion
            response = await handler
            if not agent_final_response:
                agent_final_response = str(response)

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

        except Exception as e:
            logger.error("Orchestrator execution failed: %s", e, exc_info=True)
            yield OrchestratorEvent(
                token=f"I encountered an error while processing your request: {e}"
            )
            return

        # --- Phase 2: Response generation ---
        # If no tools were called, the orchestrator's direct response is the
        # final answer (no tool context to synthesize). Stream it directly.
        if not tools_called:
            yield OrchestratorEvent(token=agent_final_response)
            logger.info(
                "Orchestrator execution complete (direct): 0 tools, %d chars",
                len(agent_final_response),
            )
            return

        # Tools were called — delegate to the synthesis service for a
        # comprehensive final answer (always two-phase when tools are called).
        from tensortruth.services.synthesis_service import get_synthesis_service

        # Build structured source reference for citation guidance
        source_reference = build_source_reference(
            tool_results_context,
            rag_result=self._last_rag_result,
            web_sources=self._last_web_sources,
            web_search_results=self._last_web_search_results,
        )

        synthesis = get_synthesis_service(
            self._model, self._base_url, self._context_window
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
            "Orchestrator execution complete: %d tools called",
            len(tools_called),
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
    web_search_results: Optional[List[Dict[str, Any]]] = None,
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

    # Fallback: use raw search results when no pages were fetched
    if not web_sources and web_search_results:
        for result in web_search_results:
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            lines.append(f'[{idx}] "{title}" (web, snippet only) - {url}')
            idx += 1

    if not lines:
        return ""

    has_snippet_only = any("snippet only" in line for line in lines)

    header = "--- Source Reference ---\n"
    if has_snippet_only:
        header += (
            "Note: Sources marked '(snippet only)' were not fetched — "
            "only search result snippets are available for these.\n"
        )

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
