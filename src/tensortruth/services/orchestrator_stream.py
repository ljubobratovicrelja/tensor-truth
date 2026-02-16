"""Event translation layer for orchestrator -> WebSocket streaming.

Converts OrchestratorEvent objects from OrchestratorService.execute() into
WebSocket-compatible message dicts matching the schemas in api/schemas/chat.py.
Also accumulates sources from RAG and web tool results and produces a batched
sources message after all events have been processed.

This module is consumed by Story 7 (WebSocket handler refactor) which wires
it into the chat WebSocket endpoint.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tensortruth.core.source import SourceNode as CoreSourceNode
from tensortruth.core.source_converter import SourceConverter
from tensortruth.services.models import RAGRetrievalResult, ToolProgress

logger = logging.getLogger(__name__)

# Tool names whose results may contain extractable sources.
_RAG_TOOL_NAME = "rag_query"
_WEB_SEARCH_TOOL_NAME = "web_search"
_FETCH_PAGE_TOOL_NAME = "fetch_page"


@dataclass
class OrchestratorStreamResult:
    """Accumulated result from an orchestrator execution stream.

    Populated by ``translate_events()`` as it processes OrchestratorEvent
    objects. After iteration completes, contains all the data needed by the
    WebSocket handler to send the final ``sources`` and ``done`` messages.

    Attributes:
        full_response: Accumulated text from all token events.
        sources: Batched source dicts in API schema format.
        metrics: RAG retrieval metrics (only when RAG was called).
        confidence_level: Derived from RAG result or default "high".
        tool_steps: List of tool step dicts for session storage.
        rag_called: Whether the rag_query tool was invoked.
        web_called: Whether web_search was invoked.
        source_types: List of source type strings present (e.g. ["rag", "web"]).
        rag_count: Number of RAG sources.
        web_count: Number of web sources.
    """

    full_response: str = ""
    full_thinking: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    confidence_level: str = "high"
    tool_steps: List[Dict[str, Any]] = field(default_factory=list)
    rag_called: bool = False
    web_called: bool = False
    source_types: List[str] = field(default_factory=list)
    rag_count: int = 0
    web_count: int = 0


def translate_event(event: Any) -> Optional[Dict[str, Any]]:
    """Translate a single OrchestratorEvent into a WebSocket message dict.

    Handles four event kinds:
    - ``token``: text delta from the LLM's response
    - ``tool_call``: a tool invocation starting
    - ``tool_call_result``: a tool invocation completing
    - ``tool_phase``: phase-level progress within a tool

    Args:
        event: An OrchestratorEvent instance.

    Returns:
        A dict matching one of the WebSocket message schemas
        (StreamToken, StreamToolProgress, StreamToolPhase), or ``None`` if
        the event type is unrecognised.
    """
    # Thinking/reasoning delta
    if getattr(event, "thinking", None) is not None:
        return {"type": "thinking", "content": event.thinking}

    # Token delta
    if event.token is not None:
        return {"type": "token", "content": event.token}

    # Tool call started
    if event.tool_call is not None:
        tc = event.tool_call
        return {
            "type": "tool_progress",
            "tool": tc.get("tool", ""),
            "action": "calling",
            "params": tc.get("params", {}),
        }

    # Tool call completed
    if event.tool_call_result is not None:
        tcr = event.tool_call_result
        action = "failed" if tcr.get("is_error") else "completed"
        return {
            "type": "tool_progress",
            "tool": tcr.get("tool", ""),
            "action": action,
            "params": tcr.get("params", {}),
            "output": tcr.get("output"),
            "is_error": tcr.get("is_error", False),
        }

    # Tool phase progress
    if event.tool_phase is not None:
        tp: ToolProgress = event.tool_phase
        return {
            "type": "tool_phase",
            "tool_id": tp.tool_id,
            "phase": tp.phase,
            "message": tp.message,
            "metadata": tp.metadata,
        }

    return None


class OrchestratorStreamTranslator:
    """Stateful translator that converts an orchestrator event stream into
    WebSocket messages and accumulates sources, metrics, and confidence.

    Usage::

        translator = OrchestratorStreamTranslator(chat_service)
        async for event in orchestrator.execute(prompt, history, emitter):
            msg = translator.process_event(event)
            if msg is not None:
                await websocket.send_json(msg)

        result = translator.finalize()
        # result.sources, result.metrics, result.confidence_level, etc.
    """

    def __init__(
        self,
        chat_service: Any = None,
    ):
        """Initialize the translator.

        Args:
            chat_service: Optional ChatService instance used for converting
                RAG source nodes to API-compatible dicts. When not provided,
                RAG sources are extracted from the tool output text instead.
        """
        self._chat_service = chat_service

        # Internal accumulation state
        self._full_response = ""
        self._full_thinking = ""
        self._rag_sources: List[Dict[str, Any]] = []
        self._web_sources: List[Dict[str, Any]] = []
        self._rag_metrics: Optional[Dict[str, Any]] = None
        self._rag_confidence: Optional[str] = None
        self._tool_steps: List[Dict[str, Any]] = []
        self._rag_called = False
        self._web_called = False

        # Store raw RAGRetrievalResult when available (set externally)
        self._rag_retrieval_result: Optional[RAGRetrievalResult] = None

    def set_rag_retrieval_result(self, result: RAGRetrievalResult) -> None:
        """Inject a RAGRetrievalResult for proper source extraction.

        When the rag_query tool wrapper stores its RAGRetrievalResult,
        the WebSocket handler can forward it here for proper source
        conversion using ChatService.extract_sources().

        Args:
            result: The RAGRetrievalResult from the rag_query tool execution.
        """
        self._rag_retrieval_result = result

    def set_web_source_nodes(self, nodes: List[CoreSourceNode]) -> None:
        """Inject web SourceNodes from the reranking pipeline.

        Converts core SourceNode objects (from SourceFetchPipeline) to
        API-compatible source dicts with real relevance scores. Replaces
        any web sources that were extracted from raw JSON parsing.

        Args:
            nodes: List of SourceNode objects from fetch_pages_batch.
        """
        self._web_sources = []
        for node in nodes:
            if not node.url:
                continue
            source_dict: Dict[str, Any] = {
                "text": node.snippet or node.content or "",
                "score": node.effective_score,
                "metadata": {
                    "source_url": node.url,
                    "display_name": node.title,
                    "doc_type": "web",
                    "fetch_status": node.status.value,
                    "content_chars": node.content_chars,
                    "url": node.url,
                    "title": node.title,
                    "snippet": node.snippet or "",
                    "relevance_score": node.score,
                },
            }
            self._web_sources.append(source_dict)
        self._web_called = True

    def process_event(self, event: Any) -> Optional[Dict[str, Any]]:
        """Process a single OrchestratorEvent, accumulating state and
        returning the WebSocket message dict to send (if any).

        Args:
            event: An OrchestratorEvent instance.

        Returns:
            A WebSocket message dict, or None if no message should be sent.
        """
        # Accumulate thinking text
        if getattr(event, "thinking", None) is not None:
            self._full_thinking += event.thinking

        # Accumulate token text
        if event.token is not None:
            self._full_response += event.token

        # Track tool call results for source extraction and step storage
        if event.tool_call_result is not None:
            tcr = event.tool_call_result
            tool_name = tcr.get("tool", "")

            # Store tool step (strip full_output to keep storage lean)
            step = {k: v for k, v in tcr.items() if k != "full_output"}
            self._tool_steps.append(step)

            # Track which tool types were called
            if tool_name == _RAG_TOOL_NAME:
                self._rag_called = True
                self._extract_rag_data(tcr)
            elif tool_name == _WEB_SEARCH_TOOL_NAME:
                self._web_called = True
                self._extract_web_sources(tcr)
            elif tool_name == _FETCH_PAGE_TOOL_NAME:
                self._promote_web_source(tcr)

        # Track tool calls (not results) for rag/web flags
        if event.tool_call is not None:
            tool_name = event.tool_call.get("tool", "")
            if tool_name == _RAG_TOOL_NAME:
                self._rag_called = True
            elif tool_name == _WEB_SEARCH_TOOL_NAME:
                self._web_called = True

        # Translate to WebSocket message
        return translate_event(event)

    def finalize(self) -> OrchestratorStreamResult:
        """Finalize the stream and produce the accumulated result.

        Computes the batched sources list, confidence level, and source
        type breakdown. Call this after the orchestrator event stream
        has been fully consumed AND after set_rag_retrieval_result() has
        been called (if applicable).

        Returns:
            OrchestratorStreamResult with all accumulated data.
        """
        # Re-extract RAG sources if the RAGRetrievalResult was injected
        # after event processing (the normal flow: set_rag_retrieval_result
        # is called after the event loop completes).
        if self._rag_retrieval_result is not None and not self._rag_sources:
            self._extract_sources_from_rag_result()

        # Build combined sources list
        all_sources = self._rag_sources + self._web_sources

        # Determine source types
        source_types: List[str] = []
        if self._rag_sources:
            source_types.append("rag")
        if self._web_sources:
            source_types.append("web")

        # Determine confidence level
        confidence = self._derive_confidence()

        return OrchestratorStreamResult(
            full_response=self._full_response,
            full_thinking=self._full_thinking,
            sources=all_sources,
            metrics=self._rag_metrics,
            confidence_level=confidence,
            tool_steps=self._tool_steps,
            rag_called=self._rag_called,
            web_called=self._web_called,
            source_types=source_types,
            rag_count=len(self._rag_sources),
            web_count=len(self._web_sources),
        )

    def build_sources_message(self) -> Optional[Dict[str, Any]]:
        """Build the batched sources WebSocket message.

        Returns the ``{"type": "sources", ...}`` dict if there are any
        sources to send, or ``None`` if no sources were collected.

        This is called after ``finalize()`` to produce the sources
        message for the WebSocket.

        Returns:
            Sources message dict, or None.
        """
        result = self.finalize()
        if not result.sources:
            return None

        msg: Dict[str, Any] = {
            "type": "sources",
            "data": result.sources,
            "metrics": result.metrics,
        }

        # Include source type breakdown for mixed-source rendering (Story 9)
        if len(result.source_types) > 1:
            msg["source_types"] = result.source_types
            msg["rag_count"] = result.rag_count
            msg["web_count"] = result.web_count

        return msg

    def build_done_message(
        self,
        title_pending: bool = False,
    ) -> Dict[str, Any]:
        """Build the ``done`` WebSocket message.

        Args:
            title_pending: Whether a title generation is pending.

        Returns:
            Done message dict.
        """
        result = self.finalize()
        msg: Dict[str, Any] = {
            "type": "done",
            "content": result.full_response,
            "confidence_level": result.confidence_level,
        }
        if title_pending:
            msg["title_pending"] = True
        return msg

    # ------------------------------------------------------------------
    # Private extraction helpers
    # ------------------------------------------------------------------

    def _extract_rag_data(self, tool_call_result: Dict[str, Any]) -> None:
        """Extract RAG confidence and metrics from a rag_query result text.

        Source node extraction is deferred to finalize() because the
        RAGRetrievalResult is injected after event processing completes.
        This method only parses confidence and metrics from the text output.

        Args:
            tool_call_result: The tool_call_result dict from the orchestrator event.
        """
        output = tool_call_result.get("output", "")
        self._extract_rag_from_text(output)

    def _extract_sources_from_rag_result(self) -> None:
        """Extract proper source dicts from the injected RAGRetrievalResult.

        Called by finalize() after set_rag_retrieval_result() has been invoked.
        Converts LlamaIndex NodeWithScore objects to API-compatible source dicts.
        """
        rr = self._rag_retrieval_result
        if rr is None:
            return

        # Override text-parsed confidence/metrics with the real values
        self._rag_confidence = rr.confidence_level
        if rr.metrics:
            self._rag_metrics = rr.metrics

        if not rr.source_nodes:
            return

        # Try ChatService.extract_sources() first (proper conversion)
        if self._chat_service is not None:
            try:
                self._rag_sources = self._chat_service.extract_sources(rr.source_nodes)
                return
            except Exception:
                logger.warning(
                    "Failed to extract sources via ChatService, "
                    "falling back to SourceConverter",
                    exc_info=True,
                )

        # Fallback: convert directly via SourceConverter
        try:
            unified = SourceConverter.batch_from_rag_nodes(rr.source_nodes)
            self._rag_sources = SourceConverter.batch_to_api_schema(unified)
        except Exception:
            logger.warning(
                "Failed to convert RAG nodes via SourceConverter",
                exc_info=True,
            )

    def _extract_rag_from_text(self, output: str) -> None:
        """Parse RAG sources from the text output of _format_retrieval_result().

        Extracts confidence level and basic source info from the structured
        text format produced by orchestrator_tool_wrappers._format_retrieval_result.

        Args:
            output: The text output from the rag_query tool.
        """
        if not output:
            return

        # Extract confidence from "Found N sources (confidence: X)" line
        for line in output.split("\n"):
            if "confidence:" in line.lower():
                # Parse "confidence: normal" or "confidence: low" etc.
                parts = line.split("confidence:")
                if len(parts) > 1:
                    conf = parts[1].strip().rstrip(")")
                    if conf in ("normal", "low", "none", "high"):
                        self._rag_confidence = conf
                break

        # Extract metrics from "Metrics: avg_score=0.1234, ..." line
        for line in output.split("\n"):
            if line.startswith("Metrics:"):
                metrics_str = line[len("Metrics:") :].strip()
                metrics_dict: Dict[str, Any] = {}
                for pair in metrics_str.split(","):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        try:
                            metrics_dict[k.strip()] = float(v.strip())
                        except ValueError:
                            metrics_dict[k.strip()] = v.strip()
                if metrics_dict:
                    self._rag_metrics = {"parsed_summary": metrics_dict}
                break

        # Note: When using text-based extraction, we cannot reconstruct full
        # source dicts matching the SourceNode API schema. Sources will only
        # be available via the injected RAGRetrievalResult path.
        # This text path captures confidence and metrics only.

    def _extract_web_sources(self, tool_call_result: Dict[str, Any]) -> None:
        """Extract web sources from a web_search tool_call_result.

        The web_search tool returns JSON with search results:
        ``[{"url": "...", "title": "...", "snippet": "...", "query": "..."}]``

        We convert these into API-compatible source dicts matching the
        SourceNode schema (same format as RAG sources but with web doc_type).

        Prefers ``full_output`` (untruncated) over ``output`` (truncated to
        2000 chars) to avoid broken JSON parsing.

        Args:
            tool_call_result: The tool_call_result dict from the orchestrator event.
        """
        output = tool_call_result.get("full_output") or tool_call_result.get(
            "output", ""
        )
        if not output:
            return

        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Could not parse web_search output as JSON")
            return

        # Handle both list format and error format
        results: List[Dict[str, Any]] = []
        if isinstance(parsed, list):
            results = parsed
        elif isinstance(parsed, dict) and "results" in parsed:
            results = parsed.get("results", [])

        for result in results:
            url = result.get("url", "")
            title = result.get("title", "")
            snippet = result.get("snippet", "")

            if not url:
                continue

            # Build a source dict matching the SourceNode API schema
            # (text, score, metadata) -- same structure as ChatService.extract_sources
            source_dict: Dict[str, Any] = {
                "text": snippet or title,
                "score": None,
                "metadata": {
                    "source_url": url,
                    "display_name": title,
                    "doc_type": "web",
                    "fetch_status": "found",  # Search result, not yet fetched
                    "content_chars": len(snippet),
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "search_query": result.get("query", ""),
                },
            }
            self._web_sources.append(source_dict)

    def _promote_web_source(self, tool_call_result: Dict[str, Any]) -> None:
        """Update a web source's status after a fetch_page completes.

        Finds the matching source in ``self._web_sources`` by URL and promotes
        its ``fetch_status`` from ``"found"`` to ``"success"`` (or ``"failed"``
        on error). Also updates ``content_chars`` from the fetched content.

        Args:
            tool_call_result: The tool_call_result dict from the fetch_page event.
        """
        url = (tool_call_result.get("params") or {}).get("url", "")
        if not url:
            return

        is_error = tool_call_result.get("is_error", False)
        output = tool_call_result.get("output", "")

        for source in self._web_sources:
            meta = source.get("metadata", {})
            if meta.get("url") == url:
                if is_error:
                    meta["fetch_status"] = "failed"
                else:
                    meta["fetch_status"] = "success"
                    meta["content_chars"] = len(output)
                break

    def _derive_confidence(self) -> str:
        """Derive the overall confidence level for the response.

        Logic:
        - If RAG was called and produced a confidence level, use it.
        - If RAG was not called (tool-free or web-only response), default
          to "high" since the LLM is responding directly or from web data.
        - If RAG was called but produced no confidence, default to "normal".

        Returns:
            Confidence level string ("high", "normal", "low", "none").
        """
        if self._rag_called:
            return self._rag_confidence or "normal"
        return "high"
