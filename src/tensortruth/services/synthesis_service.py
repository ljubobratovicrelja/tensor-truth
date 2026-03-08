"""Final response synthesis for the orchestrator agentic loop.

After the orchestrator's tool-routing phase collects results, the synthesis
service generates the final user-facing answer using a dedicated system prompt
optimised for comprehensive, source-citing responses.

The LLM instance is cached as a module-level singleton keyed by
(model, base_url), following the same pattern as ``get_orchestrator_llm()``.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from tensortruth.core.prompts import current_date_context
from tensortruth.core.providers import get_tool_llm as _providers_get_tool_llm
from tensortruth.core.providers import resolve_model_from_params
from tensortruth.core.providers import resolve_thinking as _providers_resolve_thinking
from tensortruth.services.models import ToolProgress
from tensortruth.services.orchestrator_service import (
    ModuleDescription,
    OrchestratorEvent,
)
from tensortruth.services.orchestrator_tool_wrappers import ProgressEmitter

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_synthesis_service: Optional["SynthesisService"] = None
_synthesis_service_key: Optional[Tuple[str, str, int, object, Optional[str]]] = None


def get_synthesis_service(
    model: str,
    base_url: str,
    context_window: int,
    thinking: Optional[Union[bool, Literal["low", "medium", "high"]]] = None,
    provider_id: Optional[str] = None,
) -> "SynthesisService":
    """Get or create a cached SynthesisService singleton.

    Keyed by ``(model, base_url, context_window, thinking)``.  When the key
    changes the old instance is discarded and a new one is created.

    Args:
        model: Model name.
        base_url: Server base URL.
        context_window: Context window size in tokens.
        thinking: User thinking preference (``True``, ``False``, ``None``, or
            a budget string like ``"low"``).  ``None`` means auto-detect.
        provider_id: Provider ID for provider-aware model resolution.

    Returns:
        Cached SynthesisService instance.
    """
    global _synthesis_service, _synthesis_service_key

    key = (model, base_url, context_window, thinking, provider_id)
    if _synthesis_service is not None and _synthesis_service_key == key:
        return _synthesis_service

    _synthesis_service = SynthesisService(
        model, base_url, context_window, thinking, provider_id=provider_id
    )
    _synthesis_service_key = key

    logger.info(
        "Created SynthesisService singleton: model=%s, base_url=%s, " "thinking=%s",
        model,
        base_url,
        _synthesis_service.thinking_supported,
    )

    return _synthesis_service


# ---------------------------------------------------------------------------
# SynthesisService
# ---------------------------------------------------------------------------


class SynthesisService:
    """Final response synthesis for the orchestrator agentic loop.

    Cached singleton keyed by ``(model, base_url)``.  Handles both thinking
    and non-thinking models transparently.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        context_window: int,
        thinking: Optional[Union[bool, Literal["low", "medium", "high"]]] = None,
        provider_id: Optional[str] = None,
    ):
        self._model = model
        self._base_url = base_url
        self._context_window = context_window
        self._thinking_pref = thinking

        # Resolve model reference for provider-aware LLM creation
        self._model_ref = resolve_model_from_params({"provider_id": provider_id}, model)
        resolved = _providers_resolve_thinking(self._model_ref, thinking)
        self._thinking_supported = bool(resolved)
        self._llm = self._create_llm()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def thinking_supported(self) -> bool:
        """Whether the model supports thinking/reasoning tokens."""
        return self._thinking_supported

    # ------------------------------------------------------------------
    # LLM creation
    # ------------------------------------------------------------------

    def _create_llm(self) -> "LLM":
        """Get the shared tool LLM singleton for synthesis."""
        resolved = _providers_resolve_thinking(self._model_ref, self._thinking_pref)
        return _providers_get_tool_llm(
            self._model_ref, self._context_window, thinking=resolved
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        module_descriptions: Optional[List[ModuleDescription]] = None,
        custom_instructions: Optional[str] = None,
        project_metadata: Optional[str] = None,
    ) -> str:
        """Compose the synthesis-specific system prompt.

        This prompt is optimised for generating comprehensive, well-structured
        answers that cite sources.  It does NOT contain tool routing guidance,
        tool lists, or "be brief" instructions.

        Args:
            module_descriptions: Descriptions of active indexed modules.
            custom_instructions: Session-level custom instructions.
            project_metadata: Project-level metadata string.

        Returns:
            Complete system prompt string.
        """
        sections: List[str] = []

        # --- Current date (temporal grounding) ---
        sections.append(current_date_context())

        # --- Role ---
        sections.append(
            "You are an intelligent assistant that synthesizes information "
            "from tool results to provide comprehensive, well-structured "
            "answers. You have been given the results of tool calls that "
            "gathered information relevant to the user's question."
        )

        # --- Indexed modules ---
        if module_descriptions:
            module_lines = [
                f"- {mod.name}: {mod.display_name} ({mod.doc_type})"
                for mod in module_descriptions
            ]
            modules_block = "\n".join(module_lines)
            sections.append(
                "The following knowledge modules were available:\n" f"{modules_block}"
            )

        # --- Response formatting rules ---
        sections.append(
            "Response formatting rules:\n"
            "- Write a comprehensive answer based on the tool results provided.\n"
            "- Use markdown formatting (headings, lists, code blocks) for readability.\n"
            "- ALWAYS cite sources using numbered references matching the "
            "Source Reference list (e.g. [1], [2]).\n"
            "- For web sources, include a clickable link on first mention: "
            "[Title](URL) [1]\n"
            "- For knowledge base sources, reference by name: "
            "according to the documentation [1]\n"
            "- Do NOT add a Sources section at the end of your response. "
            "Sources are displayed separately in the UI.\n"
            "- Only cite sources from the Source Reference list. "
            "Do not invent sources.\n"
            "- If sources conflict, note which source says what with citations.\n"
            "- If the tool results are insufficient to fully answer the "
            "question, say so and provide what you can.\n"
            "- Source content may contain images as ![alt](url) markdown syntax. "
            "Include up to 3 relevant images that directly illustrate key points. "
            "Place images near the text they support. "
            "Only use image URLs from the provided sources; never invent URLs. "
            "If no relevant images are available, do not mention their absence."
        )

        # --- Project metadata ---
        if project_metadata:
            sections.append(f"Project context:\n{project_metadata}")

        # --- Custom instructions ---
        if custom_instructions:
            sections.append(
                f"Additional instructions from the user:\n{custom_instructions}"
            )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        prompt: str,
        chat_history: List[ChatMessage],
        tool_results: List[str],
        module_descriptions: Optional[List[ModuleDescription]] = None,
        custom_instructions: Optional[str] = None,
        project_metadata: Optional[str] = None,
        progress_emitter: Optional[ProgressEmitter] = None,
        source_reference: Optional[str] = None,
    ) -> AsyncGenerator[OrchestratorEvent, None]:
        """Synthesize the final response from tool results.

        Streams the synthesis LLM's response, yielding thinking and token
        events.

        Args:
            prompt: The original user prompt.
            chat_history: Budgeted chat history as ChatMessage objects.
            tool_results: Formatted tool result strings from the routing phase.
            module_descriptions: Descriptions of active indexed modules.
            custom_instructions: Session-level custom instructions.
            project_metadata: Project-level metadata string.
            progress_emitter: Optional callback for progress events.
            source_reference: Optional structured source reference block for
                citation guidance. When provided, appended to the user message
                so the LLM can produce numbered citations.

        Yields:
            OrchestratorEvent instances (thinking and token events).
        """
        # Emit progress phase
        if progress_emitter and self._thinking_supported:
            progress_emitter(
                ToolProgress(
                    tool_id="orchestrator",
                    phase="thinking",
                    message="Thinking...",
                )
            )
        elif progress_emitter:
            progress_emitter(
                ToolProgress(
                    tool_id="orchestrator",
                    phase="generating",
                    message="Generating response...",
                )
            )

        # Build synthesis system prompt
        system_prompt = self._build_system_prompt(
            module_descriptions=module_descriptions,
            custom_instructions=custom_instructions,
            project_metadata=project_metadata,
        )

        # Build synthesis user message with tool context
        tool_context = "\n\n".join(tool_results)
        source_ref_block = f"\n\n{source_reference}\n" if source_reference else ""
        synthesis_user_content = (
            f"{prompt}\n\n"
            f"--- Tool Results ---\n"
            f"{tool_context}\n"
            f"--- End Tool Results ---"
            f"{source_ref_block}\n\n"
            f"Using the tool results above, provide a comprehensive answer "
            f"to the user's question. Cite sources using [N] references "
            f"matching the Source Reference list."
        )

        # Compose messages
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)]
        messages.extend(chat_history)
        messages.append(
            ChatMessage(role=MessageRole.USER, content=synthesis_user_content)
        )

        # Stream synthesis using native async streaming — yields chunks
        # as they arrive from Ollama so the user sees progressive output.
        sent_generating_phase = not self._thinking_supported

        try:
            async for chunk in await self._llm.astream_chat(messages):
                thinking_delta = chunk.additional_kwargs.get("thinking_delta")
                if thinking_delta:
                    yield OrchestratorEvent(thinking=thinking_delta)
                elif chunk.delta:
                    if not sent_generating_phase and progress_emitter:
                        progress_emitter(
                            ToolProgress(
                                tool_id="orchestrator",
                                phase="generating",
                                message="Generating response...",
                            )
                        )
                        sent_generating_phase = True
                    yield OrchestratorEvent(token=chunk.delta)

        except Exception as e:
            logger.error("Synthesis failed: %s", e, exc_info=True)
            error_msg = f"I encountered an error generating the response: {e}"
            yield OrchestratorEvent(token=error_msg)
