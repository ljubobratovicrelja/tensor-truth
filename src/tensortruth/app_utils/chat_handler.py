"""Unified chat response handler for RAG and simple LLM modes."""

import time
from typing import Tuple

import streamlit as st
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode

from tensortruth.app_utils.chat_utils import build_chat_history
from tensortruth.app_utils.helpers import get_random_rag_processing_message
from tensortruth.app_utils.rendering import (
    extract_source_metadata,
    render_low_confidence_warning,
    render_message_footer,
)
from tensortruth.app_utils.session import save_sessions
from tensortruth.app_utils.streaming import (
    stream_rag_response,
    stream_simple_llm_response,
)
from tensortruth.rag_engine import (
    CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE,
    CUSTOM_CONTEXT_PROMPT_NO_SOURCES,
    NO_CONTEXT_FALLBACK_CONTEXT,
    get_llm,
)


def _check_confidence_and_adjust_prompt(
    synthesizer, context_nodes, confidence_threshold: float
) -> Tuple[bool, bool, list]:
    """Check confidence threshold and adjust synthesizer if needed.

    Args:
        synthesizer: LlamaIndex synthesizer
        context_nodes: Retrieved context nodes
        confidence_threshold: Minimum confidence score

    Returns:
        Tuple of (low_confidence_warning, has_real_sources, adjusted_context_nodes)
    """
    low_confidence_warning = False
    has_real_sources = True

    # Case 1: Nodes exist and we have a threshold
    if context_nodes and len(context_nodes) > 0 and confidence_threshold > 0:
        best_score = max(
            (node.score for node in context_nodes if node.score), default=0.0
        )

        if best_score < confidence_threshold:
            synthesizer._context_prompt_template = CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE
            render_low_confidence_warning(
                best_score, confidence_threshold, has_sources=True
            )
            low_confidence_warning = True

    # Case 2: No nodes retrieved
    elif not context_nodes or len(context_nodes) == 0:
        render_low_confidence_warning(0.0, confidence_threshold, has_sources=False)

        warning_node = NodeWithScore(
            node=TextNode(text=NO_CONTEXT_FALLBACK_CONTEXT), score=0.0
        )
        context_nodes = [warning_node]
        low_confidence_warning = True
        has_real_sources = False

        synthesizer._context_prompt_template = CUSTOM_CONTEXT_PROMPT_NO_SOURCES

    return low_confidence_warning, has_real_sources, context_nodes


def _handle_rag_mode(
    engine, prompt: str, params: dict, modules: list, has_pdf_index: bool
) -> Tuple[str, dict]:
    """Handle RAG mode: retrieval, confidence checking, streaming, and UI rendering.

    Args:
        engine: RAG engine instance
        prompt: User prompt
        params: Session parameters
        modules: Active modules
        has_pdf_index: Whether session has PDF index

    Returns:
        Tuple of (thinking, message_data_dict)
    """
    # Phase 1: RAG Retrieval (uses engine's memory for query condensing)
    with st.spinner(get_random_rag_processing_message()):
        synthesizer, condensed_query, context_nodes = engine._run_c3(
            prompt, streaming=True
        )

    # Phase 2: Confidence checking and prompt adjustment
    confidence_threshold = params.get("confidence_cutoff", 0.0)
    low_confidence_warning, has_real_sources, context_nodes = (
        _check_confidence_and_adjust_prompt(
            synthesizer, context_nodes, confidence_threshold
        )
    )

    # Capture debug data (if enabled)
    debug_data = None
    if st.session_state.get("debug_context", False):
        best_score = max(
            (node.score for node in context_nodes if node.score), default=0.0
        )

        # Build the ACTUAL context string that goes to the LLM (same as streaming.py:148)
        actual_context_str = (
            "\n\n".join([n.get_content() for n in context_nodes])
            if has_real_sources
            else ""
        )
        actual_formatted_prompt = (
            (
                f"Context information:\n{actual_context_str}\n\n"
                f"Query: {prompt}\n\nAnswer:"
            )
            if has_real_sources
            else f"Query: {prompt}\n\nAnswer:"
        )

        # Get chat history that will be sent to LLM (same as streaming.py:155-157)
        # Check both synthesizer and engine for memory
        chat_history_messages = []
        if hasattr(synthesizer, "_memory") and synthesizer._memory:
            chat_history_messages = list(synthesizer._memory.get())
        elif hasattr(engine, "_memory") and engine._memory:
            chat_history_messages = list(engine._memory.get())

        # Build complete conversation view (history + current prompt)
        complete_conversation = ""
        if chat_history_messages:
            complete_conversation += (
                f"=== CHAT HISTORY ({len(chat_history_messages)} messages) ===\n\n"
            )
            for idx, msg in enumerate(chat_history_messages, 1):
                role = (
                    str(msg.role).split(".")[-1]
                    if hasattr(msg.role, "__str__")
                    else str(msg.role)
                )
                complete_conversation += f"Message {idx} [{role}]:\n{msg.content}\n\n"
            complete_conversation += "=== CURRENT PROMPT (sent as USER role with system-formatted context) ===\n\n"
        else:
            complete_conversation += "=== NO CHAT HISTORY (First message or memory not persisted) ===\n\n=== CURRENT PROMPT (sent as USER role with system-formatted context) ===\n\n"
        complete_conversation += actual_formatted_prompt

        # Store debug data for message history
        debug_data = {
            "mode": "rag",
            "best_score": float(
                best_score
            ),  # Convert numpy.float32 to Python float for JSON
            "confidence_threshold": float(confidence_threshold),
            "has_real_sources": has_real_sources,
            "num_nodes": len(context_nodes) if has_real_sources else 0,
            "node_scores": (
                [
                    {
                        "display_name": extract_source_metadata(node, is_node=True)[
                            "display_name"
                        ],
                        "score": (
                            float(node.score) if node.score else 0.0
                        ),  # Convert numpy.float32 to Python float
                    }
                    for node in context_nodes
                ]
                if has_real_sources
                else []
            ),
            "user_query": prompt,  # Original user input
            "condensed_query": (
                str(condensed_query) if condensed_query else prompt
            ),  # Query after LLM condensing (used for retrieval)
            "actual_context_str": actual_context_str,  # RAW text sent to LLM
            "actual_formatted_prompt": actual_formatted_prompt,  # Current prompt with context
            "complete_conversation": complete_conversation,  # Full conversation: history + current prompt
        }

        # Render live during streaming
        from tensortruth.app_utils.rendering_debug import (
            render_debug_context as render_debug_rag,
        )

        render_debug_rag(
            context_nodes=context_nodes,
            best_score=best_score,
            confidence_threshold=confidence_threshold,
            synthesizer=synthesizer,
            low_confidence_warning=low_confidence_warning,
            has_real_sources=has_real_sources,
            actual_context_str=actual_context_str,
            actual_formatted_prompt=actual_formatted_prompt,
            complete_conversation=complete_conversation,
            user_query=prompt,
            condensed_query=str(condensed_query) if condensed_query else prompt,
        )

    # Phase 3: Stream response
    start_time = time.time()
    full_response, error, thinking = stream_rag_response(
        synthesizer, prompt, context_nodes
    )
    if error:
        raise error
    elapsed = time.time() - start_time

    # Phase 4: Extract sources and render UI
    source_data = []
    if has_real_sources:
        source_data = [
            extract_source_metadata(node, is_node=True) for node in context_nodes
        ]

    render_message_footer(
        sources_or_nodes=context_nodes if has_real_sources else None,
        is_nodes=True,
        time_taken=elapsed,
        low_confidence=low_confidence_warning,
        modules=modules,
        has_pdf_index=has_pdf_index,
    )

    # Phase 5: Update engine memory (only content, no metadata)
    engine._memory.put(ChatMessage(content=prompt, role=MessageRole.USER))
    engine._memory.put(ChatMessage(content=full_response, role=MessageRole.ASSISTANT))

    # Build message data with metadata (for UI display and history storage only)
    # NOTE: Metadata fields below are NEVER passed to LLM inference
    message_data = {
        "role": "assistant",
        "content": full_response,
        "sources": source_data,  # UI metadata only
        "time_taken": elapsed,  # UI metadata only
        "low_confidence": low_confidence_warning,  # UI metadata only
    }

    # Add debug data if captured (UI metadata only, never passed to LLM)
    if debug_data:
        message_data["debug_data"] = debug_data

    return thinking, message_data


def _handle_simple_llm_mode(
    session: dict, params: dict, prompt: str
) -> Tuple[str, dict]:
    """Handle simple LLM mode: loading, streaming, and UI rendering.

    Args:
        session: Current session dictionary
        params: Session parameters
        prompt: User prompt (for debug display)

    Returns:
        Tuple of (thinking, message_data_dict)
    """
    # Ensure LLM is loaded with current config
    simple_llm_config = (
        params.get("model"),
        params.get("temperature"),
        params.get("llm_device"),
        params.get("max_tokens"),
    )

    if (
        "simple_llm" not in st.session_state
        or st.session_state.get("simple_llm_config") != simple_llm_config
    ):
        st.session_state.simple_llm = get_llm(params)
        st.session_state.simple_llm_config = simple_llm_config

    llm = st.session_state.simple_llm
    chat_history = build_chat_history(session["messages"])

    # Capture debug data (if enabled)
    debug_data = None
    if st.session_state.get("debug_context", False):
        from tensortruth.app_utils.rendering_debug import render_debug_simple_llm

        # Get system prompt
        system_prompt = None
        if hasattr(llm, "system_prompt"):
            system_prompt = llm.system_prompt
        elif hasattr(llm, "_system_prompt"):
            system_prompt = llm._system_prompt

        # Build chat history summary for storage
        chat_history_summary = [
            {
                "role": (
                    str(msg.role).split(".")[-1]
                    if hasattr(msg.role, "__str__")
                    else str(msg.role)
                ),
                "content": msg.content,
            }
            for msg in chat_history
        ]

        # Store debug data for message history
        debug_data = {
            "mode": "simple_llm",
            "model": params.get("model", "Unknown"),
            "temperature": float(
                params.get("temperature", 0.7)
            ),  # Ensure Python float for JSON
            "max_tokens": int(params.get("max_tokens", 2048)),
            "device": params.get("llm_device", "auto"),
            "chat_history": chat_history_summary,
            "prompt": prompt,
            "system_prompt": str(system_prompt) if system_prompt else None,
        }

        # Render live during streaming
        render_debug_simple_llm(
            model_name=debug_data["model"],
            temperature=debug_data["temperature"],
            max_tokens=debug_data["max_tokens"],
            device=debug_data["device"],
            chat_history_summary=chat_history_summary,
            prompt=prompt,
            system_prompt=debug_data["system_prompt"],
        )

    # Stream response
    start_time = time.time()
    full_response, error, thinking = stream_simple_llm_response(llm, chat_history)
    if error:
        raise error
    elapsed = time.time() - start_time

    # Render simple footer
    st.caption(f"⏱️ {elapsed:.2f}s")

    # Build message data with metadata (for UI display and history storage only)
    # NOTE: Metadata fields below are NEVER passed to LLM inference
    message_data = {
        "role": "assistant",
        "content": full_response,
        "time_taken": elapsed,  # UI metadata only
    }

    # Add debug data if captured (UI metadata only, never passed to LLM)
    if debug_data:
        message_data["debug_data"] = debug_data

    return thinking, message_data


def handle_chat_response(
    prompt: str,
    session: dict,
    params: dict,
    current_id: str,
    sessions_file: str,
    modules: list,
    has_pdf_index: bool,
    engine=None,
) -> None:
    """Unified handler for chat responses in both RAG and simple LLM modes.

    Orchestrates the response flow: mode selection, streaming, message building,
    session updates, and title generation.

    Args:
        prompt: User's input prompt
        session: Current session dictionary
        params: Session parameters
        current_id: Current session ID
        sessions_file: Path to sessions file
        modules: List of active modules
        has_pdf_index: Whether session has PDF index
        engine: RAG engine (None for simple LLM mode)
    """
    try:
        # Generate title if needed (first message only, before mode-specific logic)
        if session.get("title_needs_update", False):
            with st.spinner("Generating title..."):
                from tensortruth.app_utils.session import update_title

                update_title(current_id, prompt, params.get("model"), sessions_file)

        # Delegate to mode-specific handler
        if engine:
            thinking, message_data = _handle_rag_mode(
                engine, prompt, params, modules, has_pdf_index
            )
        else:
            thinking, message_data = _handle_simple_llm_mode(session, params, prompt)

        # Add thinking if present
        if thinking:
            message_data["thinking"] = thinking

        # Save message and session
        session["messages"].append(message_data)
        save_sessions(sessions_file)

        st.rerun()

    except Exception as e:
        error_type = "Engine Error" if engine else "LLM Error"
        st.error(f"{error_type}: {e}")
