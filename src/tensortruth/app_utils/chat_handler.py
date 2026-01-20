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
) -> Tuple[str | None, dict]:
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
    # Phase 1: RAG Retrieval
    with st.spinner(get_random_rag_processing_message()):
        synthesizer, _, context_nodes = engine._run_c3(
            prompt, chat_history=None, streaming=True
        )

    # Phase 2: Confidence checking and prompt adjustment
    confidence_threshold = params.get("confidence_cutoff", 0.0)
    low_confidence_warning, has_real_sources, context_nodes = (
        _check_confidence_and_adjust_prompt(
            synthesizer, context_nodes, confidence_threshold
        )
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

    # Phase 5: Update engine memory
    engine._memory.put(ChatMessage(content=prompt, role=MessageRole.USER))
    engine._memory.put(ChatMessage(content=full_response, role=MessageRole.ASSISTANT))

    # Build message data
    message_data = {
        "role": "assistant",
        "content": full_response,
        "sources": source_data,
        "time_taken": elapsed,
        "low_confidence": low_confidence_warning,
    }

    return thinking, message_data


def _handle_simple_llm_mode(session: dict, params: dict) -> Tuple[str | None, dict]:
    """Handle simple LLM mode: loading, streaming, and UI rendering.

    Args:
        session: Current session dictionary
        params: Session parameters

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

    # Stream response
    start_time = time.time()
    full_response, error, thinking = stream_simple_llm_response(llm, chat_history)
    if error:
        raise error
    elapsed = time.time() - start_time

    # Render simple footer
    st.caption(f"⏱️ {elapsed:.2f}s")

    # Build message data
    message_data = {
        "role": "assistant",
        "content": full_response,
        "time_taken": elapsed,
    }

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

                update_title(
                    current_id, prompt, params.get("model", "llama3.2"), sessions_file
                )

        # Delegate to mode-specific handler
        if engine:
            thinking, message_data = _handle_rag_mode(
                engine, prompt, params, modules, has_pdf_index
            )
        else:
            thinking, message_data = _handle_simple_llm_mode(session, params)

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
