"""Token streaming utilities for LLM responses."""

import queue
import threading
from typing import Callable, List, Optional, Tuple

import streamlit as st

from tensortruth import convert_latex_delimiters
from tensortruth.app_utils import get_random_generating_message
from tensortruth.code_execution.parser import CodeBlock, CodeBlockParser


def stream_response_with_spinner(
    stream_generator_func: Callable, spinner_message: Optional[str] = None
) -> Tuple[str, Optional[Exception]]:
    """Stream tokens from a generator function with a spinner UI.

    This function handles the complex threading logic for streaming tokens
    from an LLM response while showing a spinner and updating the UI in real-time.

    Args:
        stream_generator_func: Function that returns an iterator of tokens
            Should be a callable that takes no arguments
        spinner_message: Optional custom spinner message (uses random if None)

    Returns:
        Tuple of (full_response_text, error_if_any)
    """
    token_queue = queue.Queue()
    streaming_done = threading.Event()
    error_holder = {"error": None}

    def stream_tokens_in_background():
        """Background thread that pulls tokens from generator."""
        try:
            token_gen = stream_generator_func()
            for token in token_gen:
                token_queue.put(token)
            streaming_done.set()
        except StopIteration:
            streaming_done.set()
        except Exception as e:
            error_holder["error"] = e
            streaming_done.set()

    # Start background thread
    stream_thread = threading.Thread(target=stream_tokens_in_background, daemon=True)
    stream_thread.start()

    # Display tokens as they arrive
    full_response = ""
    spinner_placeholder = st.empty()
    response_placeholder = st.empty()

    message = spinner_message if spinner_message else get_random_generating_message()

    with spinner_placeholder:
        with st.spinner(message):
            while not streaming_done.is_set() or not token_queue.empty():
                try:
                    token = token_queue.get(timeout=0.05)  # 50ms polling
                    if token is not None:
                        full_response += token
                        response_placeholder.markdown(
                            convert_latex_delimiters(full_response)
                        )
                except queue.Empty:
                    continue

    # Clear spinner after streaming completes
    spinner_placeholder.empty()

    return full_response, error_holder["error"]


def _stream_llm_with_thinking(
    response_stream, spinner_placeholder, content_placeholder
) -> Tuple[str, str, List[CodeBlock]]:
    """Common logic for streaming LLM responses with thinking token extraction.

    Args:
        response_stream: Iterator of ChatResponse chunks from LLM
        spinner_placeholder: Streamlit placeholder for spinner (will be
            replaced by thinking if present)
        content_placeholder: Streamlit placeholder for content display

    Returns:
        Tuple of (content_accumulated, thinking_accumulated, code_blocks)
    """
    thinking_accumulated = ""
    content_accumulated = ""
    thinking_placeholder = None
    code_parser = CodeBlockParser()

    for chunk in response_stream:
        # Extract and display thinking delta
        thinking_delta = chunk.additional_kwargs.get("thinking_delta", None)
        if thinking_delta:
            # First thinking token - replace spinner with thinking display
            if thinking_placeholder is None:
                spinner_placeholder.empty()
                thinking_placeholder = st.empty()

            thinking_accumulated += thinking_delta
            # Use markdown with custom CSS class for smaller font
            thinking_placeholder.markdown(
                f"""<div class="thinking-content">

**🧠 Reasoning:**

{convert_latex_delimiters(thinking_accumulated)}

</div>""",
                unsafe_allow_html=True,
            )

        # Extract and display content delta
        if chunk.delta:
            content_accumulated += chunk.delta
            code_parser.feed_token(chunk.delta)  # Parse for code blocks
            content_placeholder.markdown(convert_latex_delimiters(content_accumulated))

    # Finalize code block parsing and get all blocks
    code_parser.finalize()
    all_blocks = code_parser.get_all_blocks()

    return content_accumulated, thinking_accumulated, all_blocks


def stream_rag_response(
    synthesizer, prompt: str, context_nodes, engine=None
) -> Tuple[str, Optional[Exception], Optional[str], List[CodeBlock]]:
    """Stream a RAG response using the synthesizer.

    Args:
        synthesizer: LlamaIndex synthesizer instance
        prompt: User query
        context_nodes: Retrieved context nodes
        engine: Optional chat engine (for accessing persistent memory)

    Returns:
        Tuple of (full_response_text, error_if_any, thinking_text, code_blocks)
    """
    content_accumulated = ""
    thinking_accumulated = ""
    code_blocks = []
    error = None

    spinner_placeholder = st.empty()
    content_placeholder = st.empty()

    try:
        with spinner_placeholder:
            with st.spinner(get_random_generating_message()):
                # Try to stream directly from LLM to access thinking tokens
                if hasattr(synthesizer, "_llm") and hasattr(
                    synthesizer._llm, "stream_chat"
                ):
                    from llama_index.core.base.llms.generic_utils import (
                        messages_to_history_str,
                    )
                    from llama_index.core.base.llms.types import (
                        ChatMessage,
                        MessageRole,
                    )

                    # Get chat history from ENGINE (not synthesizer - synthesizer is ephemeral!)
                    chat_history = []
                    if engine and hasattr(engine, "_memory") and engine._memory:
                        chat_history = list(engine._memory.get())

                    print("\n=== RAG MODE - Chat History (from engine) ===")
                    print(f"Total messages in history: {len(chat_history)}")
                    for i, msg in enumerate(chat_history):
                        role_name = (
                            msg.role.value
                            if hasattr(msg.role, "value")
                            else str(msg.role)
                        )
                        content_preview = (
                            msg.content[:100] if len(msg.content) > 100 else msg.content
                        )
                        print(f"  [{i}] {role_name}: {content_preview}...")
                        if role_name == "system":
                            print(f"      FULL SYSTEM MESSAGE: {msg.content}")
                    print("=" * 80)

                    # Build context string from retrieved documents
                    context_str = ""
                    if context_nodes and len(context_nodes) > 0:
                        context_str = "\n\n".join(
                            [n.get_content() for n in context_nodes]
                        )

                    # Format chat history as string for the template
                    chat_history_str = (
                        messages_to_history_str(chat_history) if chat_history else ""
                    )

                    # Choose appropriate template based on confidence/sources
                    # Import all template options
                    from tensortruth.rag_engine import (
                        CUSTOM_CONTEXT_PROMPT_NO_SOURCES,
                        CUSTOM_CONTEXT_PROMPT_TEMPLATE,
                    )

                    if context_nodes and len(context_nodes) > 0:
                        template = CUSTOM_CONTEXT_PROMPT_TEMPLATE
                    else:
                        template = CUSTOM_CONTEXT_PROMPT_NO_SOURCES

                    # Format the system prompt with all context
                    system_content = template.format(
                        context_str=context_str,
                        chat_history=chat_history_str,
                        query_str=prompt,
                    )

                    # Build message list: SYSTEM + HISTORY + USER
                    messages = (
                        [ChatMessage(role=MessageRole.SYSTEM, content=system_content)]
                        + chat_history
                        + [ChatMessage(role=MessageRole.USER, content=prompt)]
                    )

                    print("\n=== FINAL MESSAGES TO LLM ===")
                    print(f"Total messages: {len(messages)}")
                    for i, msg in enumerate(messages):
                        role_name = (
                            msg.role.value
                            if hasattr(msg.role, "value")
                            else str(msg.role)
                        )
                        content_len = len(msg.content)
                        print(f"  [{i}] {role_name}: {content_len} chars")
                        if i == 0:  # System message
                            print(
                                f"      System prompt preview: {msg.content[:200]}..."
                            )
                    print("=" * 80)

                    # Stream from LLM with thinking token support
                    response_stream = synthesizer._llm.stream_chat(messages)
                    content_accumulated, thinking_accumulated, code_blocks = (
                        _stream_llm_with_thinking(
                            response_stream, spinner_placeholder, content_placeholder
                        )
                    )
                else:
                    # Fallback to synthesizer (no thinking tokens available)
                    code_parser = CodeBlockParser()
                    response = synthesizer.synthesize(prompt, context_nodes)
                    for token in response.response_gen:
                        content_accumulated += token
                        code_parser.feed_token(token)
                        content_placeholder.markdown(
                            convert_latex_delimiters(content_accumulated)
                        )
                    code_blocks = code_parser.get_all_blocks()

        spinner_placeholder.empty()

    except Exception as e:
        error = e

    return (
        content_accumulated,
        error,
        thinking_accumulated if thinking_accumulated else None,
        code_blocks,
    )


def stream_simple_llm_response(
    llm, chat_history
) -> Tuple[str, Optional[Exception], Optional[str], List[CodeBlock]]:
    """Stream a response from Ollama without RAG.

    Args:
        llm: LlamaIndex Ollama LLM instance
        chat_history: List of ChatMessage objects

    Returns:
        Tuple of (full_response_text, error_if_any, thinking_text, code_blocks)
    """
    content_accumulated = ""
    thinking_accumulated = ""
    code_blocks = []
    error = None

    spinner_placeholder = st.empty()
    content_placeholder = st.empty()

    try:
        with spinner_placeholder:
            with st.spinner(get_random_generating_message()):
                response_stream = llm.stream_chat(chat_history)
                content_accumulated, thinking_accumulated, code_blocks = (
                    _stream_llm_with_thinking(
                        response_stream, spinner_placeholder, content_placeholder
                    )
                )

        spinner_placeholder.empty()

    except Exception as e:
        error = e

    return (
        content_accumulated,
        error,
        thinking_accumulated if thinking_accumulated else None,
        code_blocks,
    )
