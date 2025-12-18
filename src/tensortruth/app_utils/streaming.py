"""Token streaming utilities for LLM responses."""

import queue
import threading
from typing import Callable, Optional, Tuple

import streamlit as st

from tensortruth import convert_latex_delimiters
from tensortruth.app_utils import get_random_generating_message


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


def stream_rag_response(
    synthesizer, prompt: str, context_nodes
) -> Tuple[str, Optional[Exception]]:
    """Stream a RAG response using the synthesizer.

    Args:
        synthesizer: LlamaIndex synthesizer instance
        prompt: User query
        context_nodes: Retrieved context nodes

    Returns:
        Tuple of (full_response_text, error_if_any)
    """

    def generate_tokens():
        """Generator function for RAG tokens."""
        response = synthesizer.synthesize(prompt, context_nodes)
        for token in response.response_gen:
            yield token

    return stream_response_with_spinner(generate_tokens)


def stream_simple_llm_response(llm, chat_history) -> Tuple[str, Optional[Exception]]:
    """Stream a response from Ollama without RAG.

    Args:
        llm: LlamaIndex Ollama LLM instance
        chat_history: List of ChatMessage objects

    Returns:
        Tuple of (full_response_text, error_if_any)
    """

    def generate_tokens():
        """Generator function for simple LLM tokens."""
        response = llm.stream_chat(chat_history)
        for token in response:
            yield str(token.delta)

    return stream_response_with_spinner(generate_tokens)
