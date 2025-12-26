"""Chat mode - Main conversation interface with RAG."""

import os
import threading
import time

import streamlit as st

from tensortruth.app_utils.chat_utils import build_chat_history, preserve_chat_history
from tensortruth.app_utils.config import compute_config_hash
from tensortruth.app_utils.helpers import (
    free_memory,
    get_available_modules,
    get_random_rag_processing_message,
)
from tensortruth.app_utils.paths import get_session_index_dir
from tensortruth.app_utils.rendering import (
    extract_source_metadata,
    render_chat_message,
    render_low_confidence_warning,
    render_message_footer,
)
from tensortruth.app_utils.session import save_sessions
from tensortruth.app_utils.setup_state import get_session_params_with_defaults
from tensortruth.app_utils.streaming import (
    stream_rag_response,
    stream_simple_llm_response,
)

from ..commands import process_command


def render_chat_mode():
    """Render the chat mode UI for conversation."""
    current_id = st.session_state.chat_data.get("current_id")
    if not current_id:
        st.session_state.mode = "setup"
        st.rerun()

    session = st.session_state.chat_data["sessions"][current_id]
    modules = session.get("modules", [])
    # Get params with config defaults as fallback
    params = get_session_params_with_defaults(session.get("params", {}))

    st.title(session.get("title", "Untitled"))
    st.caption(f"🤖 {params.get('model', 'Unknown')}")
    st.divider()
    st.empty()

    # Initialize engine loading state
    if "engine_loading" not in st.session_state:
        st.session_state.engine_loading = False
    if "engine_load_error" not in st.session_state:
        st.session_state.engine_load_error = None

    # Determine target configuration
    has_pdf_index = session.get("has_temp_index", False)
    target_config = compute_config_hash(modules, params, has_pdf_index, current_id)
    current_config = st.session_state.get("loaded_config")
    engine = st.session_state.get("engine")

    # Check if we need to load/reload the engine
    needs_loading = (modules or has_pdf_index) and (current_config != target_config)

    # Background engine loading
    if needs_loading and not st.session_state.engine_loading:
        st.session_state.engine_loading = True
        st.session_state.engine_load_error = None

        if "engine_load_event" not in st.session_state:
            st.session_state.engine_load_event = threading.Event()
        if "engine_load_result" not in st.session_state:
            st.session_state.engine_load_result = {"engine": None, "error": None}

        load_event = st.session_state.engine_load_event
        load_result = st.session_state.engine_load_result
        load_event.clear()

        def load_engine_background():
            try:
                preserved_history = preserve_chat_history(session["messages"])

                if current_config is not None:
                    free_memory()

                # Check for session index
                session_index_path = None
                if session.get("has_temp_index", False):
                    index_path = get_session_index_dir(current_id)
                    if os.path.exists(str(index_path)):
                        session_index_path = str(index_path)

                from tensortruth import load_engine_for_modules

                loaded_engine = load_engine_for_modules(
                    modules, params, preserved_history, session_index_path
                )
                load_result["engine"] = loaded_engine
                load_result["config"] = target_config
            except Exception as e:
                load_result["error"] = str(e)
            finally:
                load_event.set()

        thread = threading.Thread(target=load_engine_background, daemon=True)
        thread.start()

    # Handle engine load errors or missing modules
    if st.session_state.engine_load_error:
        st.error(f"Failed to load engine: {st.session_state.engine_load_error}")
        engine = None
    elif not modules and not has_pdf_index:
        st.info(
            "💬 Simple LLM mode (No RAG) - "
            "Use `/load <name>` to attach a knowledge base."
        )
        engine = None

    # Render message history
    messages_to_render = session["messages"]
    if st.session_state.get("skip_last_message_render", False):
        messages_to_render = session["messages"][:-1]
        st.session_state.skip_last_message_render = False

    for msg in messages_to_render:
        render_chat_message(msg, params, modules, has_pdf_index)

    # Get user input
    prompt = st.chat_input("Ask or type /cmd...")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Show tip if no messages exist
    if not session["messages"] and not prompt:
        st.caption(
            "💡 Tip: Type **/help** to see all commands. "
            "Use `/device` to manage hardware."
        )

    if prompt:
        # Wait for engine if still loading
        if st.session_state.engine_loading:
            with st.spinner("⏳ Waiting for model to finish loading..."):
                if "engine_load_event" in st.session_state:
                    event_triggered = st.session_state.engine_load_event.wait(
                        timeout=60.0
                    )

                    if not event_triggered:
                        st.error("Model loading timed out after 60 seconds")
                        st.session_state.engine_loading = False
                    else:
                        load_result = st.session_state.engine_load_result
                        if load_result.get("error"):
                            st.session_state.engine_load_error = load_result["error"]
                        elif load_result.get("engine"):
                            st.session_state.engine = load_result["engine"]
                            st.session_state.loaded_config = load_result["config"]
                        st.session_state.engine_loading = False

        # Check if background loading completed
        if (
            "engine_load_result" in st.session_state
            and not st.session_state.engine_loading
        ):
            load_result = st.session_state.engine_load_result
            if load_result.get("engine") and not st.session_state.get("engine"):
                st.session_state.engine = load_result["engine"]
                st.session_state.loaded_config = load_result["config"]
            if load_result.get("error") and not st.session_state.engine_load_error:
                st.session_state.engine_load_error = load_result["error"]

        engine = st.session_state.get("engine")

        # COMMAND PROCESSING
        if prompt.startswith("/"):
            available_mods_tuples = get_available_modules(st.session_state.index_dir)
            available_mods = [mod for mod, _ in available_mods_tuples]
            is_cmd, response, state_modifier = process_command(
                prompt, session, available_mods
            )

            if is_cmd:
                session["messages"].append({"role": "command", "content": response})

                with st.chat_message("command", avatar=":material/settings:"):
                    st.markdown(response)

                save_sessions(st.session_state.sessions_file)

                if state_modifier is not None:
                    with st.spinner("⚙️ Applying changes..."):
                        state_modifier()

                st.rerun()

        # STANDARD CHAT PROCESSING
        session["messages"].append({"role": "user", "content": prompt})
        save_sessions(st.session_state.sessions_file)

        # Check if title needs updating
        # (avoid race conditions by doing it after response)
        should_update_title = session.get("title_needs_update", False)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if engine:
                start_time = time.time()
                try:
                    # Phase 1: RAG Retrieval
                    with st.spinner(get_random_rag_processing_message()):
                        synthesizer, context_source, context_nodes = engine._run_c3(
                            prompt, chat_history=None, streaming=True
                        )

                    # Check confidence threshold
                    low_confidence_warning = False
                    has_real_sources = True  # Track if we have actual retrieved sources
                    confidence_threshold = params.get("confidence_cutoff", 0.0)

                    if (
                        context_nodes
                        and len(context_nodes) > 0
                        and confidence_threshold > 0
                    ):
                        best_score = max(
                            (node.score for node in context_nodes if node.score),
                            default=0.0,
                        )

                        if best_score < confidence_threshold:
                            from tensortruth.rag_engine import (
                                CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE,
                            )

                            synthesizer._context_prompt_template = (
                                CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE
                            )

                            render_low_confidence_warning(
                                best_score, confidence_threshold, has_sources=True
                            )
                            low_confidence_warning = True
                    elif not context_nodes or len(context_nodes) == 0:
                        from llama_index.core.schema import NodeWithScore, TextNode

                        from tensortruth.rag_engine import (
                            CUSTOM_CONTEXT_PROMPT_NO_SOURCES,
                            NO_CONTEXT_FALLBACK_CONTEXT,
                        )

                        render_low_confidence_warning(
                            0.0, confidence_threshold, has_sources=False
                        )

                        warning_node = NodeWithScore(
                            node=TextNode(text=NO_CONTEXT_FALLBACK_CONTEXT),
                            score=0.0,
                        )
                        context_nodes = [warning_node]
                        low_confidence_warning = True
                        has_real_sources = False  # No real sources, just synthetic node

                        synthesizer._context_prompt_template = (
                            CUSTOM_CONTEXT_PROMPT_NO_SOURCES
                        )

                    # Phase 2: LLM Streaming
                    full_response, error, thinking, code_blocks = stream_rag_response(
                        synthesizer, prompt, context_nodes
                    )

                    if error:
                        raise error

                    elapsed = time.time() - start_time

                    # Phase 3: Code Execution (if enabled and blocks found)
                    execution_results = []
                    if code_blocks and st.session_state.get(
                        "code_execution_enabled", True
                    ):
                        from tensortruth.code_execution import ExecutionOrchestrator

                        try:
                            orchestrator = ExecutionOrchestrator()
                            execution_results = orchestrator.execute_blocks(
                                session_id=current_id,
                                code_blocks=code_blocks,
                                timeout=st.session_state.get("code_exec_timeout", 30),
                                enabled=True,
                            )
                        except Exception as exec_error:
                            st.warning(f"Code execution error: {exec_error}")

                    # Extract source metadata (only for real sources)
                    source_data = []
                    if has_real_sources:
                        for node in context_nodes:
                            metadata = extract_source_metadata(node, is_node=True)
                            source_data.append(metadata)

                    # Render footer (only show sources if we have real ones)
                    render_message_footer(
                        sources_or_nodes=context_nodes if has_real_sources else None,
                        is_nodes=True,
                        time_taken=elapsed,
                        low_confidence=low_confidence_warning,
                        modules=modules,
                        has_pdf_index=has_pdf_index,
                    )

                    # Update engine memory
                    from llama_index.core.base.llms.types import (
                        ChatMessage,
                        MessageRole,
                    )

                    user_message = ChatMessage(content=prompt, role=MessageRole.USER)
                    assistant_message = ChatMessage(
                        content=full_response, role=MessageRole.ASSISTANT
                    )
                    engine._memory.put(user_message)
                    engine._memory.put(assistant_message)

                    message_data = {
                        "role": "assistant",
                        "content": full_response,
                        "sources": source_data,
                        "time_taken": elapsed,
                        "low_confidence": low_confidence_warning,
                    }
                    if thinking:
                        message_data["thinking"] = thinking
                    if code_blocks:
                        message_data["code_blocks"] = [
                            block.to_dict() for block in code_blocks
                        ]
                    if execution_results:
                        message_data["execution_results"] = [
                            result.to_dict() for result in execution_results
                        ]

                    session["messages"].append(message_data)

                    save_sessions(st.session_state.sessions_file)

                    # Update title after successful response
                    if should_update_title:
                        from tensortruth.app_utils.session import update_title

                        with st.spinner("Generating title..."):
                            update_title(
                                current_id,
                                prompt,
                                params.get("model"),
                                st.session_state.sessions_file,
                            )

                    st.rerun()

                except Exception as e:
                    st.error(f"Engine Error: {e}")

            elif not modules:
                # NO RAG MODE
                start_time = time.time()
                try:
                    # Check if simple_llm needs to be reloaded due to param changes
                    from tensortruth.rag_engine import get_llm

                    # Compute a simple hash of relevant params for simple LLM
                    simple_llm_config = (
                        params.get("model"),
                        params.get("temperature"),
                        params.get("llm_device"),
                        params.get("max_tokens"),
                    )

                    # Reload if config changed or LLM doesn't exist
                    if (
                        "simple_llm" not in st.session_state
                        or st.session_state.get("simple_llm_config")
                        != simple_llm_config
                    ):
                        st.session_state.simple_llm = get_llm(params)
                        st.session_state.simple_llm_config = simple_llm_config

                    llm = st.session_state.simple_llm

                    chat_history = build_chat_history(session["messages"])

                    # Stream response
                    full_response, error, thinking, code_blocks = (
                        stream_simple_llm_response(llm, chat_history)
                    )

                    if error:
                        raise error

                    elapsed = time.time() - start_time

                    # Code Execution (if enabled and blocks found)
                    execution_results = []
                    if code_blocks and st.session_state.get(
                        "code_execution_enabled", True
                    ):
                        from tensortruth.code_execution import ExecutionOrchestrator

                        try:
                            orchestrator = ExecutionOrchestrator()
                            execution_results = orchestrator.execute_blocks(
                                session_id=current_id,
                                code_blocks=code_blocks,
                                timeout=st.session_state.get("code_exec_timeout", 30),
                                enabled=True,
                            )
                        except Exception as exec_error:
                            st.warning(f"Code execution error: {exec_error}")

                    st.caption(f"⏱️ {elapsed:.2f}s | 🔴 No RAG")

                    message_data = {
                        "role": "assistant",
                        "content": full_response,
                        "time_taken": elapsed,
                    }
                    if thinking:
                        message_data["thinking"] = thinking
                    if code_blocks:
                        message_data["code_blocks"] = [
                            block.to_dict() for block in code_blocks
                        ]
                    if execution_results:
                        message_data["execution_results"] = [
                            result.to_dict() for result in execution_results
                        ]

                    session["messages"].append(message_data)

                    save_sessions(st.session_state.sessions_file)

                    # Update title after successful response
                    if should_update_title:
                        from tensortruth.app_utils.session import update_title

                        with st.spinner("Generating title..."):
                            update_title(
                                current_id,
                                prompt,
                                params.get("model"),
                                st.session_state.sessions_file,
                            )

                    st.rerun()

                except Exception as e:
                    st.error(f"LLM Error: {e}")
            else:
                st.error("Engine not loaded!")
