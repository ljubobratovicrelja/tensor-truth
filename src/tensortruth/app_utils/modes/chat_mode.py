"""Chat mode - Main conversation interface with RAG."""

import os
import threading

import streamlit as st

from tensortruth.app_utils.chat_handler import handle_chat_response
from tensortruth.app_utils.chat_utils import preserve_chat_history
from tensortruth.app_utils.config import compute_config_hash
from tensortruth.app_utils.helpers import free_memory, get_available_modules
from tensortruth.app_utils.paths import get_session_index_dir
from tensortruth.app_utils.rendering import render_chat_message
from tensortruth.app_utils.session import save_sessions
from tensortruth.app_utils.setup_state import get_session_params_with_defaults

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
            "💬 Simple LLM mode (No RAG) - Use `/load <name>` to attach a knowledge base."
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
            "💡 Tip: Type **/help** to see all commands. Use `/device` to manage hardware."
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
                # Traditional command - display as command message (not in LLM history)
                session["messages"].append({"role": "command", "content": response})

                with st.chat_message("command", avatar=":material/settings:"):
                    st.markdown(response)

                save_sessions(st.session_state.sessions_file)

                if state_modifier is not None:
                    with st.spinner("⚙️ Applying changes..."):
                        state_modifier()

                st.rerun()
            elif response:
                # Command returned is_cmd=False but provided a response
                # (e.g., websearch) - treat as assistant message for LLM history

                # Update title if this is the first message
                if session.get("title_needs_update", False):
                    with st.spinner("Generating title..."):
                        from tensortruth.app_utils.session import update_title

                        update_title(
                            current_id,
                            prompt,
                            params.get("model"),
                            st.session_state.sessions_file,
                        )

                session["messages"].append({"role": "user", "content": prompt})
                session["messages"].append({"role": "assistant", "content": response})
                save_sessions(st.session_state.sessions_file)

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    st.markdown(response)

                st.rerun()

        # STANDARD CHAT PROCESSING
        session["messages"].append({"role": "user", "content": prompt})
        save_sessions(st.session_state.sessions_file)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Unified handler for both RAG and simple LLM modes
            handle_chat_response(
                prompt=prompt,
                session=session,
                params=params,
                current_id=current_id,
                sessions_file=st.session_state.sessions_file,
                modules=modules,
                has_pdf_index=has_pdf_index,
                engine=engine,
            )
