"""Chat mode - Main conversation interface with RAG."""

import os
import threading

import streamlit as st

from tensortruth.app_utils.chat_handler import handle_chat_response
from tensortruth.app_utils.chat_utils import preserve_chat_history
from tensortruth.app_utils.config import compute_config_hash
from tensortruth.app_utils.helpers import free_memory, get_available_modules
from tensortruth.app_utils.intent_classifier import (
    detect_and_classify,
    has_agent_triggers,
)
from tensortruth.app_utils.paths import get_session_index_dir
from tensortruth.app_utils.rendering import render_chat_message
from tensortruth.app_utils.session import save_sessions
from tensortruth.app_utils.setup_state import get_session_params_with_defaults

from ..commands import process_command


def _should_route_to_agent(prompt: str, session: dict, params: dict) -> bool:
    """Check if message should be routed to an agent via natural language.

    Args:
        prompt: User's input message
        session: Current session data
        params: Session parameters

    Returns:
        True if message should be classified for potential agent routing
    """
    # Check if natural language agents are enabled
    try:
        config = st.session_state.config
        if not config.agent.enable_natural_language_agents:
            return False
    except (AttributeError, KeyError):
        # Config not available, use default (enabled)
        pass

    # Quick trigger word check (no LLM call)
    return has_agent_triggers(prompt)


def _handle_natural_language_agent(
    prompt: str,
    session: dict,
    params: dict,
    current_id: str,
) -> None:
    """Handle natural language agent routing.

    Classifies the user's intent and routes to appropriate agent.

    Args:
        prompt: User's input message
        session: Current session data
        params: Session parameters
        current_id: Current session ID
    """
    from tensortruth.core.ollama import get_ollama_url

    # Get config
    try:
        config = st.session_state.config
        classifier_model = config.agent.intent_classifier_model
    except (AttributeError, KeyError):
        classifier_model = "llama3.2:3b"

    ollama_url = get_ollama_url()

    # Classify intent
    with st.spinner("🤔 Understanding your request..."):
        intent_result = detect_and_classify(
            message=prompt,
            recent_messages=session.get("messages", [])[-4:],
            ollama_url=ollama_url,
            classifier_model=classifier_model,
        )

    # Route based on intent
    if intent_result.intent == "browse":
        # Route to browse agent
        query = intent_result.query or prompt
        _execute_browse_agent(query, session, params, current_id)

    elif intent_result.intent == "search":
        # Route to web search
        query = intent_result.query or prompt
        _execute_web_search(query, session, params, current_id)

    else:
        # Intent is "chat" - let it fall through to standard processing
        # This shouldn't happen since we checked triggers, but handle gracefully
        # by adding message to history and letting standard handler process it
        session["messages"].append({"role": "user", "content": prompt})
        save_sessions(st.session_state.sessions_file)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Note: We return without rerun, so standard chat will NOT run
        # because we're in an elif block. We need to handle chat here.
        with st.chat_message("assistant"):
            engine = st.session_state.get("engine")
            modules = session.get("modules", [])
            has_pdf_index = session.get("has_temp_index", False)
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


def _execute_browse_agent(
    query: str,
    session: dict,
    params: dict,
    current_id: str,
) -> None:
    """Execute the browse agent with the given query.

    Args:
        query: Research query extracted from user message
        session: Current session data
        params: Session parameters
        current_id: Current session ID
    """
    from tensortruth import convert_latex_delimiters
    from tensortruth.agents.mcp_agent import browse_agent
    from tensortruth.app_utils.rendering_agent import render_agent_progress
    from tensortruth.app_utils.session import update_title
    from tensortruth.core.ollama import get_ollama_url

    # Get models - check session params first, then config, then defaults
    main_model = params.get("model")

    # Check for per-session reasoning model (from presets)
    reasoning_model = params.get("agent_reasoning_model")
    if reasoning_model is None:
        try:
            config = st.session_state.config
            reasoning_model = config.agent.reasoning_model
        except (AttributeError, KeyError):
            from tensortruth.core.constants import DEFAULT_AGENT_REASONING_MODEL

            reasoning_model = DEFAULT_AGENT_REASONING_MODEL

    ollama_url = get_ollama_url()
    context_window = min(params.get("context_window", 16384), 8192)

    # Update title if needed
    if session.get("title_needs_update", False):
        with st.spinner("Generating title..."):
            update_title(
                current_id,
                f"Research: {query}",
                params.get("model"),
                st.session_state.sessions_file,
            )

    # Show what we're doing
    st.info(f"🔍 **Researching:** {query}")

    # Progress tracking
    progress_placeholder = st.empty()
    progress_updates = []

    def update_progress(message: str):
        progress_updates.append(message)
        render_agent_progress(
            "\n\n".join(progress_updates), placeholder=progress_placeholder
        )

    # Streaming placeholder
    response_placeholder = st.empty()
    accumulated_response = {"text": ""}

    def stream_token(token: str):
        accumulated_response["text"] += token
        response_placeholder.markdown(
            convert_latex_delimiters(accumulated_response["text"])
        )

    # Execute agent
    result = browse_agent(
        goal=query,
        model_name=reasoning_model,
        synthesis_model=main_model,
        ollama_url=ollama_url,
        max_iterations=params.get("agent_max_iterations", 10),
        progress_callback=update_progress,
        stream_callback=stream_token,
        context_window=context_window,
    )

    response_placeholder.empty()

    # Add to history
    # Use original prompt as user message
    session["messages"].append({"role": "user", "content": f"Research: {query}"})

    response = result.final_answer or "No results found."
    session["messages"].append({"role": "assistant", "content": response})
    save_sessions(st.session_state.sessions_file)

    # Display
    with st.chat_message("user"):
        st.markdown(f"Research: {query}")

    with st.chat_message("assistant"):
        st.markdown(response)


def _execute_web_search(
    query: str,
    session: dict,
    params: dict,
    current_id: str,
) -> None:
    """Execute web search with the given query.

    Args:
        query: Search query extracted from user message
        session: Current session data
        params: Session parameters
        current_id: Current session ID
    """
    from tensortruth.app_utils.rendering import render_web_search_progress
    from tensortruth.app_utils.session import update_title
    from tensortruth.utils.web_search import web_search

    # Update title if needed
    if session.get("title_needs_update", False):
        with st.spinner("Generating title..."):
            update_title(
                current_id,
                f"Search: {query}",
                params.get("model"),
                st.session_state.sessions_file,
            )

    # Show what we're doing
    st.info(f"🔎 **Searching:** {query}")

    # Progress tracking
    progress_placeholder = st.empty()
    progress_updates = []

    def update_progress(message: str):
        progress_updates.append(message)
        render_web_search_progress(
            "\n\n".join(progress_updates), placeholder=progress_placeholder
        )

    # Execute search
    response = web_search(
        query=query,
        llm_model=params.get("model"),
        top_k=5,
        progress_callback=update_progress,
        max_concurrent=5,
    )

    # Add to history
    session["messages"].append({"role": "user", "content": f"Search: {query}"})
    session["messages"].append({"role": "assistant", "content": response})
    save_sessions(st.session_state.sessions_file)

    # Display
    with st.chat_message("user"):
        st.markdown(f"Search: {query}")

    with st.chat_message("assistant"):
        st.markdown(response)


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
                            # Mark models as loaded for browser refresh detection
                            from tensortruth.app_utils.app_state import (
                                mark_models_loaded,
                            )

                            mark_models_loaded()
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
                # Mark models as loaded for browser refresh detection
                from tensortruth.app_utils.app_state import mark_models_loaded

                mark_models_loaded()
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
                # (e.g., websearch, browse agent) - treat as assistant message for LLM history

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

                # Check if agent thinking metadata is available
                agent_thinking = None
                if hasattr(st.session_state, "last_agent_thinking"):
                    agent_thinking = st.session_state.last_agent_thinking
                    # Clear it after use
                    delattr(st.session_state, "last_agent_thinking")

                # Build assistant message with optional agent thinking
                assistant_msg = {"role": "assistant", "content": response}
                if agent_thinking:
                    assistant_msg["agent_thinking"] = agent_thinking

                session["messages"].append(assistant_msg)
                save_sessions(st.session_state.sessions_file)

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    # Render agent thinking if present
                    if agent_thinking:
                        from tensortruth.app_utils.rendering_agent import (
                            render_agent_thinking,
                        )

                        render_agent_thinking(agent_thinking)

                    st.markdown(response)

                st.rerun()

        # NATURAL LANGUAGE AGENT ROUTING
        # Check if message contains trigger words and NL agents are enabled
        elif _should_route_to_agent(prompt, session, params):
            _handle_natural_language_agent(prompt, session, params, current_id)
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
