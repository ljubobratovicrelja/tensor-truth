"""Tensor-Truth Streamlit Application - Main Entry Point."""

import asyncio
import os
import threading
import time
from pathlib import Path

import streamlit as st

from tensortruth import convert_chat_to_markdown, get_max_memory_gb
from tensortruth.app_utils import (
    create_session,
    download_indexes_with_ui,
    free_memory,
    get_available_modules,
    get_config_file_path,
    get_favorites,
    get_indexes_dir,
    get_ollama_models,
    get_presets_file,
    get_random_rag_processing_message,
    get_sessions_file,
    get_system_devices,
    get_user_data_dir,
    load_config,
    load_presets,
    load_sessions,
    process_command,
    rename_session,
    save_preset,
    save_sessions,
)
from tensortruth.app_utils.chat_utils import preserve_chat_history
from tensortruth.app_utils.config import compute_config_hash, update_config
from tensortruth.app_utils.dialogs import (
    show_delete_preset_dialog,
    show_delete_session_dialog,
    show_no_rag_warning_dialog,
)
from tensortruth.app_utils.paths import get_session_index_dir
from tensortruth.app_utils.pdf_ui import render_pdf_documents_section
from tensortruth.app_utils.presets_ui import (
    render_favorite_preset_cards,
    render_presets_manager,
)
from tensortruth.app_utils.rendering import (
    extract_source_metadata,
    render_chat_message,
    render_low_confidence_warning,
    render_message_footer,
)
from tensortruth.app_utils.session import update_title_async
from tensortruth.app_utils.streaming import (
    stream_rag_response,
    stream_simple_llm_response,
)
from tensortruth.core.ollama import get_ollama_url

# --- CONFIG ---
SESSIONS_FILE = get_sessions_file()
PRESETS_FILE = get_presets_file()
USER_DIR = get_user_data_dir()
INDEX_DIR = get_indexes_dir()
GDRIVE_LINK = (
    "https://drive.google.com/file/d/12wZsBwrywl9nXOCLr50lpWB2SiFdu1XB/view?usp=sharing"
)
MAX_VRAM_GB = get_max_memory_gb()

ICON_PATH = Path(__file__).parent / "media" / "tensor_truth_icon_256.png"
st.set_page_config(
    page_title="Tensor-Truth",
    layout="wide",
    page_icon=str(ICON_PATH),
    initial_sidebar_state="auto",
)

# --- CSS ---
CSS_PATH = Path(__file__).parent / "media" / "app_styles.css"
with open(CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- INITIALIZATION ---
# Initialize config file with smart defaults if it doesn't exist
if os.path.exists(get_config_file_path()) is False:
    _ = load_config()

# Download indexes from Google Drive if directory is empty or missing
if os.path.exists(INDEX_DIR) is False or not os.listdir(INDEX_DIR):
    download_indexes_with_ui(USER_DIR, GDRIVE_LINK)

# Path to logo
LOGO_PATH = Path(__file__).parent / "media" / "tensor_truth_banner.png"

if "chat_data" not in st.session_state:
    st.session_state.chat_data = load_sessions(SESSIONS_FILE)
if "mode" not in st.session_state:
    st.session_state.mode = "setup"
if "loaded_config" not in st.session_state:
    st.session_state.loaded_config = None
if "engine" not in st.session_state:
    st.session_state.engine = None

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.image(str(LOGO_PATH), width=500)

    if st.button("Start New Chat", type="primary", use_container_width=True):
        st.session_state.mode = "setup"
        st.session_state.chat_data["current_id"] = None
        st.session_state.expand_config_section = False
        st.rerun()

    st.divider()
    st.empty()

    # Session list
    session_ids = list(st.session_state.chat_data["sessions"].keys())
    for sess_id in reversed(session_ids):
        sess = st.session_state.chat_data["sessions"][sess_id]
        title = sess.get("title", "Untitled")
        current_id = st.session_state.chat_data.get("current_id")

        label = f" {title} "
        if st.button(label, key=sess_id, use_container_width=True):
            st.session_state.chat_data["current_id"] = sess_id
            st.session_state.mode = "chat"
            st.rerun()

    st.divider()

    # PDF Upload Section (Chat Mode Only)
    if st.session_state.mode == "chat" and st.session_state.chat_data.get("current_id"):
        curr_id = st.session_state.chat_data["current_id"]
        render_pdf_documents_section(curr_id, SESSIONS_FILE)
        st.divider()

    # Session Settings (Chat Mode Only)
    if st.session_state.mode == "chat" and st.session_state.chat_data.get("current_id"):
        curr_id = st.session_state.chat_data["current_id"]
        curr_sess = st.session_state.chat_data["sessions"][curr_id]

        with st.expander("Session Settings", expanded=True):
            new_name = st.text_input("Rename:", value=curr_sess.get("title"))

            if st.button("Update", use_container_width=True):
                rename_session(new_name, SESSIONS_FILE)

            st.caption("Active Indices:")
            mods = curr_sess.get("modules", [])
            if not mods:
                st.caption("*None*")
            else:
                # Get display names for active modules
                available_mods_tuples = get_available_modules(INDEX_DIR)
                module_to_display = {mod: disp for mod, disp in available_mods_tuples}

                for m in mods:
                    display_name = module_to_display.get(m, m)
                    st.caption(f"• {display_name}")

            md_data = convert_chat_to_markdown(curr_sess)
            st.download_button(
                "Export",
                md_data,
                f"{curr_sess['title'][:20]}.md",
                "text/markdown",
                use_container_width=True,
            )

            if st.button("Delete Chat", use_container_width=True):
                st.session_state.show_delete_confirm = True
                st.rerun()

# Dialog handlers
if st.session_state.get("show_delete_confirm", False):
    show_delete_session_dialog(SESSIONS_FILE)

if st.session_state.get("show_preset_delete_confirm", False):
    show_delete_preset_dialog(PRESETS_FILE)

if st.session_state.get("show_no_rag_warning", False):
    show_no_rag_warning_dialog(SESSIONS_FILE)

# ==========================================
# MAIN CONTENT AREA
# ==========================================

if st.session_state.mode == "setup":
    with st.container():
        # Fetch data
        available_mods_tuples = get_available_modules(INDEX_DIR)
        module_to_display = {mod: disp for mod, disp in available_mods_tuples}
        display_to_module = {disp: mod for mod, disp in available_mods_tuples}
        available_mods = [mod for mod, _ in available_mods_tuples]

        available_models = get_ollama_models()
        system_devices = get_system_devices()
        presets = load_presets(PRESETS_FILE)

        default_model_idx = 0
        for i, m in enumerate(available_models):
            if "deepseek-r1:8b" in m:
                default_model_idx = i

        # Initialize widget state if new
        if "setup_init" not in st.session_state:
            try:
                cpu_index = system_devices.index("cpu")
            except ValueError:
                cpu_index = 0

            st.session_state.setup_mods = []
            st.session_state.setup_model = (
                available_models[default_model_idx] if available_models else None
            )
            st.session_state.setup_reranker = "BAAI/bge-reranker-v2-m3"
            st.session_state.setup_ctx = 4096
            st.session_state.setup_temp = 0.3
            st.session_state.setup_top_n = 3
            st.session_state.setup_conf = 0.3
            st.session_state.setup_sys_prompt = ""

            # Smart device defaults
            if "mps" in system_devices:
                st.session_state.setup_rag_device = "mps"
                st.session_state.setup_llm_device = "gpu"
            else:
                st.session_state.setup_rag_device = "cpu"
                st.session_state.setup_llm_device = "gpu"

            st.session_state.setup_init = True

        st.markdown("### Start a New Research Session")

        # Quick launch favorites
        favorites = get_favorites(PRESETS_FILE)
        if favorites:
            render_favorite_preset_cards(
                favorites, available_mods, PRESETS_FILE, SESSIONS_FILE
            )

        # All presets manager
        if presets:
            render_presets_manager(
                presets, available_mods, available_models, system_devices, PRESETS_FILE
            )

        # Manual configuration
        expand_config = st.session_state.get("expand_config_section", False)
        with st.expander("Configure New Session", expanded=expand_config):
            with st.form("launch_form"):
                st.subheader("1. Knowledge Base")
                available_display_names = [
                    module_to_display[mod] for mod in available_mods
                ]
                current_display_selection = [
                    module_to_display[mod]
                    for mod in st.session_state.get("setup_mods", [])
                    if mod in module_to_display
                ]

                selected_display_names = st.multiselect(
                    "Active Indices:",
                    available_display_names,
                    default=current_display_selection,
                )

                selected_mods = [
                    display_to_module[disp] for disp in selected_display_names
                ]
                st.session_state.setup_mods = selected_mods

                st.subheader("2. Model Selection")

                model_col, context_win_col, temperature_col = st.columns(3)

                with model_col:
                    if available_models:
                        selected_model = st.selectbox(
                            "LLM:", available_models, key="setup_model"
                        )
                    else:
                        st.error("No models found in Ollama.")
                        selected_model = "None"

                with context_win_col:
                    ctx = st.select_slider(
                        "Context Window",
                        options=[2048, 4096, 8192, 16384, 32768],
                        key="setup_ctx",
                    )

                with temperature_col:
                    temp = st.slider(
                        "Temperature", 0.0, 1.0, step=0.1, key="setup_temp"
                    )

                st.subheader("3. RAG Parameters")

                rerank_col, top_n_col, conf_col = st.columns(3)
                with rerank_col:
                    reranker_model = st.selectbox(
                        "Reranker",
                        options=[
                            "BAAI/bge-reranker-v2-m3",
                            "BAAI/bge-reranker-base",
                            "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        ],
                        key="setup_reranker",
                    )
                with top_n_col:
                    top_n = st.number_input(
                        "Top N (Final Context)",
                        min_value=1,
                        max_value=20,
                        key="setup_top_n",
                    )

                with conf_col:
                    conf = st.slider(
                        "Confidence Warning Threshold",
                        0.0,
                        1.0,
                        step=0.05,
                        key="setup_conf",
                        help=(
                            "Show a warning if the best similarity score is below "
                            "this threshold (soft hint, doesn't filter results)"
                        ),
                    )

                conf_cutoff_hard = st.slider(
                    "Confidence Cutoff (Hard Filter)",
                    0.0,
                    1.0,
                    step=0.05,
                    key="setup_conf_cutoff_hard",
                    help=(
                        "Hard cutoff - all sources with reranked scores below "
                        "this threshold will be filtered out after reranking. "
                        "Should be lower than the warning threshold."
                    ),
                )

                sys_prompt = st.text_area(
                    "System Instructions:",
                    height=68,
                    placeholder="Optional...",
                    key="setup_sys_prompt",
                )

                st.markdown("#### Hardware Allocation")
                h1, h2 = st.columns(2)

                with h1:
                    rag_device = st.selectbox(
                        "Pipeline Device (Embed/Rerank)",
                        options=system_devices,
                        help=(
                            "Run Retrieval on specific hardware. "
                            "CPU saves VRAM but is slower."
                        ),
                        key="setup_rag_device",
                    )
                with h2:
                    llm_device = st.selectbox(
                        "Model Device (Ollama)",
                        options=["gpu", "cpu"],
                        help="Force Ollama to run on CPU to save VRAM for other tasks.",
                        key="setup_llm_device",
                    )

                st.markdown("---")

                submitted_start = st.form_submit_button(
                    "Start Session", type="primary", use_container_width=True
                )

            if submitted_start:
                if not selected_mods:
                    st.session_state.show_no_rag_warning = True
                    st.session_state.pending_params = {
                        "model": selected_model,
                        "temperature": temp,
                        "context_window": ctx,
                        "system_prompt": sys_prompt,
                        "reranker_model": reranker_model,
                        "reranker_top_n": top_n,
                        "confidence_cutoff": conf,
                        "confidence_cutoff_hard": conf_cutoff_hard,
                        "rag_device": rag_device,
                        "llm_device": llm_device,
                    }
                    st.rerun()
                else:
                    with st.spinner("Creating session..."):
                        params = {
                            "model": selected_model,
                            "temperature": temp,
                            "context_window": ctx,
                            "system_prompt": sys_prompt,
                            "reranker_model": reranker_model,
                            "reranker_top_n": top_n,
                            "confidence_cutoff": conf,
                            "confidence_cutoff_hard": conf_cutoff_hard,
                            "rag_device": rag_device,
                            "llm_device": llm_device,
                        }
                        create_session(selected_mods, params, SESSIONS_FILE)
                        st.session_state.mode = "chat"
                        st.session_state.sidebar_state = "collapsed"
                    st.rerun()

        # Save preset section
        with st.expander("Save Current Configuration as Preset", expanded=False):
            new_preset_name = st.text_input(
                "Preset Name", placeholder="e.g. 'Deep Search 32B'"
            )
            new_preset_description = st.text_input(
                "Description (optional)",
                placeholder="Brief description of this preset...",
            )
            mark_as_favorite = st.checkbox("Mark as Favorite", value=False)

            if st.button("Save Preset", use_container_width=True, type="primary"):
                if new_preset_name:
                    config_to_save = {
                        "modules": st.session_state.setup_mods,
                        "model": st.session_state.setup_model,
                        "reranker_model": st.session_state.setup_reranker,
                        "context_window": st.session_state.setup_ctx,
                        "temperature": st.session_state.setup_temp,
                        "reranker_top_n": st.session_state.setup_top_n,
                        "confidence_cutoff": st.session_state.setup_conf,
                        "confidence_cutoff_hard": st.session_state.setup_conf_cutoff_hard,
                        "system_prompt": st.session_state.setup_sys_prompt,
                        "rag_device": st.session_state.setup_rag_device,
                        "llm_device": st.session_state.setup_llm_device,
                    }

                    if new_preset_description:
                        config_to_save["description"] = new_preset_description

                    if mark_as_favorite:
                        all_presets = load_presets(PRESETS_FILE)
                        max_order = -1
                        for preset in all_presets.values():
                            if preset.get("favorite", False):
                                order = preset.get("favorite_order", 0)
                                if order > max_order:
                                    max_order = order
                        config_to_save["favorite"] = True
                        config_to_save["favorite_order"] = max_order + 1

                    save_preset(new_preset_name, config_to_save, PRESETS_FILE)
                    st.success(f"Saved: {new_preset_name}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("Please enter a preset name")

        # Connection settings
        with st.expander("Connection Settings", expanded=False):
            config = load_config()
            current_url = get_ollama_url()

            new_url = st.text_input(
                "Ollama Base URL",
                value=current_url,
                help="e.g. http://localhost:11434 or http://192.168.1.50:11434",
            )

            if st.button("Save Connection URL"):
                if new_url != current_url:
                    try:
                        update_config(ollama_base_url=new_url)
                        get_ollama_models.clear()
                        st.success(
                            "Configuration saved! Model list will refresh from new URL."
                        )
                    except Exception as e:
                        st.error(f"Failed to save config: {e}")
                else:
                    st.info("No changes made.")


elif st.session_state.mode == "chat":
    current_id = st.session_state.chat_data.get("current_id")
    if not current_id:
        st.session_state.mode = "setup"
        st.rerun()

    session = st.session_state.chat_data["sessions"][current_id]
    modules = session.get("modules", [])
    params = session.get(
        "params",
        {
            "model": "deepseek-r1:8b",
            "temperature": 0.3,
            "context_window": 4096,
            "confidence_cutoff": 0.2,
            "confidence_cutoff_hard": 0.0,
        },
    )

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
    target_config = compute_config_hash(modules, params, has_pdf_index)
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
    elif not modules:
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
        render_chat_message(msg, params, modules)

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
            available_mods_tuples = get_available_modules(INDEX_DIR)
            available_mods = [mod for mod, _ in available_mods_tuples]
            is_cmd, response, state_modifier = process_command(
                prompt, session, available_mods
            )

            if is_cmd:
                session["messages"].append({"role": "command", "content": response})

                with st.chat_message("command", avatar=":material/settings:"):
                    st.markdown(response)

                save_sessions(SESSIONS_FILE)

                if state_modifier is not None:
                    with st.spinner("⚙️ Applying changes..."):
                        state_modifier()

                st.rerun()

        # STANDARD CHAT PROCESSING
        session["messages"].append({"role": "user", "content": prompt})
        save_sessions(SESSIONS_FILE)

        # Update title in background
        def run_async_in_thread(coro):
            """Run async coroutine in a new thread with its own event loop."""

            def run():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

        if session["title"] == "New Session":
            chat_data_snapshot = st.session_state.chat_data

            async def update_title_task():
                await update_title_async(
                    current_id,
                    prompt,
                    params.get("model"),
                    SESSIONS_FILE,
                    chat_data=chat_data_snapshot,
                )

            run_async_in_thread(update_title_task())

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

                        synthesizer._context_prompt_template = (
                            CUSTOM_CONTEXT_PROMPT_NO_SOURCES
                        )

                    # Phase 2: LLM Streaming
                    full_response, error = stream_rag_response(
                        synthesizer, prompt, context_nodes
                    )

                    if error:
                        raise error

                    elapsed = time.time() - start_time

                    # Extract source metadata
                    source_data = []
                    for node in context_nodes:
                        metadata = extract_source_metadata(node, is_node=True)
                        source_data.append(metadata)

                    # Render footer
                    render_message_footer(
                        sources_or_nodes=context_nodes,
                        is_nodes=True,
                        time_taken=elapsed,
                        low_confidence=low_confidence_warning,
                        modules=modules,
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

                    session["messages"].append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "sources": source_data,
                            "time_taken": elapsed,
                            "low_confidence": low_confidence_warning,
                        }
                    )

                    save_sessions(SESSIONS_FILE)
                    st.rerun()

                except Exception as e:
                    st.error(f"Engine Error: {e}")

            elif not modules:
                # NO RAG MODE
                start_time = time.time()
                try:
                    if "simple_llm" not in st.session_state:
                        from tensortruth.rag_engine import get_llm

                        st.session_state.simple_llm = get_llm(params)

                    llm = st.session_state.simple_llm

                    from tensortruth.app_utils.chat_utils import build_chat_history

                    chat_history = build_chat_history(session["messages"])

                    # Stream response
                    full_response, error = stream_simple_llm_response(llm, chat_history)

                    if error:
                        raise error

                    elapsed = time.time() - start_time

                    st.caption(f"⏱️ {elapsed:.2f}s | 🔴 No RAG")

                    session["messages"].append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "time_taken": elapsed,
                        }
                    )

                    save_sessions(SESSIONS_FILE)
                    st.rerun()

                except Exception as e:
                    st.error(f"LLM Error: {e}")
            else:
                st.error("Engine not loaded!")
