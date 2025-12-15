"""Tensor-Truth Streamlit Application - Main Entry Point."""

import asyncio
import threading
import time

import streamlit as st

from tensortruth import convert_chat_to_markdown, get_max_memory_gb
from tensortruth.app_utils import (
    apply_preset,
    create_session,
    delete_preset,
    download_indexes_with_ui,
    free_memory,
    get_available_modules,
    get_ollama_models,
    get_random_rag_processing_message,
    get_system_devices,
    load_presets,
    load_sessions,
    process_command,
    rename_session,
    render_vram_gauge,
    save_preset,
    save_sessions,
)
from tensortruth.app_utils.session import update_title_async

# --- CONFIG ---
SESSIONS_FILE = "chat_sessions.json"
PRESETS_FILE = "presets.json"
INDEX_DIR = "./indexes"
GDRIVE_LINK = (
    "https://drive.google.com/file/d/1jILgN1ADgDgUt5EzkUnFMI8xwY2M_XTu/view?usp=sharing"
)
MAX_VRAM_GB = get_max_memory_gb()


st.set_page_config(page_title="Tensor-Truth", layout="wide", page_icon="⚡")

# --- CSS ---
st.markdown(
    """
<style>
    .stButton button { text-align: left; padding-left: 10px; width: 100%; }
    .stChatMessage { padding: 1rem; border-radius: 10px; }
    div[data-testid="stExpander"] { border: none; box-shadow: none; }
    code { color: #d63384; }
</style>
""",
    unsafe_allow_html=True,
)

# --- INITIALIZATION ---
# Download indexes from Google Drive if directory is empty or missing
download_indexes_with_ui(INDEX_DIR, GDRIVE_LINK)

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
    st.header("⚡ Tensor-Truth")

    if st.button("➕ Start New Chat", type="primary", use_container_width=True):
        st.session_state.mode = "setup"
        st.session_state.chat_data["current_id"] = None
        st.rerun()

    st.divider()
    st.subheader("🗂️ History")

    session_ids = list(st.session_state.chat_data["sessions"].keys())
    for sess_id in reversed(session_ids):
        sess = st.session_state.chat_data["sessions"][sess_id]
        title = sess.get("title", "Untitled")
        current_id = st.session_state.chat_data.get("current_id")
        is_active = sess_id == current_id

        label = f" {title} "
        if st.button(label, key=sess_id, use_container_width=True):
            st.session_state.chat_data["current_id"] = sess_id
            st.session_state.mode = "chat"
            st.rerun()

    st.divider()

    if st.session_state.mode == "chat" and st.session_state.chat_data.get("current_id"):
        curr_id = st.session_state.chat_data["current_id"]
        curr_sess = st.session_state.chat_data["sessions"][curr_id]

        with st.expander("⚙️ Session Settings", expanded=True):
            new_name = st.text_input("Rename:", value=curr_sess.get("title"))
            if st.button("Update Title"):
                rename_session(new_name, SESSIONS_FILE)

            st.caption("Active Indices:")
            mods = curr_sess.get("modules", [])
            if not mods:
                st.caption("*None*")
            for m in mods:
                st.code(m, language="text")

            md_data = convert_chat_to_markdown(curr_sess)
            st.download_button(
                "📥 Export", md_data, f"{curr_sess['title'][:20]}.md", "text/markdown"
            )

            st.markdown("---")
            if st.button("🗑️ Delete Chat"):
                st.session_state.show_delete_confirm = True
                st.rerun()

# Delete confirmation dialog
if st.session_state.get("show_delete_confirm", False):

    @st.dialog("Delete Chat Session?")
    def confirm_delete():
        st.write("Are you sure you want to delete this chat session?")
        session_title = st.session_state.chat_data["sessions"][
            st.session_state.chat_data["current_id"]
        ]["title"]
        st.write(f"**{session_title}**")
        st.caption("This action cannot be undone.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_delete_confirm = False
                st.rerun()
        with col2:
            if st.button("Delete", type="primary", use_container_width=True):
                curr_id = st.session_state.chat_data["current_id"]
                del st.session_state.chat_data["sessions"][curr_id]
                st.session_state.chat_data["current_id"] = None
                st.session_state.mode = "setup"
                free_memory()
                st.session_state.loaded_config = None
                st.session_state.show_delete_confirm = False
                save_sessions(SESSIONS_FILE)
                st.rerun()

    confirm_delete()

# Preset delete confirmation dialog
if st.session_state.get("show_preset_delete_confirm", False):

    @st.dialog("Delete Preset?")
    def confirm_preset_delete():
        preset_name = st.session_state.get("preset_to_delete", "")
        st.write("Are you sure you want to delete this preset?")
        st.write(f"**{preset_name}**")
        st.caption("This action cannot be undone.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_preset_delete_confirm = False
                st.session_state.preset_to_delete = None
                st.rerun()
        with col2:
            if st.button("Delete", type="primary", use_container_width=True):
                delete_preset(preset_name, PRESETS_FILE)
                st.session_state.show_preset_delete_confirm = False
                st.session_state.preset_to_delete = None
                st.rerun()

    confirm_preset_delete()

# ==========================================
# MAIN CONTENT AREA
# ==========================================

if st.session_state.mode == "setup":
    st.title("Control Center")

    with st.container():
        # 1. Fetch Data
        available_mods = get_available_modules(INDEX_DIR)
        available_models = get_ollama_models()
        system_devices = get_system_devices()
        presets = load_presets(PRESETS_FILE)

        default_model_idx = 0
        for i, m in enumerate(available_models):
            if "deepseek-r1:8b" in m:
                default_model_idx = i

        # 2. Initialize Widget State if New
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

            # Smart device defaults: prefer MPS on Apple Silicon, otherwise CPU/GPU split
            if "mps" in system_devices:
                # Apple Silicon - use MPS for both RAG and LLM
                st.session_state.setup_rag_device = "mps"
                st.session_state.setup_llm_device = (
                    "gpu"  # Ollama will use MPS when available
                )
            else:
                # Desktop/CUDA - keep original defaults
                st.session_state.setup_rag_device = "cpu"
                st.session_state.setup_llm_device = "gpu"

            st.session_state.setup_init = True

        st.markdown("### 🚀 Start a New Research Session")
        st.caption("Configure your knowledge base and model parameters.")

        # --- PRESETS SECTION ---
        if presets:
            with st.expander("📁 Saved Configurations (Presets)", expanded=True):
                col_p1, col_p2, col_p3 = st.columns([3, 1, 1])
                with col_p1:
                    selected_preset = st.selectbox(
                        "Select Preset:",
                        list(presets.keys()),
                        label_visibility="collapsed",
                    )
                with col_p2:
                    if st.button("📂 Load", use_container_width=True):
                        apply_preset(
                            selected_preset,
                            available_mods,
                            available_models,
                            system_devices,
                            PRESETS_FILE,
                        )
                        st.rerun()
                with col_p3:
                    if st.button("🗑️ Delete", type="primary", use_container_width=True):
                        st.session_state.show_preset_delete_confirm = True
                        st.session_state.preset_to_delete = selected_preset
                        st.rerun()

        # --- FORM WRAPPER FOR STABILITY ---
        with st.form("launch_form"):
            # --- SELECTION AREA ---
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("1. Knowledge Base")
                selected_mods = st.multiselect(
                    "Active Indices:", available_mods, key="setup_mods"
                )

            with col_b:
                st.subheader("2. Model Selection")
                if available_models:
                    selected_model = st.selectbox(
                        "LLM:", available_models, key="setup_model"
                    )
                else:
                    st.error("No models found in Ollama.")
                    selected_model = "None"

            st.subheader("3. RAG Parameters")
            p1, p2, p3 = st.columns(3)
            with p1:
                reranker_model = st.selectbox(
                    "Reranker",
                    options=[
                        "BAAI/bge-reranker-v2-m3",
                        "BAAI/bge-reranker-base",
                        "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    ],
                    key="setup_reranker",
                )
            with p2:
                ctx = st.select_slider(
                    "Context Window",
                    options=[2048, 4096, 8192, 16384, 32768],
                    key="setup_ctx",
                )
            with p3:
                temp = st.slider("Temperature", 0.0, 1.0, step=0.1, key="setup_temp")

            with st.expander("Advanced Settings"):
                top_n = st.number_input(
                    "Top N (Final Context)",
                    min_value=1,
                    max_value=20,
                    key="setup_top_n",
                )
                conf = st.slider(
                    "Confidence Cutoff", 0.0, 1.0, step=0.05, key="setup_conf"
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
                        help="Run Retrieval on specific hardware. CPU saves VRAM but is slower.",
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

            # VRAM GAUGE (Inside Form = Updates on Submit)
            st.caption(
                "Click 'Refresh Estimate' to update resource calculation "
                "based on current form selections."
            )

            vram_est = render_vram_gauge(
                st.session_state.setup_model,
                len(st.session_state.setup_mods),
                st.session_state.setup_ctx,
                st.session_state.setup_rag_device,
                st.session_state.setup_llm_device,
                MAX_VRAM_GB,
            )

            c_btn1, c_btn2 = st.columns([1, 1])
            with c_btn1:
                submitted_check = st.form_submit_button(
                    "🔄 Refresh Estimate", use_container_width=True
                )
            with c_btn2:
                submitted_start = st.form_submit_button(
                    "🚀 Start Session", type="primary", use_container_width=True
                )

            if submitted_check:
                # Just triggers rerun to update gauge
                pass

            if submitted_start:
                if not selected_mods:
                    st.error("Please select at least one index.")
                elif vram_est > (MAX_VRAM_GB + 4.0):
                    st.error(
                        f"Config is extremely heavy ({vram_est:.1f}GB). Reduce parameters."
                    )
                else:
                    # Show immediate feedback before transition
                    with st.spinner("🚀 Creating session..."):
                        params = {
                            "model": selected_model,
                            "temperature": temp,
                            "context_window": ctx,
                            "system_prompt": sys_prompt,
                            "reranker_model": reranker_model,
                            "reranker_top_n": top_n,
                            "confidence_cutoff": conf,
                            "rag_device": rag_device,
                            "llm_device": llm_device,
                        }
                        create_session(selected_mods, params, SESSIONS_FILE)
                        st.session_state.mode = "chat"
                    st.rerun()

        # --- SAVE PRESET SECTION (Outside form to allow name typing without submit) ---
        with st.expander("💾 Save Configuration as Preset"):
            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                new_preset_name = st.text_input(
                    "Preset Name", placeholder="e.g. 'Deep Search 32B'"
                )
            with col_s2:
                st.write("")  # Spacer
                st.write("")
                if st.button("Save", use_container_width=True):
                    if new_preset_name:
                        config_to_save = {
                            "modules": st.session_state.setup_mods,
                            "model": st.session_state.setup_model,
                            "reranker_model": st.session_state.setup_reranker,
                            "context_window": st.session_state.setup_ctx,
                            "temperature": st.session_state.setup_temp,
                            "reranker_top_n": st.session_state.setup_top_n,
                            "confidence_cutoff": st.session_state.setup_conf,
                            "system_prompt": st.session_state.setup_sys_prompt,
                            "rag_device": st.session_state.setup_rag_device,
                            "llm_device": st.session_state.setup_llm_device,
                        }
                        save_preset(new_preset_name, config_to_save, PRESETS_FILE)
                        st.success(f"Saved: {new_preset_name}")
                        time.sleep(1)
                        st.rerun()

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
        },
    )

    st.title(session.get("title", "Untitled"))

    # Initialize engine loading state if needed
    if "engine_loading" not in st.session_state:
        st.session_state.engine_loading = False
    if "engine_load_error" not in st.session_state:
        st.session_state.engine_load_error = None

    # Determine target configuration
    target_tuple = tuple(sorted(modules)) if modules else None
    param_items = sorted([(k, v) for k, v in params.items()])
    param_hash = frozenset(param_items)
    target_config = (target_tuple, param_hash) if target_tuple else None

    current_config = st.session_state.get("loaded_config")
    engine = st.session_state.get("engine")

    # Check if we need to load/reload the engine
    needs_loading = modules and (current_config != target_config)

    # Background engine loading with threading
    if needs_loading and not st.session_state.engine_loading:
        # Start loading in background thread
        st.session_state.engine_loading = True
        st.session_state.engine_load_error = None

        # Create threading primitives (shared between main and background threads)
        if "engine_load_event" not in st.session_state:
            st.session_state.engine_load_event = threading.Event()
        if "engine_load_result" not in st.session_state:
            st.session_state.engine_load_result = {"engine": None, "error": None}

        # Capture references for thread closure
        load_event = st.session_state.engine_load_event
        load_result = st.session_state.engine_load_result
        load_event.clear()

        def load_engine_background():
            try:
                if current_config is not None:
                    free_memory()
                # Call the actual engine loading function directly (bypass UI parts)
                from tensortruth import load_engine_for_modules

                loaded_engine = load_engine_for_modules(modules, params)
                # Store in shared dict (not session_state directly)
                load_result["engine"] = loaded_engine
                load_result["config"] = target_config
            except Exception as e:
                load_result["error"] = str(e)
            finally:
                load_event.set()  # Signal completion

        # Start background thread
        thread = threading.Thread(target=load_engine_background, daemon=True)
        thread.start()

    # Handle engine load errors or missing modules (don't show loading status - let user type)
    if st.session_state.engine_load_error:
        st.error(f"Failed to load engine: {st.session_state.engine_load_error}")
        engine = None
    elif not modules:
        st.warning("No linked knowledge base. Use `/load <name>` to attach one.")
        engine = None

    # Render message history (but skip the last message if we just added it this run)
    messages_to_render = session["messages"]
    if st.session_state.get("skip_last_message_render", False):
        messages_to_render = session["messages"][:-1]
        st.session_state.skip_last_message_render = False

    for msg in messages_to_render:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta_cols = st.columns([3, 1])
            with meta_cols[0]:
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources"):
                        for src in msg["sources"]:
                            st.caption(f"{src['file']} ({src['score']:.2f})")
            with meta_cols[1]:
                if "time_taken" in msg:
                    st.caption(f"⏱️ {msg['time_taken']:.2f}s")

    # Get user input
    prompt = st.chat_input("Ask or type /cmd...")

    # Show tip if no messages exist AND no prompt being processed
    if not session["messages"] and not prompt:
        st.caption(
            "💡 Tip: Type **/help** to see all commands. Use `/device` to manage hardware."
        )

    if prompt:
        # If engine is still loading, wait for it to complete
        if st.session_state.engine_loading:
            with st.spinner("⏳ Waiting for model to finish loading..."):
                # Wait for the threading event with timeout (60 seconds)
                if "engine_load_event" in st.session_state:
                    event_triggered = st.session_state.engine_load_event.wait(
                        timeout=60.0
                    )

                    if not event_triggered:
                        st.error("Model loading timed out after 60 seconds")
                        st.session_state.engine_loading = False
                    else:
                        # Transfer results from shared dict to session_state
                        load_result = st.session_state.engine_load_result
                        if load_result.get("error"):
                            st.session_state.engine_load_error = load_result["error"]
                        elif load_result.get("engine"):
                            st.session_state.engine = load_result["engine"]
                            st.session_state.loaded_config = load_result["config"]
                        st.session_state.engine_loading = False

        # Check if background loading completed before prompt (transfer results)
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

        # Always refresh engine reference from session state (may have been loaded in background)
        engine = st.session_state.get("engine")

        # 1. COMMAND PROCESSING
        if prompt.startswith("/"):
            # Show immediate feedback for command execution
            with st.spinner(f"⚙️ Processing command: {prompt}"):
                available_mods = get_available_modules(INDEX_DIR)
                is_cmd, response = process_command(
                    prompt, session, available_mods, SESSIONS_FILE
                )
            if is_cmd:
                session["messages"].append({"role": "assistant", "content": response})
                save_sessions(SESSIONS_FILE)
                st.rerun()

        # 2. STANDARD CHAT PROCESSING
        # Add user message to history, save, and display it via message loop
        session["messages"].append({"role": "user", "content": prompt})
        save_sessions(SESSIONS_FILE)

        print(f"PROMPT: {prompt}")

        # Render the user message inline (before message loop picks it up on next rerun)
        with st.chat_message("user"):
            st.markdown(prompt)

        st.empty()  # Spacer - forces next message to be below user input

        with st.chat_message("assistant"):
            if engine:
                start_time = time.time()
                try:
                    # Show RAG pipeline status with actual spinner
                    with st.spinner(get_random_rag_processing_message()):
                        response = engine.chat(prompt)

                    # Display response
                    answer = str(response)
                    st.markdown(answer)

                    elapsed = time.time() - start_time

                    # Handle source nodes
                    source_data = []
                    if hasattr(response, "source_nodes") and response.source_nodes:
                        with st.expander("📚 Sources"):
                            for node in response.source_nodes:
                                score = float(node.score) if node.score else 0.0
                                fname = node.metadata.get("file_name", "Unknown")
                                st.caption(f"{fname} ({score:.2f})")
                                source_data.append({"file": fname, "score": score})

                    st.caption(f"⏱️ {elapsed:.2f}s")
                    session["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_data,
                            "time_taken": elapsed,
                        }
                    )
                    save_sessions(SESSIONS_FILE)
                except Exception as e:
                    st.error(f"Engine Error: {e}")
            else:
                st.error("Engine not loaded!")

        # Update title in background (can be slow with LLM) - fire and forget
        def run_async_in_thread(coro):
            """Run async coroutine in a new thread with its own event loop (non-blocking)."""

            def run():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            # Don't wait for title generation - it can complete in background

        # Capture chat_data reference for background thread
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
