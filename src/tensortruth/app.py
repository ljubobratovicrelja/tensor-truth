"""Tensor-Truth Streamlit Application - Main Entry Point."""

import os
import sys
import time

import streamlit as st

sys.path.append(os.path.abspath("./src"))

from tensortruth import convert_chat_to_markdown, get_max_memory_gb, run_ingestion
from tensortruth.app_utils import (
    apply_preset,
    create_session,
    delete_preset,
    download_indexes_with_ui,
    ensure_engine_loaded,
    free_memory,
    get_available_modules,
    get_ollama_models,
    get_system_devices,
    load_presets,
    load_sessions,
    process_command,
    rename_session,
    render_vram_gauge,
    save_preset,
    save_sessions,
    update_title,
)

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
        st.write(
            f"**{st.session_state.chat_data['sessions'][st.session_state.chat_data['current_id']]['title']}**"
        )
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

    tab_launch, tab_ingest = st.tabs(["🚀 Launch Session", "📥 Library Ingestion"])

    with tab_launch:
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
                "Click 'Refresh Estimate' to update resource calculation based on current form selections."
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

    with tab_ingest:
        st.subheader("Add Papers to Library")
        arxiv_input = st.text_input("ArXiv ID (e.g., 2310.06825):")
        if st.button("Fetch & Index"):
            if not arxiv_input:
                st.warning("Please enter an ID.")
            else:
                progress = st.empty()
                progress.info("⏳ Starting ingestion pipeline...")
                success, logs = run_ingestion("papers", arxiv_input)
                for log in logs:
                    st.text(log)
                if success:
                    st.success("Ingestion Complete!")
                    time.sleep(2)
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

    # Engine loading with visual feedback
    if modules:
        engine = ensure_engine_loaded(modules, params)
    else:
        st.warning("No linked knowledge base. Use `/load <name>` to attach one.")
        engine = None

    st.title(session.get("title", "Untitled"))
    st.caption(
        "💡 Tip: Type **/help** to see all commands. Use `/device` to manage hardware."
    )

    for msg in session["messages"]:
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

    if prompt := st.chat_input("Ask or type /cmd..."):
        # 1. COMMAND PROCESSING
        if prompt.startswith("/"):
            available_mods = get_available_modules(INDEX_DIR)
            is_cmd, response = process_command(
                prompt, session, available_mods, SESSIONS_FILE
            )
            if is_cmd:
                session["messages"].append({"role": "assistant", "content": response})
                save_sessions(SESSIONS_FILE)
                st.rerun()

        # 2. STANDARD CHAT PROCESSING
        update_title(current_id, prompt, params.get("model"), SESSIONS_FILE)

        with st.chat_message("user"):
            st.markdown(prompt)
        session["messages"].append({"role": "user", "content": prompt})
        save_sessions(SESSIONS_FILE)

        with st.chat_message("assistant"):
            if engine:
                start_time = time.time()
                try:
                    # Show RAG pipeline status.
                    status_container = st.empty()

                    # Use container to show multiple status steps
                    with status_container.container():
                        step1 = st.empty()
                        step2 = st.empty()
                        step3 = st.empty()
                        step4 = st.empty()

                        step1.markdown("**Embedding query...**")
                        time.sleep(0.2)

                        step2.markdown("**Searching knowledge base...**")

                        # Start the streaming response (RAG happens here)
                        streaming_response = engine.stream_chat(prompt)

                        step3.markdown("**Ranking results...**")
                        time.sleep(0.15)

                    # Clear all status messages
                    status_container.empty()

                    # Create a placeholder for the response that we can replace
                    response_container = st.empty()

                    # Stream the response progressively into the placeholder
                    with response_container.container():

                        def response_generator():
                            for token in streaming_response.response_gen:
                                yield token

                        # Display streaming response
                        answer = st.write_stream(response_generator())

                    elapsed = time.time() - start_time

                    # Handle source nodes
                    source_data = []
                    if (
                        hasattr(streaming_response, "source_nodes")
                        and streaming_response.source_nodes
                    ):
                        with st.expander("📚 Sources"):
                            for node in streaming_response.source_nodes:
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
                st.error("Engine not loaded. Use `/load <index>` to start.")
        st.rerun()
