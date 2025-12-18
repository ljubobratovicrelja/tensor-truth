"""Tensor-Truth Streamlit Application - Main Entry Point."""

import asyncio
import os
import threading
import time
from pathlib import Path

import streamlit as st

from tensortruth import (
    convert_chat_to_markdown,
    convert_latex_delimiters,
    get_max_memory_gb,
)
from tensortruth.app_utils import (
    apply_preset,
    create_session,
    delete_preset,
    download_indexes_with_ui,
    free_memory,
    get_available_modules,
    get_config_file_path,
    get_favorites,
    get_indexes_dir,
    get_ollama_models,
    get_presets_file,
    get_random_generating_message,
    get_random_rag_processing_message,
    get_sessions_file,
    get_system_devices,
    get_user_data_dir,
    load_config,
    load_presets,
    load_sessions,
    process_command,
    quick_launch_preset,
    rename_session,
    save_preset,
    save_sessions,
    toggle_favorite,
)
from tensortruth.app_utils.session import update_title_async
from tensortruth.core.ollama import get_ollama_url


# --- PDF HELPER FUNCTIONS ---
def process_pdf_upload(uploaded_file, session_id: str, sessions_file: str):
    """Handle PDF upload: save, convert, index, update session."""
    from datetime import datetime

    from tensortruth.app_utils.paths import get_session_dir
    from tensortruth.pdf_handler import PDFHandler
    from tensortruth.session_index import SessionIndexBuilder

    session = st.session_state.chat_data["sessions"][session_id]

    # Check if this file is already being processed or has been processed
    if "pdf_documents" not in session:
        session["pdf_documents"] = []

    # Check for duplicate by filename
    for doc in session["pdf_documents"]:
        if doc["filename"] == uploaded_file.name:
            # File already exists, skip processing
            return

    # Create PDF handler
    handler = PDFHandler(get_session_dir(session_id))

    # Save PDF and get metadata
    with st.spinner(f"Uploading {uploaded_file.name}..."):
        pdf_metadata = handler.upload_pdf(uploaded_file)

    # Add to session with "processing" status
    session["pdf_documents"].append(
        {
            "id": pdf_metadata["id"],
            "filename": uploaded_file.name,
            "uploaded_at": str(datetime.now()),
            "file_size": pdf_metadata["file_size"],
            "page_count": pdf_metadata.get("page_count", 0),
            "status": "processing",
            "error_message": None,
        }
    )
    save_sessions(sessions_file)

    # Convert to markdown (with progress spinner)
    try:
        with st.spinner(f"Converting {uploaded_file.name} (this may take a while)..."):
            # TODO: investigate
            _ = handler.convert_pdf_to_markdown(pdf_metadata["path"])

        # Build/rebuild index with metadata extraction
        # Load existing metadata cache from session
        metadata_cache = session.get("pdf_metadata_cache", {})
        builder = SessionIndexBuilder(session_id, metadata_cache=metadata_cache)

        with st.spinner("Indexing document..."):
            markdown_files = handler.get_all_markdown_files()
            builder.build_index(markdown_files)

        # Save updated metadata cache back to session
        session["pdf_metadata_cache"] = builder.get_metadata_cache()

        # Update PDF document status with extracted metadata
        for doc in session["pdf_documents"]:
            if doc["id"] == pdf_metadata["id"]:
                doc["status"] = "indexed"

                # Add extracted metadata to PDF document record
                pdf_meta = metadata_cache.get(pdf_metadata["id"], {})
                if pdf_meta:
                    doc["display_name"] = pdf_meta.get("display_name")
                    doc["authors"] = pdf_meta.get("authors")
                    doc["source_url"] = pdf_meta.get("source_url")
                    doc["metadata_extracted_at"] = str(datetime.now())

        session["has_temp_index"] = True
        save_sessions(sessions_file)

        # Invalidate engine cache to reload with new index
        st.session_state.loaded_config = None

        st.success(f"✅ {uploaded_file.name} indexed successfully!")

    except Exception as e:
        # Update status to "error"
        for doc in session["pdf_documents"]:
            if doc["id"] == pdf_metadata["id"]:
                doc["status"] = "error"
                doc["error_message"] = str(e)
        save_sessions(sessions_file)
        st.error(f"Failed to process PDF: {str(e)}")


def delete_pdf_from_session(pdf_id: str, session_id: str, sessions_file: str):
    """Delete PDF and rebuild index."""
    from tensortruth.app_utils.paths import get_session_dir
    from tensortruth.pdf_handler import PDFHandler
    from tensortruth.session_index import SessionIndexBuilder

    session = st.session_state.chat_data["sessions"][session_id]

    # Remove from session data
    pdf_docs = session.get("pdf_documents", [])
    session["pdf_documents"] = [doc for doc in pdf_docs if doc["id"] != pdf_id]

    # Delete PDF and markdown
    handler = PDFHandler(get_session_dir(session_id))
    handler.delete_pdf(pdf_id)

    # Rebuild index if any PDFs remain, otherwise delete index
    # Load metadata cache
    metadata_cache = session.get("pdf_metadata_cache", {})
    builder = SessionIndexBuilder(session_id, metadata_cache=metadata_cache)

    if session["pdf_documents"]:
        try:
            markdown_files = handler.get_all_markdown_files()
            if markdown_files:
                builder.build_index(markdown_files)
                # Update metadata cache
                session["pdf_metadata_cache"] = builder.get_metadata_cache()
            else:
                builder.delete_index()
                session["has_temp_index"] = False
        except Exception as e:
            st.warning(f"Failed to rebuild index: {e}")
            builder.delete_index()
            session["has_temp_index"] = False
    else:
        builder.delete_index()
        session["has_temp_index"] = False

    save_sessions(sessions_file)

    # Invalidate engine cache
    st.session_state.loaded_config = None


# --- CONFIG ---
# Use platform-specific user data directory (~/.tensortruth)
SESSIONS_FILE = get_sessions_file()
PRESETS_FILE = get_presets_file()
USER_DIR = get_user_data_dir()
INDEX_DIR = get_indexes_dir()
GDRIVE_LINK = (
    "https://drive.google.com/file/d/1jILgN1ADgDgUt5EzkUnFMI8xwY2M_XTu/view?usp=sharing"
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
# Load external stylesheet
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

# Path to logo (now inside the package)
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

    # PDF Upload Section (Chat Mode Only)
    if st.session_state.mode == "chat" and st.session_state.chat_data.get("current_id"):
        curr_id = st.session_state.chat_data["current_id"]
        curr_sess = st.session_state.chat_data["sessions"][curr_id]

        st.markdown("**📄 Session Documents**")
        pdf_docs = curr_sess.get("pdf_documents", [])

        if pdf_docs:
            for doc in pdf_docs:
                status_icons = {
                    "uploading": "⏳",
                    "processing": "🔄",
                    "indexed": "✅",
                    "error": "❌",
                }
                status_icon = status_icons.get(doc["status"], "❓")

                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.caption(f"{status_icon} {doc['filename']}")
                    if doc.get("error_message"):
                        st.caption(f"⚠️ {doc['error_message']}", help="Error details")
                with col2:
                    if st.button("🗑️", key=f"del_pdf_{doc['id']}", help="Remove PDF"):
                        delete_pdf_from_session(doc["id"], curr_id, SESSIONS_FILE)
                        st.rerun()
        else:
            st.caption("No documents uploaded")

        # Upload widget
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="pdf_uploader",
            label_visibility="collapsed",
            help="Upload a PDF document to query alongside knowledge bases",
        )

        # Track processed files to avoid reprocessing on rerun
        if "processed_pdfs" not in st.session_state:
            st.session_state.processed_pdfs = set()

        if uploaded_file:
            # Create a unique identifier for this file (name + size)
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"

            # Only process if not already processed in this session state
            if file_id not in st.session_state.processed_pdfs:
                process_pdf_upload(uploaded_file, curr_id, SESSIONS_FILE)
                st.session_state.processed_pdfs.add(file_id)
                st.rerun()

        st.divider()

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
                from tensortruth.app_utils.session import delete_session

                curr_id = st.session_state.chat_data["current_id"]
                delete_session(curr_id, SESSIONS_FILE)
                st.session_state.chat_data["current_id"] = None
                st.session_state.mode = "setup"
                free_memory()
                st.session_state.loaded_config = None
                st.session_state.show_delete_confirm = False
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

# No RAG warning dialog
if st.session_state.get("show_no_rag_warning", False):

    @st.dialog("No Knowledge Base Selected")
    def confirm_no_rag():
        st.warning(
            "You haven't selected any knowledge base modules. "
            "This will run as a **simple LLM chat without RAG** - "
            "the model won't have access to your indexed documents."
        )
        st.write("")
        st.write("Do you want to proceed anyway?")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.session_state.show_no_rag_warning = False
                st.session_state.pending_params = None
                st.rerun()
        with col2:
            if st.button("Proceed", type="primary", use_container_width=True):
                # Create session with empty modules list (no RAG)
                params = st.session_state.pending_params
                create_session([], params, SESSIONS_FILE)
                st.session_state.mode = "chat"
                st.session_state.show_no_rag_warning = False
                st.session_state.pending_params = None
                # Collapse sidebar when entering chat mode
                st.session_state.sidebar_state = "collapsed"
                st.rerun()

    confirm_no_rag()

# ==========================================
# MAIN CONTENT AREA
# ==========================================

if st.session_state.mode == "setup":
    with st.container():
        # 1. Fetch Data
        available_mods_tuples = get_available_modules(INDEX_DIR)
        # Create mappings for display name <-> module name
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

        st.markdown("### Start a New Research Session")

        # --- QUICK LAUNCH FAVORITES ---
        favorites = get_favorites(PRESETS_FILE)
        if favorites:
            st.caption("One-click start with your favorite configurations")

            # Display favorites in cards (3 per row)
            fav_items = list(favorites.items())
            num_cols = 3
            num_rows = (len(fav_items) + num_cols - 1) // num_cols

            for row in range(num_rows):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    item_idx = row * num_cols + col_idx
                    if item_idx < len(fav_items):
                        preset_name, preset_config = fav_items[item_idx]
                        with cols[col_idx]:
                            with st.container(border=True, height=200):
                                st.markdown(f"**{preset_name}**")
                                # Show description if available
                                description = preset_config.get("description", "")
                                if description:
                                    st.caption(description)
                                else:
                                    # Fallback to module count and device
                                    num_modules = len(preset_config.get("modules", []))
                                    device = preset_config.get(
                                        "llm_device", "gpu"
                                    ).upper()
                                    st.caption(f"{num_modules} modules • {device}")

                                if st.button(
                                    "LAUNCH",
                                    key=f"launch_{preset_name}",
                                    type="primary",
                                    use_container_width=True,
                                ):
                                    success, error = quick_launch_preset(
                                        preset_name,
                                        available_mods,
                                        PRESETS_FILE,
                                        SESSIONS_FILE,
                                    )
                                    if success:
                                        st.session_state.mode = "chat"
                                        st.rerun()
                                    else:
                                        st.error(error)

            st.markdown("---")

        # --- ALL PRESETS MANAGER ---
        if presets:
            # Collapse presets when config section is expanded
            expand_presets = not st.session_state.get("expand_config_section", False)
            with st.expander("Presets", expanded=expand_presets):
                for preset_name in presets.keys():
                    is_favorite = presets[preset_name].get("favorite", False)
                    star_icon = "⭐" if is_favorite else "☆"

                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.markdown(f"**{preset_name}**")
                    with col2:
                        if st.button(
                            "Load", key=f"load_{preset_name}", use_container_width=True
                        ):
                            apply_preset(
                                preset_name,
                                available_mods,
                                available_models,
                                system_devices,
                                PRESETS_FILE,
                            )
                            # Expand the config section to show loaded values
                            st.session_state.expand_config_section = True
                            st.rerun()
                    with col3:
                        if st.button(
                            "Delete", key=f"del_{preset_name}", use_container_width=True
                        ):
                            st.session_state.show_preset_delete_confirm = True
                            st.session_state.preset_to_delete = preset_name
                            st.rerun()
                    with col4:
                        if st.button(
                            star_icon,
                            key=f"fav_{preset_name}",
                            use_container_width=True,
                        ):
                            toggle_favorite(preset_name, PRESETS_FILE)
                            st.rerun()

        # --- MANUAL CONFIGURATION ---
        # Expand when a preset is loaded
        expand_config = st.session_state.get("expand_config_section", False)
        with st.expander("Configure New Session", expanded=expand_config):
            with st.form("launch_form"):
                # --- SELECTION AREA ---
                st.subheader("1. Knowledge Base")
                # Get display names for current selection
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

                # Convert back to module names
                selected_mods = [
                    display_to_module[disp] for disp in selected_display_names
                ]
                # Update session state with module names
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
                    # No modules selected - show warning dialog
                    st.session_state.show_no_rag_warning = True
                    st.session_state.pending_params = {
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
                    st.rerun()
                else:
                    # Show immediate feedback before transition
                    with st.spinner("Creating session..."):
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
                        # Collapse sidebar when entering chat mode
                        st.session_state.sidebar_state = "collapsed"
                    st.rerun()

        # --- SAVE PRESET SECTION ---
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
                        "system_prompt": st.session_state.setup_sys_prompt,
                        "rag_device": st.session_state.setup_rag_device,
                        "llm_device": st.session_state.setup_llm_device,
                    }

                    # Add description if provided
                    if new_preset_description:
                        config_to_save["description"] = new_preset_description

                    if mark_as_favorite:
                        # Find the highest favorite_order
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

        # --- CONNECTION SETTINGS ---
        with st.expander("Connection Settings", expanded=False):
            from tensortruth.app_utils.config import update_config

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

                        # Clear cached model list since URL changed
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
        },
    )

    st.title(session.get("title", "Untitled"))

    # Display model name passively under the title
    model_name = params.get("model", "Unknown")
    st.caption(f"🤖 {model_name}")

    st.divider()

    st.empty()

    # Initialize engine loading state if needed
    if "engine_loading" not in st.session_state:
        st.session_state.engine_loading = False
    if "engine_load_error" not in st.session_state:
        st.session_state.engine_load_error = None

    # Determine target configuration (include session index flag)
    target_tuple = tuple(sorted(modules)) if modules else None
    param_items = sorted([(k, v) for k, v in params.items()])
    param_hash = frozenset(param_items)
    has_pdf_index = session.get("has_temp_index", False)
    target_config = (target_tuple, param_hash, has_pdf_index) if target_tuple else None

    current_config = st.session_state.get("loaded_config")
    engine = st.session_state.get("engine")

    # Check if we need to load/reload the engine
    needs_loading = (modules or has_pdf_index) and (current_config != target_config)

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
                # Extract chat history from session messages (not from engine memory)
                # Convert session messages to ChatMessage format for the new engine
                preserved_history = None

                if session["messages"]:
                    try:
                        from llama_index.core.base.llms.types import (
                            ChatMessage,
                            MessageRole,
                        )

                        chat_messages = []
                        # Only include user and assistant messages, skip command messages
                        for msg in session["messages"]:
                            if msg["role"] == "user":
                                chat_messages.append(
                                    ChatMessage(
                                        content=msg["content"], role=MessageRole.USER
                                    )
                                )
                            elif msg["role"] == "assistant":
                                chat_messages.append(
                                    ChatMessage(
                                        content=msg["content"],
                                        role=MessageRole.ASSISTANT,
                                    )
                                )

                        # Preserve only the last 4 messages (2 conversation turns)
                        # This maintains immediate context without causing hallucinations
                        max_messages = 4
                        if len(chat_messages) > max_messages:
                            preserved_history = chat_messages[-max_messages:]
                        else:
                            preserved_history = chat_messages if chat_messages else None

                    except Exception as e:
                        print(f"Error preserving chat history: {e}")
                        preserved_history = None

                if current_config is not None:
                    free_memory()

                # Check for session index
                from tensortruth.app_utils.paths import get_session_index_dir

                session_index_path = None
                if session.get("has_temp_index", False):
                    index_path = get_session_index_dir(current_id)
                    if os.path.exists(str(index_path)):
                        session_index_path = str(index_path)

                # Call the actual engine loading function directly (bypass UI parts)
                from tensortruth import load_engine_for_modules

                loaded_engine = load_engine_for_modules(
                    modules, params, preserved_history, session_index_path
                )
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
        st.info(
            "💬 Simple LLM mode (No RAG) - Use `/load <name>` to attach a knowledge base."
        )
        engine = None

    # Render message history (but skip the last message if we just added it this run)
    messages_to_render = session["messages"]
    if st.session_state.get("skip_last_message_render", False):
        messages_to_render = session["messages"][:-1]
        st.session_state.skip_last_message_render = False

    for msg in messages_to_render:
        avatar = ":material/settings:" if msg["role"] == "command" else None
        with st.chat_message(msg["role"], avatar=avatar):
            # Show low confidence warning BEFORE the message content for visibility
            if msg.get("low_confidence", False) and modules:
                # Get the confidence threshold and best score from the message if available
                confidence_threshold = params.get("confidence_cutoff", 0.0)
                # Try to get the best score from sources, otherwise show generic warning
                if msg.get("sources") and len(msg["sources"]) > 0:
                    best_score = max(
                        (src["score"] for src in msg["sources"]), default=0.0
                    )
                    st.warning(
                        f"⚠️ **Low Confidence Match** - Best similarity score ({best_score:.2f}) "
                        f"is below your threshold ({confidence_threshold:.2f}). "
                        "The answer may not be reliable. Consider lowering the threshold "
                        "or rephrasing your query."
                    )
                else:
                    # No sources at all (edge case)
                    st.warning(
                        "⚠️ **Low Confidence Match** - "
                        "The answer may not be reliable. Consider lowering the threshold "
                        "or rephrasing your query."
                    )

            st.markdown(convert_latex_delimiters(msg["content"]))

            meta_cols = st.columns([3, 1])
            with meta_cols[0]:
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources"):
                        for src in msg["sources"]:
                            # Extract metadata (with fallbacks for old sessions)
                            fname = src.get("file", "Unknown")
                            display_name = src.get("display_name", fname)
                            source_url = src.get("source_url")
                            authors = src.get("authors")
                            doc_type = src.get("doc_type", "unknown")
                            score = src.get("score", 0.0)

                            # Icon based on document type
                            icon_map = {
                                "paper": "📄",
                                "library_doc": "📚",
                                "uploaded_pdf": "📎",
                                "book": "📖",
                            }
                            icon = icon_map.get(doc_type, "📄")

                            # Use display_name as-is (it already includes authors for books)
                            label = display_name

                            # Render with optional link
                            if source_url:
                                st.caption(
                                    f"{icon} [{label}]({source_url}) ({score:.2f})"
                                )
                            else:
                                st.caption(f"{icon} {label} ({score:.2f})")
            with meta_cols[1]:
                if "time_taken" in msg:
                    # Check different response types
                    if msg.get("low_confidence", False):
                        st.caption(f"⏱️ {msg['time_taken']:.2f}s | ⚠️ Low Confidence")
                    elif msg["role"] == "assistant" and (
                        "sources" not in msg or not msg.get("sources")
                    ):
                        st.caption(f"⏱️ {msg['time_taken']:.2f}s | 🔴 No RAG")
                    else:
                        st.caption(f"⏱️ {msg['time_taken']:.2f}s")

    # Get user input
    prompt = st.chat_input("Ask or type /cmd...")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

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
            # Process command (returns immediately with response message)
            available_mods_tuples = get_available_modules(INDEX_DIR)
            available_mods = [mod for mod, _ in available_mods_tuples]
            is_cmd, response, state_modifier = process_command(
                prompt, session, available_mods
            )

            if is_cmd:
                # Add command message to history immediately
                session["messages"].append({"role": "command", "content": response})

                # Display the response immediately (non-blocking)
                with st.chat_message("command", avatar=":material/settings:"):
                    st.markdown(response)

                save_sessions(SESSIONS_FILE)

                # Apply state changes with a spinner (blocking but with feedback)
                if state_modifier is not None:
                    with st.spinner("⚙️ Applying changes..."):
                        state_modifier()

                st.rerun()

        # 2. STANDARD CHAT PROCESSING
        # Add user message to history, save, and display it via message loop
        session["messages"].append({"role": "user", "content": prompt})
        save_sessions(SESSIONS_FILE)

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

        if session["title"] == "New Session":
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

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if engine:
                start_time = time.time()
                try:
                    # Phase 1: RAG Retrieval (blocking with spinner)
                    with st.spinner(get_random_rag_processing_message()):
                        # Call the internal _run_c3 method to get context
                        # This does the expensive RAG retrieval upfront
                        synthesizer, context_source, context_nodes = engine._run_c3(
                            prompt, chat_history=None, streaming=True
                        )

                    # Check confidence threshold for soft warning
                    low_confidence_warning = False
                    confidence_threshold = params.get("confidence_cutoff", 0.0)

                    if (
                        context_nodes
                        and len(context_nodes) > 0
                        and confidence_threshold > 0
                    ):
                        # Get the best (highest) similarity score from nodes
                        best_score = max(
                            (node.score for node in context_nodes if node.score),
                            default=0.0,
                        )

                        if best_score < confidence_threshold:
                            # Import the low confidence prompt
                            from tensortruth.rag_engine import (
                                CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE,
                            )

                            # Override prompt to make LLM aware of low confidence
                            synthesizer._context_prompt_template = (
                                CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE
                            )

                            # Show soft warning but still use the sources
                            st.warning(
                                "⚠️ **Low Confidence Match** - "
                                f"Best similarity score ({best_score:.2f}) "
                                f"is below your threshold ({confidence_threshold:.2f}). "
                                "The answer may not be reliable. Consider lowering the threshold "
                                "or rephrasing your query."
                            )
                            low_confidence_warning = True
                    elif not context_nodes or len(context_nodes) == 0:
                        # No nodes at all - this shouldn't happen with the reranker, but handle it
                        from llama_index.core.schema import NodeWithScore, TextNode

                        from tensortruth.rag_engine import (
                            CUSTOM_CONTEXT_PROMPT_NO_SOURCES,
                            NO_CONTEXT_FALLBACK_CONTEXT,
                        )

                        st.info(
                            "⚠️ **NO SOURCES RETRIEVED** - "
                            "Response based on general knowledge only, "
                            "not your indexed documents."
                        )

                        # Create a synthetic node with warning context
                        warning_node = NodeWithScore(
                            node=TextNode(text=NO_CONTEXT_FALLBACK_CONTEXT),
                            score=0.0,
                        )
                        context_nodes = [warning_node]
                        low_confidence_warning = True

                        # Override the context prompt to include warning acknowledgment instruction
                        synthesizer._context_prompt_template = (
                            CUSTOM_CONTEXT_PROMPT_NO_SOURCES
                        )

                    # Phase 2: LLM Streaming (responsive, token-by-token)
                    # Now generate the streaming response using the pre-retrieved context
                    import queue

                    token_queue = queue.Queue()
                    synthesizer_ready = threading.Event()
                    streaming_done = threading.Event()
                    error_holder = {"error": None}

                    def stream_tokens_in_background():
                        """Background thread that calls synthesize and streams tokens."""
                        try:
                            # Move the blocking synthesize() call into background thread
                            response = synthesizer.synthesize(prompt, context_nodes)

                            # Get the generator - this doesn't trigger Ollama yet
                            token_gen = iter(response.response_gen)

                            # Pull the FIRST token - THIS is what triggers Ollama and blocks
                            # Only signal ready after we have the first token
                            first_token = next(token_gen)
                            token_queue.put(first_token)

                            # Now signal that streaming has actually started
                            synthesizer_ready.set()

                            # Stream remaining tokens to queue
                            for token in token_gen:
                                token_queue.put(token)
                            streaming_done.set()
                        except StopIteration:
                            # Empty response
                            synthesizer_ready.set()
                            streaming_done.set()
                        except Exception as e:
                            error_holder["error"] = e
                            synthesizer_ready.set()  # Signal even on error
                            streaming_done.set()

                    # Start background thread immediately
                    stream_thread = threading.Thread(
                        target=stream_tokens_in_background, daemon=True
                    )
                    stream_thread.start()

                    # Display tokens as they arrive with immediate updates
                    full_response = ""

                    # Create two placeholders: spinner above, response below
                    spinner_placeholder = st.empty()
                    response_placeholder = st.empty()

                    # Show spinner in the top placeholder
                    with spinner_placeholder:
                        with st.spinner(get_random_generating_message()):
                            # Stream remaining tokens with responsive updates
                            while (
                                not streaming_done.is_set() or not token_queue.empty()
                            ):
                                try:
                                    token = token_queue.get(
                                        timeout=0.05
                                    )  # 50ms polling
                                    if token is not None:
                                        full_response += token
                                        response_placeholder.markdown(
                                            convert_latex_delimiters(full_response)
                                        )
                                except queue.Empty:
                                    continue

                    # Clear the spinner placeholder after streaming is done
                    spinner_placeholder.empty()

                    elapsed = time.time() - start_time

                    # Handle source nodes and timing
                    source_data = []
                    meta_cols = st.columns([3, 1])

                    with meta_cols[0]:
                        if context_nodes:
                            with st.expander("📚 Sources"):
                                for node in context_nodes:
                                    score = float(node.score) if node.score else 0.0

                                    # Extract metadata (with fallbacks for old indexes)
                                    fname = node.metadata.get("file_name", "Unknown")
                                    display_name = node.metadata.get(
                                        "display_name", fname
                                    )
                                    source_url = node.metadata.get("source_url")
                                    authors = node.metadata.get("authors")
                                    doc_type = node.metadata.get("doc_type", "unknown")

                                    # Build source data for message storage
                                    source_data.append(
                                        {
                                            "file": fname,  # Keep original for backwards compat
                                            "display_name": display_name,
                                            "source_url": source_url,
                                            "authors": authors,
                                            "doc_type": doc_type,
                                            "score": score,
                                        }
                                    )

                                    # Icon based on document type
                                    icon_map = {
                                        "paper": "📄",
                                        "library_doc": "📚",
                                        "uploaded_pdf": "📎",
                                        "book": "📖",
                                    }
                                    icon = icon_map.get(doc_type, "📄")

                                    # Use display_name as-is (it already includes authors for books)
                                    label = display_name

                                    # Render with optional link
                                    if source_url:
                                        st.caption(
                                            f"{icon} [{label}]({source_url}) ({score:.2f})"
                                        )
                                    else:
                                        st.caption(f"{icon} {label} ({score:.2f})")

                    with meta_cols[1]:
                        if low_confidence_warning:
                            st.caption(f"⏱️ {elapsed:.2f}s | ⚠️ Low Confidence")
                        else:
                            st.caption(f"⏱️ {elapsed:.2f}s")

                    # Manually update memory (since we bypassed stream_chat)
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

                    # Rerun to display the new assistant message
                    st.rerun()

                except Exception as e:
                    st.error(f"Engine Error: {e}")
            elif not modules:
                # NO RAG MODE - Direct Ollama chat
                start_time = time.time()
                try:
                    # Initialize simple LLM if not already loaded for this session
                    if "simple_llm" not in st.session_state:
                        from tensortruth.rag_engine import get_llm

                        st.session_state.simple_llm = get_llm(params)

                    llm = st.session_state.simple_llm

                    # Build chat history for context
                    from llama_index.core.base.llms.types import (
                        ChatMessage,
                        MessageRole,
                    )

                    chat_history = []
                    for msg in session["messages"]:
                        if msg["role"] == "user":
                            chat_history.append(
                                ChatMessage(
                                    content=msg["content"], role=MessageRole.USER
                                )
                            )
                        elif msg["role"] == "assistant":
                            chat_history.append(
                                ChatMessage(
                                    content=msg["content"], role=MessageRole.ASSISTANT
                                )
                            )

                    # Stream response from Ollama
                    import queue

                    token_queue = queue.Queue()
                    streaming_done = threading.Event()
                    error_holder = {"error": None}

                    def stream_simple_llm():
                        """Stream directly from Ollama without RAG."""
                        try:
                            response = llm.stream_chat(chat_history)
                            for token in response:
                                token_queue.put(str(token.delta))
                            streaming_done.set()
                        except Exception as e:
                            error_holder["error"] = e
                            streaming_done.set()

                    stream_thread = threading.Thread(
                        target=stream_simple_llm, daemon=True
                    )
                    stream_thread.start()

                    full_response = ""
                    spinner_placeholder = st.empty()
                    response_placeholder = st.empty()

                    with spinner_placeholder:
                        with st.spinner(get_random_generating_message()):
                            while (
                                not streaming_done.is_set() or not token_queue.empty()
                            ):
                                try:
                                    token = token_queue.get(timeout=0.05)
                                    if token:
                                        full_response += token
                                        response_placeholder.markdown(
                                            convert_latex_delimiters(full_response)
                                        )
                                except queue.Empty:
                                    continue

                    spinner_placeholder.empty()

                    if error_holder["error"]:
                        raise error_holder["error"]

                    elapsed = time.time() - start_time

                    # Show time with no RAG indicator
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
