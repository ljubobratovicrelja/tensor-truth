"""Tensor-Truth Streamlit Application - Main Entry Point.

This is a thin routing layer that initializes the app and delegates
to mode-specific modules (setup_mode, chat_mode) for rendering.
"""

import streamlit as st

from tensortruth import convert_chat_to_markdown
from tensortruth.app_utils.app_state import init_app_state
from tensortruth.app_utils.dialog_manager import (
    init_dialog_state,
    open_delete_dialog,
    show_delete_dialog,
    show_no_rag_dialog,
    show_preset_delete_dialog,
)
from tensortruth.app_utils.dialogs import (
    show_delete_preset_dialog,
    show_delete_session_dialog,
    show_no_rag_warning_dialog,
)
from tensortruth.app_utils.helpers import (
    download_indexes_with_ui,
    get_available_modules,
)
from tensortruth.app_utils.modes import render_chat_mode, render_setup_mode
from tensortruth.app_utils.pdf_ui import render_pdf_documents_section
from tensortruth.app_utils.session import rename_session

# ==========================================
# PAGE CONFIGURATION
# ==========================================

# Must be called before any other st commands
st.set_page_config(
    page_title="Tensor-Truth",
    layout="wide",
    page_icon=str(__file__).replace("app.py", "media/tensor_truth_icon_256.png"),
    initial_sidebar_state="auto",
)

# ==========================================
# INITIALIZATION
# ==========================================

# Initialize app state (paths, constants, session data)
init_app_state()

# Initialize dialog state flags
init_dialog_state()

# Apply CSS styles
st.markdown(st.session_state.css_data, unsafe_allow_html=True)

# Initialize config file with smart defaults if it doesn't exist
# (This is now handled by init_app_state which loads and caches the config)

# Download indexes from Google Drive if directory is empty or missing
index_dir = st.session_state.index_dir
if not index_dir.exists() or not any(index_dir.iterdir()):
    download_indexes_with_ui(st.session_state.user_dir)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.image(str(st.session_state.logo_path), width=500)

    if st.button("Start New Chat", type="primary", use_container_width=True):
        st.session_state.mode = "setup"
        st.session_state.chat_data["current_id"] = None
        st.session_state.expand_config_section = False
        st.session_state.session_parameters_initialized = False
        st.rerun()

    st.divider()
    st.empty()

    # Session list
    session_ids = list(st.session_state.chat_data["sessions"].keys())
    for sess_id in reversed(session_ids):
        sess = st.session_state.chat_data["sessions"][sess_id]
        title = sess.get("title", "Untitled")

        label = f" {title} "
        if st.button(label, key=sess_id, use_container_width=True):
            st.session_state.chat_data["current_id"] = sess_id
            st.session_state.mode = "chat"
            st.rerun()

    st.divider()

    # PDF Upload Section (Chat Mode Only)
    if st.session_state.mode == "chat" and st.session_state.chat_data.get("current_id"):
        curr_id = st.session_state.chat_data["current_id"]
        render_pdf_documents_section(curr_id, st.session_state.sessions_file)
        st.divider()

    # Session Settings (Chat Mode Only)
    if st.session_state.mode == "chat" and st.session_state.chat_data.get("current_id"):
        curr_id = st.session_state.chat_data["current_id"]
        curr_sess = st.session_state.chat_data["sessions"][curr_id]

        with st.expander("Session Settings", expanded=True):
            new_name = st.text_input("Rename:", value=curr_sess.get("title"))

            if st.button("Update", use_container_width=True):
                rename_session(new_name, st.session_state.sessions_file)

            st.caption("Active Indices:")
            mods = curr_sess.get("modules", [])
            pdf_docs = curr_sess.get("pdf_documents", [])
            has_any_indices = bool(mods) or bool(pdf_docs)

            if not has_any_indices:
                st.caption("*None*")
            else:
                # Show permanent knowledge base modules
                if mods:
                    available_mods_tuples = get_available_modules(index_dir)
                    module_to_display = {
                        mod: disp for mod, disp in available_mods_tuples
                    }

                    for m in mods:
                        display_name = module_to_display.get(m, m)
                        st.caption(f"{display_name}")

                # Show session PDF documents
                if pdf_docs:
                    indexed_pdfs = [
                        pdf for pdf in pdf_docs if pdf.get("status") == "indexed"
                    ]
                    for pdf in indexed_pdfs:
                        display_name = pdf.get("display_name", pdf.get("filename"))
                        st.caption(f"📄 {display_name}")

            md_data = convert_chat_to_markdown(curr_sess)
            st.download_button(
                "Export",
                md_data,
                f"{curr_sess['title'][:20]}.md",
                "text/markdown",
                use_container_width=True,
            )

            if st.button("Delete Chat", use_container_width=True):
                open_delete_dialog()
                st.rerun()

# ==========================================
# DIALOG HANDLERS
# ==========================================

if show_delete_dialog():
    show_delete_session_dialog(st.session_state.sessions_file)

if show_preset_delete_dialog():
    show_delete_preset_dialog(st.session_state.presets_file)

if show_no_rag_dialog():
    show_no_rag_warning_dialog(st.session_state.sessions_file)

# ==========================================
# MAIN CONTENT - MODE ROUTING
# ==========================================

if st.session_state.mode == "setup":
    render_setup_mode()
elif st.session_state.mode == "chat":
    render_chat_mode()
