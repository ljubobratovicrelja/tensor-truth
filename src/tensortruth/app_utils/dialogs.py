"""Streamlit dialog components."""

import streamlit as st

from tensortruth.app_utils import delete_preset, free_memory
from tensortruth.app_utils.session import create_session, delete_session


def show_delete_session_dialog(sessions_file: str):
    """Show dialog to confirm session deletion.

    Args:
        sessions_file: Path to sessions JSON file
    """

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
            if st.button("Cancel", width="stretch"):
                st.session_state.show_delete_confirm = False
                st.rerun()
        with col2:
            if st.button("Delete", type="primary", width="stretch"):
                curr_id = st.session_state.chat_data["current_id"]
                delete_session(curr_id, sessions_file)
                st.session_state.chat_data["current_id"] = None
                st.session_state.mode = "setup"
                free_memory()
                st.session_state.loaded_config = None
                st.session_state.show_delete_confirm = False
                st.session_state.session_parameters_initialized = False
                st.rerun()

    confirm_delete()


def show_delete_preset_dialog(presets_file: str):
    """Show dialog to confirm preset deletion.

    Args:
        presets_file: Path to presets JSON file
    """

    @st.dialog("Delete Preset?")
    def confirm_preset_delete():
        preset_name = st.session_state.get("preset_to_delete", "")
        st.write("Are you sure you want to delete this preset?")
        st.write(f"**{preset_name}**")
        st.caption("This action cannot be undone.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", width="stretch"):
                st.session_state.show_preset_delete_confirm = False
                st.session_state.preset_to_delete = None
                st.rerun()
        with col2:
            if st.button("Delete", type="primary", width="stretch"):
                delete_preset(preset_name, presets_file)
                st.session_state.show_preset_delete_confirm = False
                st.session_state.preset_to_delete = None
                st.rerun()

    confirm_preset_delete()


def show_no_rag_warning_dialog(sessions_file: str):
    """Show warning dialog when no knowledge base modules are selected.

    Args:
        sessions_file: Path to sessions JSON file
    """

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
            if st.button("Cancel", width="stretch"):
                st.session_state.show_no_rag_warning = False
                st.session_state.pending_params = None
                st.rerun()
        with col2:
            if st.button("Proceed", type="primary", width="stretch"):
                # Create session with empty modules list (no RAG)
                params = st.session_state.pending_params
                create_session([], params, sessions_file)
                st.session_state.mode = "chat"
                st.session_state.show_no_rag_warning = False
                st.session_state.pending_params = None
                st.session_state.sidebar_state = "collapsed"
                st.rerun()

    confirm_no_rag()
