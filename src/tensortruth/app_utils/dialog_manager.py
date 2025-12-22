"""Dialog state management for the application."""

import streamlit as st


def init_dialog_state():
    """Initialize dialog state flags in session_state if not present."""
    if "show_delete_confirm" not in st.session_state:
        st.session_state.show_delete_confirm = False
    if "show_preset_delete_confirm" not in st.session_state:
        st.session_state.show_preset_delete_confirm = False
    if "show_no_rag_warning" not in st.session_state:
        st.session_state.show_no_rag_warning = False


def show_delete_dialog():
    """Check if delete session dialog should be shown."""
    return st.session_state.get("show_delete_confirm", False)


def show_preset_delete_dialog():
    """Check if delete preset dialog should be shown."""
    return st.session_state.get("show_preset_delete_confirm", False)


def show_no_rag_dialog():
    """Check if no-RAG warning dialog should be shown."""
    return st.session_state.get("show_no_rag_warning", False)


def open_delete_dialog():
    """Open the delete session confirmation dialog."""
    st.session_state.show_delete_confirm = True


def open_preset_delete_dialog():
    """Open the delete preset confirmation dialog."""
    st.session_state.show_preset_delete_confirm = True


def open_no_rag_dialog():
    """Open the no-RAG warning dialog."""
    st.session_state.show_no_rag_warning = True


def close_delete_dialog():
    """Close the delete session confirmation dialog."""
    st.session_state.show_delete_confirm = False


def close_preset_delete_dialog():
    """Close the delete preset confirmation dialog."""
    st.session_state.show_preset_delete_confirm = False


def close_no_rag_dialog():
    """Close the no-RAG warning dialog."""
    st.session_state.show_no_rag_warning = False
