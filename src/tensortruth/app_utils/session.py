"""Session management for chat sessions."""

import json
import os
import uuid
from datetime import datetime

import streamlit as st

from .title_generation import generate_smart_title


def load_sessions(sessions_file: str):
    """Load chat sessions from JSON file."""
    if os.path.exists(sessions_file):
        try:
            with open(sessions_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"current_id": None, "sessions": {}}


def save_sessions(sessions_file: str):
    """Save chat sessions to JSON file."""
    with open(sessions_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_data, f, indent=2)


def create_session(modules, params, sessions_file: str):
    """Create a new chat session."""
    new_id = str(uuid.uuid4())
    st.session_state.chat_data["sessions"][new_id] = {
        "title": "New Session",
        "created_at": str(datetime.now()),
        "messages": [],
        "modules": modules,
        "params": params,
    }
    st.session_state.chat_data["current_id"] = new_id
    save_sessions(sessions_file)
    return new_id


def update_title(session_id, text, model_name, sessions_file: str):
    """Update session title using smart title generation."""
    session = st.session_state.chat_data["sessions"][session_id]
    if session.get("title") == "New Session":
        new_title = generate_smart_title(text, model_name)
        session["title"] = new_title
        save_sessions(sessions_file)


def rename_session(new_title, sessions_file: str):
    """Rename the current session."""
    current_id = st.session_state.chat_data.get("current_id")
    if current_id:
        st.session_state.chat_data["sessions"][current_id]["title"] = new_title
        save_sessions(sessions_file)
        st.rerun()
