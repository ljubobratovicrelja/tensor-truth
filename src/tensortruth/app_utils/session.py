"""Session management for chat sessions."""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from .title_generation import generate_smart_title_async


def load_sessions(sessions_file: str) -> Dict[str, Any]:
    """Load chat sessions from JSON file."""
    if os.path.exists(sessions_file):
        try:
            with open(sessions_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"current_id": None, "sessions": {}}


def save_sessions(sessions_file: str) -> None:
    """Save chat sessions to JSON file."""
    with open(sessions_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_data, f, indent=2)


def create_session(
    modules: Optional[List[str]], params: Dict[str, Any], sessions_file: str
) -> str:
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


async def update_title_async(
    session_id: str,
    text: str,
    model_name: str,
    sessions_file: str,
    chat_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Update session title using smart title generation (async version)."""
    # Accept chat_data as parameter to avoid accessing st.session_state from background thread
    if chat_data is None:
        chat_data = st.session_state.chat_data

    session = chat_data["sessions"][session_id]
    if session.get("title") == "New Session":
        new_title = await generate_smart_title_async(text, model_name, keep_alive=1)
        session["title"] = new_title
        # Write directly to file instead of using save_sessions (which accesses session_state)
        with open(sessions_file, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2)


def update_title(
    session_id: str, text: str, model_name: str, sessions_file: str
) -> None:
    """Update session title using smart title generation (sync wrapper)."""
    asyncio.run(update_title_async(session_id, text, model_name, sessions_file))


def rename_session(new_title: str, sessions_file: str) -> None:
    """Rename the current session."""
    current_id = st.session_state.chat_data.get("current_id")
    if current_id:
        st.session_state.chat_data["sessions"][current_id]["title"] = new_title
        save_sessions(sessions_file)
        st.rerun()


def delete_session(session_id: str, sessions_file: str) -> None:
    """Delete a session and all associated files (PDFs, markdown, indexes)."""
    import shutil

    from .paths import get_session_dir

    # Delete from session data
    if session_id in st.session_state.chat_data["sessions"]:
        del st.session_state.chat_data["sessions"][session_id]

    # Delete session directory (PDFs + markdown + index)
    session_dir = get_session_dir(session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir)

    save_sessions(sessions_file)
