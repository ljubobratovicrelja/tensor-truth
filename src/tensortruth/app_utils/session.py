"""Session management for chat sessions."""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st

from .title_generation import generate_smart_title_async


def load_sessions(sessions_file: Union[str, Path]) -> Dict[str, Any]:
    """Load chat sessions from JSON file.

    Args:
        sessions_file: Path to sessions file (str or Path)

    Returns:
        Dictionary with current_id and sessions data
    """
    sessions_file = Path(sessions_file)
    if sessions_file.exists():
        try:
            with open(sessions_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Backward compatibility: Add title_needs_update flag to existing sessions
            migrated = False
            for session in data.get("sessions", {}).values():
                if "title_needs_update" not in session:
                    # If title is still "New Session", flag it for update
                    session["title_needs_update"] = (
                        session.get("title") == "New Session"
                    )
                    migrated = True

            # Save migrated data back to file
            if migrated:
                with open(sessions_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            return data
        except Exception:
            pass
    return {"current_id": None, "sessions": {}}


def save_sessions(sessions_file: Union[str, Path]) -> None:
    """Save chat sessions to JSON file.

    Args:
        sessions_file: Path to sessions file (str or Path)
    """
    sessions_file = Path(sessions_file)
    with open(sessions_file, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_data, f, indent=2)


def create_session(
    modules: Optional[List[str]],
    params: Dict[str, Any],
    sessions_file: Union[str, Path],
) -> str:
    """Create a new chat session."""
    new_id = str(uuid.uuid4())
    st.session_state.chat_data["sessions"][new_id] = {
        "title": "New Session",
        "created_at": str(datetime.now()),
        "messages": [],
        "modules": modules,
        "params": params,
        "title_needs_update": True,  # Flag to trigger auto-title generation
    }
    st.session_state.chat_data["current_id"] = new_id
    save_sessions(sessions_file)
    return new_id


async def update_title_async(
    session_id: str,
    text: str,
    model_name: str,
    sessions_file: Union[str, Path],
    chat_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Update session title using smart title generation (async version).

    Args:
        session_id: Session ID to update
        text: User input text for title generation
        model_name: Model name for generation
        sessions_file: Path to sessions file (str or Path)
        chat_data: Optional chat data (uses session_state if not provided)
    """
    sessions_file = Path(sessions_file)

    # Accept chat_data as parameter to avoid accessing st.session_state
    # from background thread
    if chat_data is None:
        chat_data = st.session_state.chat_data

    session = chat_data["sessions"].get(session_id)
    if not session:
        return

    # Check the flag instead of comparing string values
    if session.get("title_needs_update", False):
        new_title = await generate_smart_title_async(text, model_name, keep_alive=1)
        session["title"] = new_title
        session["title_needs_update"] = False  # Clear the flag
        # Write directly to file instead of using save_sessions
        # (which accesses session_state)
        with open(sessions_file, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2)


def update_title(
    session_id: str, text: str, model_name: str, sessions_file: Union[str, Path]
) -> None:
    """Update session title using smart title generation (sync wrapper).

    Args:
        session_id: Session ID to update
        text: User input text for title generation
        model_name: Model name for generation
        sessions_file: Path to sessions file (str or Path)
    """
    asyncio.run(update_title_async(session_id, text, model_name, sessions_file))


def rename_session(new_title: str, sessions_file: Union[str, Path]) -> None:
    """Rename the current session.

    Args:
        new_title: New title for the session
        sessions_file: Path to sessions file (str or Path)
    """
    current_id = st.session_state.chat_data.get("current_id")
    if current_id:
        st.session_state.chat_data["sessions"][current_id]["title"] = new_title
        save_sessions(sessions_file)
        st.rerun()


def delete_session(session_id: str, sessions_file: Union[str, Path]) -> None:
    """Delete a session and all associated files (PDFs, markdown, indexes)."""
    import shutil

    from .paths import get_session_dir

    # Cleanup Docker container for code execution
    try:
        from tensortruth.code_execution import ExecutionOrchestrator

        orchestrator = ExecutionOrchestrator()
        orchestrator.cleanup_session(session_id)
    except Exception:
        pass  # Continue deletion even if container cleanup fails

    # Delete from session data
    if session_id in st.session_state.chat_data["sessions"]:
        del st.session_state.chat_data["sessions"][session_id]

    # Delete session directory (PDFs + markdown + index)
    session_dir = get_session_dir(session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir)

    save_sessions(sessions_file)
