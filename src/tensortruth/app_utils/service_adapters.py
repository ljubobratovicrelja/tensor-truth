"""Streamlit adapters for service layer.

This module provides bridge functions between the pure service layer and
Streamlit's session_state. Services return new state objects; adapters
sync that state back to session_state.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from tensortruth.services import (
    ConfigService,
    IntentService,
    PDFService,
    RAGService,
    SessionData,
    SessionService,
)


def get_session_service() -> SessionService:
    """Get or create the session service singleton.

    Returns:
        SessionService instance stored in session_state.
    """
    if "session_service" not in st.session_state:
        from tensortruth.app_utils.paths import get_sessions_file

        st.session_state.session_service = SessionService(get_sessions_file())
    return st.session_state.session_service


def get_config_service() -> ConfigService:
    """Get or create the config service singleton.

    Returns:
        ConfigService instance stored in session_state.
    """
    if "config_service" not in st.session_state:
        st.session_state.config_service = ConfigService()
    return st.session_state.config_service


def get_intent_service() -> IntentService:
    """Get or create the intent service singleton.

    Returns:
        IntentService instance stored in session_state.
    """
    if "intent_service" not in st.session_state:
        config_service = get_config_service()
        config = config_service.load()
        st.session_state.intent_service = IntentService(
            ollama_url=config.ollama.base_url,
            classifier_model=config.agent.intent_classifier_model,
        )
    return st.session_state.intent_service


def get_rag_service() -> RAGService:
    """Get or create the RAG service singleton.

    Returns:
        RAGService instance stored in session_state.
    """
    if "rag_service" not in st.session_state:
        config_service = get_config_service()
        st.session_state.rag_service = RAGService(config=config_service.load())
    return st.session_state.rag_service


def get_pdf_service(session_id: str) -> PDFService:
    """Get or create a PDF service for the specified session.

    Args:
        session_id: Session identifier.

    Returns:
        PDFService instance for the session.
    """
    cache_key = f"pdf_service_{session_id}"
    if cache_key not in st.session_state:
        from tensortruth.app_utils.paths import get_session_dir

        # Get metadata cache from session if available
        metadata_cache = {}
        if "chat_data" in st.session_state:
            session = st.session_state.chat_data.get("sessions", {}).get(session_id, {})
            metadata_cache = session.get("pdf_metadata_cache", {})

        st.session_state[cache_key] = PDFService(
            session_id=session_id,
            session_dir=get_session_dir(session_id),
            metadata_cache=metadata_cache,
        )
    return st.session_state[cache_key]


def sync_session_state(data: SessionData) -> None:
    """Sync SessionData back to Streamlit session_state.

    Args:
        data: SessionData to sync.
    """
    st.session_state.chat_data = data.to_dict()


def get_session_data() -> SessionData:
    """Get current session data from Streamlit session_state.

    Returns:
        SessionData from session_state, or loaded from file if not present.
    """
    if "chat_data" not in st.session_state:
        service = get_session_service()
        data = service.load()
        st.session_state.chat_data = data.to_dict()
        return data

    return SessionData.from_dict(st.session_state.chat_data)


# Adapter functions that use services internally while maintaining
# backward-compatible function signatures


def create_session_via_service(
    modules: Optional[list],
    params: Dict[str, Any],
    sessions_file: Path,
) -> str:
    """Create a new session using the service layer.

    This is a drop-in replacement for the old create_session function.

    Args:
        modules: List of module names.
        params: Session parameters.
        sessions_file: Path to sessions file (for compatibility).

    Returns:
        New session ID.
    """
    service = get_session_service()
    data = get_session_data()

    new_id, new_data = service.create(modules, params, data)
    service.save(new_data)
    sync_session_state(new_data)

    return new_id


def rename_session_via_service(new_title: str, sessions_file: Path) -> None:
    """Rename the current session using the service layer.

    This is a drop-in replacement for the old rename_session function.

    Args:
        new_title: New title for the session.
        sessions_file: Path to sessions file (for compatibility).
    """
    service = get_session_service()
    data = get_session_data()

    if data.current_id:
        new_data = service.update_title(data.current_id, new_title, data)
        service.save(new_data)
        sync_session_state(new_data)
        st.rerun()


def delete_session_via_service(session_id: str, sessions_file: Path) -> None:
    """Delete a session using the service layer.

    This is a drop-in replacement for the old delete_session function.

    Args:
        session_id: Session ID to delete.
        sessions_file: Path to sessions file (for compatibility).
    """
    from tensortruth.app_utils.paths import get_session_dir

    service = get_session_service()
    data = get_session_data()

    session_dir = get_session_dir(session_id)
    new_data = service.delete(session_id, data, session_dir)
    service.save(new_data)
    sync_session_state(new_data)


def add_message_via_service(
    session_id: str,
    message: Dict[str, Any],
) -> None:
    """Add a message to a session using the service layer.

    Args:
        session_id: Session ID.
        message: Message dict with 'role' and 'content'.
    """
    service = get_session_service()
    data = get_session_data()

    new_data = service.add_message(session_id, message, data)
    sync_session_state(new_data)
    # Note: Don't save to disk on every message - save is called separately


def save_sessions_via_service() -> None:
    """Save current session state using the service layer."""
    service = get_session_service()
    data = get_session_data()
    service.save(data)
