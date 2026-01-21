"""Session management service - pure business logic without Streamlit dependencies.

This service handles all session CRUD operations, returning new state objects
instead of mutating global state. The UI adapter layer is responsible for
syncing the returned state back to Streamlit session_state.
"""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from .models import SessionData

if TYPE_CHECKING:
    from .config_service import ConfigService


class SessionService:
    """Service for managing chat sessions.

    All methods are pure - they accept state as input and return new state
    as output, without accessing Streamlit session_state directly.
    """

    def __init__(self, sessions_file: Union[str, Path]):
        """Initialize session service.

        Args:
            sessions_file: Path to the JSON file storing session data.
        """
        self.sessions_file = Path(sessions_file)

    def _apply_config_defaults(
        self, params: Dict[str, Any], config_service: "ConfigService"
    ) -> Dict[str, Any]:
        """Apply config defaults to incomplete session params.

        Args:
            params: User-provided params (may be empty or partial)
            config_service: ConfigService to load defaults from

        Returns:
            Complete params dict with all defaults filled in
        """
        try:
            config = config_service.load()

            # Define defaults from config
            defaults = {
                "temperature": config.ui.default_temperature,
                "context_window": config.ui.default_context_window,
                "max_tokens": config.ui.default_max_tokens,
                "reranker_model": config.ui.default_reranker,
                "reranker_top_n": config.ui.default_top_n,
                "confidence_cutoff": config.ui.default_confidence_threshold,
                "confidence_cutoff_hard": config.ui.default_confidence_cutoff_hard,
                "rag_device": config.rag.default_device,
                "balance_strategy": config.rag.default_balance_strategy,
                "embedding_model": config.rag.default_embedding_model,
                "llm_device": "gpu",  # Reasonable default
            }

            # User params override defaults
            return {**defaults, **params}

        except Exception as e:
            # If config loading fails, return params as-is
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to apply config defaults: {e}")
            return params

    def load(self) -> SessionData:
        """Load chat sessions from JSON file.

        Returns:
            SessionData with current_id and sessions dict.
            Returns empty data if file doesn't exist or is corrupted.
        """
        if not self.sessions_file.exists():
            return SessionData(current_id=None, sessions={})

        try:
            with open(self.sessions_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Backward compatibility: Add title_needs_update flag to existing sessions
            migrated = False
            for session in raw_data.get("sessions", {}).values():
                if "title_needs_update" not in session:
                    session["title_needs_update"] = (
                        session.get("title") == "New Session"
                    )
                    migrated = True

            # Save migrated data back to file
            if migrated:
                with open(self.sessions_file, "w", encoding="utf-8") as f:
                    json.dump(raw_data, f, indent=2)

            return SessionData.from_dict(raw_data)

        except (json.JSONDecodeError, IOError, KeyError):
            return SessionData(current_id=None, sessions={})

    def save(self, data: SessionData) -> None:
        """Save chat sessions to JSON file.

        Filters out empty sessions (no messages) except for the current session.

        Args:
            data: SessionData to save.
        """
        # Filter out empty sessions (sessions with no messages)
        filtered_sessions = {}
        for session_id, session in data.sessions.items():
            messages = session.get("messages", [])
            # Always keep the current session or sessions with messages
            if session_id == data.current_id or messages:
                filtered_sessions[session_id] = session

        # Create data to save
        save_data = SessionData(current_id=data.current_id, sessions=filtered_sessions)

        # Ensure parent directory exists
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.sessions_file, "w", encoding="utf-8") as f:
            json.dump(save_data.to_dict(), f, indent=2)

    def create(
        self,
        modules: Optional[List[str]],
        params: Dict[str, Any],
        data: SessionData,
        config_service: Optional["ConfigService"] = None,
    ) -> Tuple[str, SessionData]:
        """Create a new chat session.

        Args:
            modules: List of module names to load.
            params: Session parameters (model, temperature, etc).
            data: Current session data.
            config_service: Optional ConfigService for applying defaults.

        Returns:
            Tuple of (new_session_id, updated_SessionData).
        """
        # Apply config defaults if service provided
        if config_service:
            params = self._apply_config_defaults(params, config_service)

        new_id = str(uuid.uuid4())
        new_session = {
            "title": "New Session",
            "created_at": str(datetime.now()),
            "messages": [],
            "modules": modules,
            "params": params,  # Now contains complete defaults
            "title_needs_update": True,
        }

        # Create new sessions dict with the new session
        new_sessions = dict(data.sessions)
        new_sessions[new_id] = new_session

        return new_id, SessionData(current_id=new_id, sessions=new_sessions)

    def update_title(
        self, session_id: str, title: str, data: SessionData
    ) -> SessionData:
        """Update the title of a session.

        Args:
            session_id: Session ID to update.
            title: New title.
            data: Current session data.

        Returns:
            Updated SessionData.
        """
        if session_id not in data.sessions:
            return data

        new_sessions = dict(data.sessions)
        new_sessions[session_id] = dict(new_sessions[session_id])
        new_sessions[session_id]["title"] = title
        new_sessions[session_id]["title_needs_update"] = False

        return SessionData(current_id=data.current_id, sessions=new_sessions)

    def set_title_needs_update(
        self, session_id: str, needs_update: bool, data: SessionData
    ) -> SessionData:
        """Set the title_needs_update flag for a session.

        Args:
            session_id: Session ID to update.
            needs_update: Whether title needs update.
            data: Current session data.

        Returns:
            Updated SessionData.
        """
        if session_id not in data.sessions:
            return data

        new_sessions = dict(data.sessions)
        new_sessions[session_id] = dict(new_sessions[session_id])
        new_sessions[session_id]["title_needs_update"] = needs_update

        return SessionData(current_id=data.current_id, sessions=new_sessions)

    def delete(
        self,
        session_id: str,
        data: SessionData,
        session_dir: Optional[Path] = None,
    ) -> SessionData:
        """Delete a session and optionally its associated files.

        Args:
            session_id: Session ID to delete.
            data: Current session data.
            session_dir: Optional path to session directory (PDFs, index, etc).

        Returns:
            Updated SessionData without the deleted session.
        """
        if session_id not in data.sessions:
            return data

        # Remove session from data
        new_sessions = {
            sid: sess for sid, sess in data.sessions.items() if sid != session_id
        }

        # Update current_id if we deleted the current session
        new_current_id = data.current_id
        if data.current_id == session_id:
            new_current_id = None

        # Delete session directory if provided
        if session_dir and session_dir.exists():
            shutil.rmtree(session_dir)

        return SessionData(current_id=new_current_id, sessions=new_sessions)

    def add_message(
        self,
        session_id: str,
        message: Dict[str, Any],
        data: SessionData,
    ) -> SessionData:
        """Add a message to a session.

        Args:
            session_id: Session ID.
            message: Message dict with 'role' and 'content'.
            data: Current session data.

        Returns:
            Updated SessionData with the new message.
        """
        if session_id not in data.sessions:
            return data

        new_sessions = dict(data.sessions)
        new_sessions[session_id] = dict(new_sessions[session_id])
        new_sessions[session_id]["messages"] = list(
            new_sessions[session_id].get("messages", [])
        )
        new_sessions[session_id]["messages"].append(message)

        return SessionData(current_id=data.current_id, sessions=new_sessions)

    def update_last_message(
        self,
        session_id: str,
        content: str,
        data: SessionData,
    ) -> SessionData:
        """Update the content of the last message in a session.

        Args:
            session_id: Session ID.
            content: New content for the last message.
            data: Current session data.

        Returns:
            Updated SessionData.
        """
        if session_id not in data.sessions:
            return data

        messages = data.sessions[session_id].get("messages", [])
        if not messages:
            return data

        new_sessions = dict(data.sessions)
        new_sessions[session_id] = dict(new_sessions[session_id])
        new_sessions[session_id]["messages"] = list(messages)
        new_sessions[session_id]["messages"][-1] = dict(messages[-1])
        new_sessions[session_id]["messages"][-1]["content"] = content

        return SessionData(current_id=data.current_id, sessions=new_sessions)

    def set_current(self, session_id: str, data: SessionData) -> SessionData:
        """Set the current active session.

        Args:
            session_id: Session ID to make current.
            data: Current session data.

        Returns:
            Updated SessionData with new current_id.
        """
        if session_id not in data.sessions:
            return data

        return SessionData(current_id=session_id, sessions=data.sessions)

    def get_session(
        self, session_id: str, data: SessionData
    ) -> Optional[Dict[str, Any]]:
        """Get a session by ID.

        Args:
            session_id: Session ID.
            data: Current session data.

        Returns:
            Session dict or None if not found.
        """
        return data.sessions.get(session_id)

    def get_current_session(self, data: SessionData) -> Optional[Dict[str, Any]]:
        """Get the current active session.

        Args:
            data: Current session data.

        Returns:
            Current session dict or None if no current session.
        """
        if not data.current_id:
            return None
        return data.sessions.get(data.current_id)

    def get_messages(self, session_id: str, data: SessionData) -> List[Dict[str, Any]]:
        """Get messages for a session.

        Args:
            session_id: Session ID.
            data: Current session data.

        Returns:
            List of message dicts.
        """
        session = data.sessions.get(session_id)
        if not session:
            return []
        return session.get("messages", [])

    def needs_title_update(self, session_id: str, data: SessionData) -> bool:
        """Check if a session needs its title updated.

        Args:
            session_id: Session ID.
            data: Current session data.

        Returns:
            True if title needs update.
        """
        session = data.sessions.get(session_id)
        if not session:
            return False
        return session.get("title_needs_update", False)
