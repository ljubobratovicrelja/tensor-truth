"""Session management service - pure business logic.

This service handles all session CRUD operations, returning new state objects
instead of mutating global state.
"""

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from tensortruth.app_utils.file_utils import atomic_write_json

from .models import SessionData

if TYPE_CHECKING:
    from .config_service import ConfigService

logger = logging.getLogger(__name__)


class SessionService:
    """Service for managing chat sessions.

    All methods are pure - they accept state as input and return new state
    as output.

    Storage model:
    - Per-session files: ~/.tensortruth/sessions/{session_id}/session.json
    - Index file: ~/.tensortruth/sessions/sessions_index.json (cache for fast listing)
    - Per-session files are authoritative; index is a cache
    """

    def __init__(
        self,
        sessions_file: Union[str, Path],
        sessions_dir: Path,
    ):
        """Initialize session service.

        Args:
            sessions_file: Path to the legacy JSON file (for migration).
            sessions_dir: Path to the sessions directory.
        """
        self.legacy_sessions_file = Path(sessions_file)
        self.sessions_dir = Path(sessions_dir)
        self.index_file = self.sessions_dir / "sessions_index.json"

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
                "reranker_model": config.rag.default_reranker,
                "reranker_top_n": config.ui.default_top_n,
                "confidence_cutoff": config.ui.default_confidence_threshold,
                "confidence_cutoff_hard": config.ui.default_confidence_cutoff_hard,
                "rag_device": config.rag.default_device,
                "balance_strategy": config.rag.default_balance_strategy,
                "embedding_model": config.rag.default_embedding_model,
                "llm_device": "gpu",  # Reasonable default
                "router_model": config.agent.router_model,
                "orchestrator_enabled": config.agent.orchestrator_enabled,
            }

            # User params override defaults
            return {**defaults, **params}

        except Exception as e:
            # If config loading fails, return params as-is
            logger.warning(f"Failed to apply config defaults: {e}")
            return params

    # -------------------------------------------------------------------------
    # Internal file operations
    # -------------------------------------------------------------------------

    def _load_index(self) -> Dict[str, Any]:
        """Load the sessions index file.

        Returns:
            Index dict with current_id and sessions metadata.
        """
        if not self.index_file.exists():
            return {"current_id": None, "sessions": {}}

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load sessions index: {e}")
            return {"current_id": None, "sessions": {}}

    def _save_index(self, index: Dict[str, Any]) -> None:
        """Save the sessions index file atomically.

        Args:
            index: Index dict with current_id and sessions metadata.
        """
        atomic_write_json(self.index_file, index)

    def _load_session_file(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a single session's data file.

        Args:
            session_id: Session ID.

        Returns:
            Session data dict or None if file doesn't exist or is corrupted.
        """
        session_file = self.sessions_dir / session_id / "session.json"
        if not session_file.exists():
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            return None

    def _save_session_file(self, session_id: str, data: Dict[str, Any]) -> None:
        """Save a single session's data file atomically.

        Args:
            session_id: Session ID.
            data: Session data dict.
        """
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / "session.json"
        atomic_write_json(session_file, data)

    def _delete_session_file(self, session_id: str) -> None:
        """Delete a session's data file and directory.

        Args:
            session_id: Session ID.
        """
        session_dir = self.sessions_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def _migrate_from_legacy(self) -> None:
        """Migrate from legacy single-file storage to per-session files.

        Called on load() when legacy file exists. Migration is idempotent:
        - If index already has sessions, does nothing
        - Migrates each session to its own file
        - Deletes legacy file after successful migration
        """
        if not self.legacy_sessions_file.exists():
            return

        # Check if already migrated (index has sessions)
        index = self._load_index()
        if index.get("sessions"):
            # Already migrated, just remove legacy file
            logger.info("Sessions already migrated, removing legacy file")
            try:
                self.legacy_sessions_file.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete legacy sessions file: {e}")
            return

        # Load legacy data
        try:
            with open(self.legacy_sessions_file, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load legacy sessions file: {e}")
            return

        # Migrate each session
        legacy_sessions = legacy_data.get("sessions", {})
        new_index = {
            "current_id": legacy_data.get("current_id"),
            "sessions": {},
        }

        for session_id, session_data in legacy_sessions.items():
            # Apply title_needs_update migration if missing
            if "title_needs_update" not in session_data:
                session_data["title_needs_update"] = (
                    session_data.get("title") == "New Session"
                )

            # Save session to individual file
            self._save_session_file(session_id, session_data)

            # Add to index (lightweight metadata only)
            new_index["sessions"][session_id] = {
                "title": session_data.get("title", "New Session"),
                "created_at": session_data.get("created_at", str(datetime.now())),
            }

        # Save new index
        self._save_index(new_index)

        # Delete legacy file
        try:
            self.legacy_sessions_file.unlink()
            logger.info(
                f"Successfully migrated {len(legacy_sessions)} sessions "
                "from legacy file"
            )
        except OSError as e:
            logger.warning(f"Failed to delete legacy sessions file: {e}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load(self) -> SessionData:
        """Load chat sessions from per-session files.

        Migration: If legacy file exists, migrates to per-session format.

        Returns:
            SessionData with current_id and sessions dict.
            Returns empty data if no sessions exist.
        """
        # Run migration if legacy file exists
        self._migrate_from_legacy()

        # Load index
        index = self._load_index()
        current_id = index.get("current_id")
        index_sessions = index.get("sessions", {})

        # Load each session file
        sessions: Dict[str, Dict[str, Any]] = {}
        stale_ids: List[str] = []

        for session_id in index_sessions:
            session_data = self._load_session_file(session_id)
            if session_data is not None:
                sessions[session_id] = session_data
            else:
                # Session file missing - stale index entry
                logger.warning(
                    f"Removing stale index entry for missing session: {session_id}"
                )
                stale_ids.append(session_id)

        # Clean up stale entries from index
        if stale_ids:
            for session_id in stale_ids:
                del index_sessions[session_id]
            # Fix current_id if it's stale
            if current_id in stale_ids:
                current_id = None
                index["current_id"] = None
            self._save_index(index)

        return SessionData(current_id=current_id, sessions=sessions)

    def save(self, data: SessionData) -> None:
        """Save chat sessions to per-session files.

        Filters out empty sessions (no messages) except for the current session.

        Args:
            data: SessionData to save.
        """
        # Load existing index to find removed sessions
        old_index = self._load_index()
        old_session_ids = set(old_index.get("sessions", {}).keys())

        # Filter out empty sessions (sessions with no messages)
        filtered_sessions: Dict[str, Dict[str, Any]] = {}
        for session_id, session in data.sessions.items():
            messages = session.get("messages", [])
            # Always keep the current session or sessions with messages
            if session_id == data.current_id or messages:
                filtered_sessions[session_id] = session

        # Build new index
        new_index: Dict[str, Any] = {
            "current_id": data.current_id,
            "sessions": {},
        }

        # Save each session file and update index
        for session_id, session_data in filtered_sessions.items():
            self._save_session_file(session_id, session_data)
            new_index["sessions"][session_id] = {
                "title": session_data.get("title", "New Session"),
                "created_at": session_data.get("created_at", str(datetime.now())),
            }

        # Save index
        self._save_index(new_index)

        # Delete removed session files
        new_session_ids = set(filtered_sessions.keys())
        removed_ids = old_session_ids - new_session_ids
        for session_id in removed_ids:
            self._delete_session_file(session_id)

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
                         Note: session.json is automatically deleted via save().

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
