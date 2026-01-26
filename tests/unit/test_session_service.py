"""Unit tests for SessionService."""

import json
from pathlib import Path

import pytest

from tensortruth.services.models import SessionData
from tensortruth.services.session_service import SessionService


@pytest.fixture
def temp_sessions_file(tmp_path: Path) -> Path:
    """Create a temporary legacy sessions file path."""
    return tmp_path / "chat_sessions.json"


@pytest.fixture
def temp_sessions_dir(tmp_path: Path) -> Path:
    """Create a temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return sessions_dir


@pytest.fixture
def session_service(
    temp_sessions_file: Path, temp_sessions_dir: Path
) -> SessionService:
    """Create a SessionService instance with temp paths."""
    return SessionService(
        sessions_file=temp_sessions_file,
        sessions_dir=temp_sessions_dir,
    )


@pytest.fixture
def sample_session_data() -> SessionData:
    """Create sample session data for testing."""
    return SessionData(
        current_id="session-1",
        sessions={
            "session-1": {
                "title": "Test Session",
                "created_at": "2024-01-01 12:00:00",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "modules": ["module1"],
                "params": {"model": "llama2"},
                "title_needs_update": False,
            },
            "session-2": {
                "title": "New Session",
                "created_at": "2024-01-02 12:00:00",
                "messages": [],
                "modules": None,
                "params": {},
                "title_needs_update": True,
            },
        },
    )


class TestSessionServiceLoad:
    """Tests for SessionService.load()."""

    def test_load_empty_returns_empty_data(self, session_service: SessionService):
        """Load returns empty data when no sessions exist."""
        data = session_service.load()

        assert data.current_id is None
        assert data.sessions == {}

    def test_load_from_per_session_files(
        self, session_service: SessionService, temp_sessions_dir: Path
    ):
        """Load returns data from per-session files."""
        # Create a session directory with session.json
        session_id = "test-session"
        session_dir = temp_sessions_dir / session_id
        session_dir.mkdir()
        session_data = {
            "title": "Test Session",
            "created_at": "2024-01-01 12:00:00",
            "messages": [{"role": "user", "content": "Hello"}],
            "modules": ["module1"],
            "params": {},
            "title_needs_update": False,
        }
        (session_dir / "session.json").write_text(json.dumps(session_data))

        # Create index file
        index = {
            "current_id": session_id,
            "sessions": {
                session_id: {
                    "title": "Test Session",
                    "created_at": "2024-01-01 12:00:00",
                }
            },
        }
        (temp_sessions_dir / "sessions_index.json").write_text(json.dumps(index))

        data = session_service.load()

        assert data.current_id == session_id
        assert session_id in data.sessions
        assert data.sessions[session_id]["title"] == "Test Session"
        assert len(data.sessions[session_id]["messages"]) == 1

    def test_load_removes_stale_index_entries(
        self, session_service: SessionService, temp_sessions_dir: Path
    ):
        """Load removes index entries for missing session files."""
        # Create index with a session that doesn't have a file
        index = {
            "current_id": "missing-session",
            "sessions": {
                "missing-session": {
                    "title": "Missing",
                    "created_at": "2024-01-01 12:00:00",
                }
            },
        }
        (temp_sessions_dir / "sessions_index.json").write_text(json.dumps(index))

        data = session_service.load()

        # Should return empty data (stale entry removed)
        assert data.current_id is None
        assert data.sessions == {}

        # Index should be updated
        updated_index = json.loads(
            (temp_sessions_dir / "sessions_index.json").read_text()
        )
        assert updated_index["sessions"] == {}
        assert updated_index["current_id"] is None


class TestSessionServiceMigration:
    """Tests for legacy file migration."""

    def test_migrate_from_legacy_file(
        self,
        session_service: SessionService,
        temp_sessions_file: Path,
        temp_sessions_dir: Path,
    ):
        """Migration creates per-session files from legacy file."""
        # Create legacy file
        legacy_data = {
            "current_id": "session-1",
            "sessions": {
                "session-1": {
                    "title": "Legacy Session",
                    "created_at": "2024-01-01 12:00:00",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "modules": ["module1"],
                    "params": {"model": "llama2"},
                },
                "session-2": {
                    "title": "Another Session",
                    "created_at": "2024-01-02 12:00:00",
                    "messages": [],
                    "modules": None,
                    "params": {},
                },
            },
        }
        temp_sessions_file.write_text(json.dumps(legacy_data))

        # Load triggers migration
        data = session_service.load()

        # Verify data loaded correctly
        assert data.current_id == "session-1"
        assert len(data.sessions) == 2
        assert data.sessions["session-1"]["title"] == "Legacy Session"

        # Verify per-session files created
        assert (temp_sessions_dir / "session-1" / "session.json").exists()
        assert (temp_sessions_dir / "session-2" / "session.json").exists()

        # Verify index created
        assert (temp_sessions_dir / "sessions_index.json").exists()
        index = json.loads((temp_sessions_dir / "sessions_index.json").read_text())
        assert "session-1" in index["sessions"]
        assert "session-2" in index["sessions"]

        # Verify legacy file deleted
        assert not temp_sessions_file.exists()

    def test_migrate_adds_title_needs_update_flag(
        self,
        session_service: SessionService,
        temp_sessions_file: Path,
        temp_sessions_dir: Path,
    ):
        """Migration adds title_needs_update flag to sessions missing it."""
        legacy_data = {
            "current_id": None,
            "sessions": {
                "needs-update": {
                    "title": "New Session",  # Default title
                    "messages": [],
                },
                "no-update": {
                    "title": "Custom Title",  # Custom title
                    "messages": [],
                },
            },
        }
        temp_sessions_file.write_text(json.dumps(legacy_data))

        data = session_service.load()

        # Default title should need update
        assert data.sessions["needs-update"]["title_needs_update"] is True
        # Custom title should not need update
        assert data.sessions["no-update"]["title_needs_update"] is False

    def test_migrate_idempotent(
        self,
        session_service: SessionService,
        temp_sessions_file: Path,
        temp_sessions_dir: Path,
    ):
        """Migration is idempotent - doesn't overwrite existing sessions."""
        # Create existing per-session file
        session_dir = temp_sessions_dir / "session-1"
        session_dir.mkdir()
        existing_session = {
            "title": "Existing Session",
            "messages": [{"role": "user", "content": "Existing message"}],
        }
        (session_dir / "session.json").write_text(json.dumps(existing_session))

        # Create index
        index = {
            "current_id": "session-1",
            "sessions": {
                "session-1": {"title": "Existing Session", "created_at": "2024-01-01"}
            },
        }
        (temp_sessions_dir / "sessions_index.json").write_text(json.dumps(index))

        # Create legacy file with different content
        legacy_data = {
            "current_id": "session-1",
            "sessions": {
                "session-1": {
                    "title": "Legacy Session",  # Different title
                    "messages": [{"role": "user", "content": "Legacy message"}],
                }
            },
        }
        temp_sessions_file.write_text(json.dumps(legacy_data))

        # Load - should NOT overwrite existing
        data = session_service.load()

        # Should have existing session data, not legacy
        assert data.sessions["session-1"]["title"] == "Existing Session"
        assert (
            data.sessions["session-1"]["messages"][0]["content"] == "Existing message"
        )

        # Legacy file should be deleted
        assert not temp_sessions_file.exists()


class TestSessionServiceSave:
    """Tests for SessionService.save()."""

    def test_save_creates_per_session_files(
        self, session_service: SessionService, temp_sessions_dir: Path
    ):
        """Save creates individual session files."""
        data = SessionData(
            current_id="new-id",
            sessions={
                "new-id": {
                    "title": "Test Session",
                    "created_at": "2024-01-01 12:00:00",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "modules": None,
                    "params": {},
                    "title_needs_update": False,
                }
            },
        )

        session_service.save(data)

        # Verify session file created
        session_file = temp_sessions_dir / "new-id" / "session.json"
        assert session_file.exists()
        saved_session = json.loads(session_file.read_text())
        assert saved_session["title"] == "Test Session"

        # Verify index created
        index_file = temp_sessions_dir / "sessions_index.json"
        assert index_file.exists()
        index = json.loads(index_file.read_text())
        assert index["current_id"] == "new-id"
        assert "new-id" in index["sessions"]

    def test_save_filters_empty_sessions(
        self, session_service: SessionService, temp_sessions_dir: Path
    ):
        """Save removes empty sessions except current."""
        data = SessionData(
            current_id="current",
            sessions={
                "current": {
                    "title": "Current",
                    "created_at": "2024-01-01",
                    "messages": [],
                },  # Keep - current
                "with-msgs": {
                    "title": "Has Messages",
                    "created_at": "2024-01-01",
                    "messages": [{"role": "user", "content": "hi"}],
                },  # Keep
                "empty": {
                    "title": "Empty",
                    "created_at": "2024-01-01",
                    "messages": [],
                },  # Remove
            },
        )

        session_service.save(data)

        # Verify correct files created
        assert (temp_sessions_dir / "current" / "session.json").exists()
        assert (temp_sessions_dir / "with-msgs" / "session.json").exists()
        assert not (temp_sessions_dir / "empty" / "session.json").exists()

        # Verify index
        index = json.loads((temp_sessions_dir / "sessions_index.json").read_text())
        assert "current" in index["sessions"]
        assert "with-msgs" in index["sessions"]
        assert "empty" not in index["sessions"]

    def test_save_deletes_removed_sessions(
        self, session_service: SessionService, temp_sessions_dir: Path
    ):
        """Save deletes session files for removed sessions."""
        # Create existing session
        session_dir = temp_sessions_dir / "to-delete"
        session_dir.mkdir()
        (session_dir / "session.json").write_text('{"title": "To Delete"}')

        # Create index with the session
        index = {
            "current_id": None,
            "sessions": {
                "to-delete": {"title": "To Delete", "created_at": "2024-01-01"}
            },
        }
        (temp_sessions_dir / "sessions_index.json").write_text(json.dumps(index))

        # Save with empty data (session removed)
        session_service.save(SessionData(current_id=None, sessions={}))

        # Session directory should be deleted
        assert not session_dir.exists()


class TestSessionServiceCreate:
    """Tests for SessionService.create()."""

    def test_create_new_session(self, session_service: SessionService):
        """Create generates new session with correct properties."""
        data = SessionData(current_id=None, sessions={})

        new_id, new_data = session_service.create(
            modules=["module1", "module2"],
            params={"model": "llama2", "temperature": 0.7},
            data=data,
        )

        assert new_id is not None
        assert new_data.current_id == new_id
        assert new_id in new_data.sessions

        session = new_data.sessions[new_id]
        assert session["title"] == "New Session"
        assert session["modules"] == ["module1", "module2"]
        assert session["params"]["model"] == "llama2"
        assert session["title_needs_update"] is True
        assert session["messages"] == []

    def test_create_preserves_existing_sessions(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Create preserves existing sessions in the data."""
        new_id, new_data = session_service.create(
            modules=None, params={}, data=sample_session_data
        )

        # Original sessions still exist
        assert "session-1" in new_data.sessions
        assert "session-2" in new_data.sessions
        # New session added
        assert new_id in new_data.sessions


class TestSessionServiceUpdateTitle:
    """Tests for SessionService.update_title()."""

    def test_update_title_success(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Update title changes the session title."""
        new_data = session_service.update_title(
            session_id="session-1",
            title="Updated Title",
            data=sample_session_data,
        )

        assert new_data.sessions["session-1"]["title"] == "Updated Title"
        assert new_data.sessions["session-1"]["title_needs_update"] is False

    def test_update_title_nonexistent_session(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Update title returns unchanged data for nonexistent session."""
        new_data = session_service.update_title(
            session_id="nonexistent",
            title="New Title",
            data=sample_session_data,
        )

        # Data unchanged
        assert new_data == sample_session_data


class TestSessionServiceDelete:
    """Tests for SessionService.delete()."""

    def test_delete_session(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Delete removes session from data."""
        new_data = session_service.delete(
            session_id="session-2",
            data=sample_session_data,
        )

        assert "session-2" not in new_data.sessions
        assert "session-1" in new_data.sessions

    def test_delete_current_session_clears_current_id(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Delete current session clears current_id."""
        new_data = session_service.delete(
            session_id="session-1",
            data=sample_session_data,
        )

        assert new_data.current_id is None

    def test_delete_with_session_dir(
        self,
        session_service: SessionService,
        sample_session_data: SessionData,
        tmp_path: Path,
    ):
        """Delete removes session directory if provided."""
        session_dir = tmp_path / "session-1"
        session_dir.mkdir()
        (session_dir / "some_file.txt").write_text("test")

        session_service.delete(
            session_id="session-1",
            data=sample_session_data,
            session_dir=session_dir,
        )

        assert not session_dir.exists()


class TestSessionServiceMessages:
    """Tests for message-related methods."""

    def test_add_message(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Add message appends to session messages."""
        new_message = {"role": "user", "content": "New message"}

        new_data = session_service.add_message(
            session_id="session-1",
            message=new_message,
            data=sample_session_data,
        )

        messages = new_data.sessions["session-1"]["messages"]
        assert len(messages) == 3
        assert messages[-1] == new_message

    def test_update_last_message(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Update last message modifies the final message."""
        new_data = session_service.update_last_message(
            session_id="session-1",
            content="Updated content",
            data=sample_session_data,
        )

        messages = new_data.sessions["session-1"]["messages"]
        assert messages[-1]["content"] == "Updated content"

    def test_get_messages(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Get messages returns session messages."""
        messages = session_service.get_messages("session-1", sample_session_data)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"

    def test_get_messages_empty_session(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Get messages returns empty list for session with no messages."""
        messages = session_service.get_messages("session-2", sample_session_data)

        assert messages == []


class TestSessionServiceHelpers:
    """Tests for helper methods."""

    def test_set_current(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Set current changes the current session ID."""
        new_data = session_service.set_current("session-2", sample_session_data)

        assert new_data.current_id == "session-2"

    def test_get_session(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Get session returns the session dict."""
        session = session_service.get_session("session-1", sample_session_data)

        assert session is not None
        assert session["title"] == "Test Session"

    def test_get_current_session(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Get current session returns the active session."""
        session = session_service.get_current_session(sample_session_data)

        assert session is not None
        assert session["title"] == "Test Session"

    def test_needs_title_update(
        self, session_service: SessionService, sample_session_data: SessionData
    ):
        """Needs title update returns correct flag value."""
        assert (
            session_service.needs_title_update("session-1", sample_session_data)
            is False
        )
        assert (
            session_service.needs_title_update("session-2", sample_session_data) is True
        )


class TestAtomicWrites:
    """Tests for atomic write functionality."""

    def test_atomic_write_creates_file(self, tmp_path: Path):
        """atomic_write_json creates file with correct content."""
        from tensortruth.app_utils.file_utils import atomic_write_json

        file_path = tmp_path / "test.json"
        data = {"key": "value", "nested": {"a": 1}}

        atomic_write_json(file_path, data)

        assert file_path.exists()
        loaded = json.loads(file_path.read_text())
        assert loaded == data

    def test_atomic_write_overwrites_existing(self, tmp_path: Path):
        """atomic_write_json overwrites existing file."""
        from tensortruth.app_utils.file_utils import atomic_write_json

        file_path = tmp_path / "test.json"
        file_path.write_text('{"old": "data"}')

        atomic_write_json(file_path, {"new": "data"})

        loaded = json.loads(file_path.read_text())
        assert loaded == {"new": "data"}

    def test_atomic_write_creates_parent_dirs(self, tmp_path: Path):
        """atomic_write_json creates parent directories."""
        from tensortruth.app_utils.file_utils import atomic_write_json

        file_path = tmp_path / "nested" / "path" / "test.json"

        atomic_write_json(file_path, {"data": True})

        assert file_path.exists()
