"""Unit tests for SessionService."""

import json
from pathlib import Path

import pytest

from tensortruth.services.models import SessionData
from tensortruth.services.session_service import SessionService


@pytest.fixture
def temp_sessions_file(tmp_path: Path) -> Path:
    """Create a temporary sessions file path."""
    return tmp_path / "chat_sessions.json"


@pytest.fixture
def session_service(temp_sessions_file: Path) -> SessionService:
    """Create a SessionService instance with a temp file."""
    return SessionService(temp_sessions_file)


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

    def test_load_empty_file(self, session_service: SessionService):
        """Load returns empty data when file doesn't exist."""
        data = session_service.load()

        assert data.current_id is None
        assert data.sessions == {}

    def test_load_existing_file(
        self, session_service: SessionService, temp_sessions_file: Path
    ):
        """Load returns data from existing file."""
        existing_data = {
            "current_id": "test-id",
            "sessions": {
                "test-id": {
                    "title": "Loaded Session",
                    "messages": [{"role": "user", "content": "test"}],
                }
            },
        }
        temp_sessions_file.write_text(json.dumps(existing_data))

        data = session_service.load()

        assert data.current_id == "test-id"
        assert "test-id" in data.sessions
        assert data.sessions["test-id"]["title"] == "Loaded Session"

    def test_load_corrupted_file(
        self, session_service: SessionService, temp_sessions_file: Path
    ):
        """Load returns empty data for corrupted file."""
        temp_sessions_file.write_text("not valid json {{{")

        data = session_service.load()

        assert data.current_id is None
        assert data.sessions == {}

    def test_load_migrates_title_needs_update(
        self, session_service: SessionService, temp_sessions_file: Path
    ):
        """Load adds title_needs_update flag to old sessions."""
        old_data = {
            "current_id": None,
            "sessions": {
                "old-session": {
                    "title": "New Session",  # Default title
                    "messages": [],
                }
            },
        }
        temp_sessions_file.write_text(json.dumps(old_data))

        data = session_service.load()

        # Should have added the flag
        assert data.sessions["old-session"]["title_needs_update"] is True


class TestSessionServiceSave:
    """Tests for SessionService.save()."""

    def test_save_creates_file(
        self, session_service: SessionService, temp_sessions_file: Path
    ):
        """Save creates file with correct content."""
        data = SessionData(
            current_id="new-id",
            sessions={
                "new-id": {
                    "title": "Test",
                    "messages": [{"role": "user", "content": "msg"}],
                }
            },
        )

        session_service.save(data)

        assert temp_sessions_file.exists()
        saved = json.loads(temp_sessions_file.read_text())
        assert saved["current_id"] == "new-id"
        assert "new-id" in saved["sessions"]

    def test_save_filters_empty_sessions(
        self, session_service: SessionService, temp_sessions_file: Path
    ):
        """Save removes empty sessions except current."""
        data = SessionData(
            current_id="current",
            sessions={
                "current": {"title": "Current", "messages": []},  # Keep - current
                "with-msgs": {
                    "title": "Has Messages",
                    "messages": [{"role": "user", "content": "hi"}],
                },  # Keep - has messages
                "empty": {"title": "Empty", "messages": []},  # Remove - empty
            },
        )

        session_service.save(data)

        saved = json.loads(temp_sessions_file.read_text())
        assert "current" in saved["sessions"]  # Kept as current
        assert "with-msgs" in saved["sessions"]  # Kept - has messages
        assert "empty" not in saved["sessions"]  # Filtered out


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
        self, session_service: SessionService, sample_session_data: SessionData, tmp_path: Path
    ):
        """Delete removes session directory if provided."""
        session_dir = tmp_path / "session-1"
        session_dir.mkdir()
        (session_dir / "some_file.txt").write_text("test")

        new_data = session_service.delete(
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
        assert session_service.needs_title_update("session-1", sample_session_data) is False
        assert session_service.needs_title_update("session-2", sample_session_data) is True
