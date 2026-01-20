"""Integration tests for sessions API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from tensortruth.api.main import create_app
from tensortruth.services import SessionService


@pytest.fixture
def app():
    """Create test application."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def temp_sessions_file(tmp_path):
    """Create a temporary sessions file."""
    sessions_file = tmp_path / "chat_sessions.json"
    return sessions_file


@pytest.fixture
def session_service_with_data(temp_sessions_file):
    """Create a session service with test data."""
    service = SessionService(sessions_file=temp_sessions_file)
    data = service.load()

    # Create a test session
    session_id, data = service.create(
        modules=["pytorch"], params={"model": "test-model"}, data=data
    )
    data = service.update_title(session_id, "Test Session", data)
    data = service.add_message(session_id, {"role": "user", "content": "Hello"}, data)
    service.save(data)

    return service, session_id


class TestSessionsAPI:
    """Test session CRUD endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, client, tmp_path, monkeypatch):
        """Test listing sessions when none exist."""
        # Use temp directory for sessions
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        # Clear LRU cache
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        response = await client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["current_id"] is None

    @pytest.mark.asyncio
    async def test_create_session(self, client, tmp_path, monkeypatch):
        """Test creating a new session."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        response = await client.post(
            "/sessions",
            json={"modules": ["pytorch"], "params": {"model": "test-model"}},
        )
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["title"] == "New Session"
        assert data["modules"] == ["pytorch"]
        assert data["params"]["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_get_session(self, client, tmp_path, monkeypatch):
        """Test getting a session by ID."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        # Create a session first
        create_response = await client.post("/sessions", json={"modules": ["test"]})
        session_id = create_response.json()["session_id"]

        # Get the session
        response = await client.get(f"/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client, tmp_path, monkeypatch):
        """Test getting a non-existent session."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        response = await client.get("/sessions/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_session_title(self, client, tmp_path, monkeypatch):
        """Test updating a session title."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        # Create a session
        create_response = await client.post("/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Update title
        response = await client.patch(
            f"/sessions/{session_id}", json={"title": "Updated Title"}
        )
        assert response.status_code == 200
        assert response.json()["title"] == "Updated Title"

    @pytest.mark.asyncio
    async def test_delete_session(self, client, tmp_path, monkeypatch):
        """Test deleting a session."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        # Create a session
        create_response = await client.post("/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Delete it
        response = await client.delete(f"/sessions/{session_id}")
        assert response.status_code == 204

        # Verify it's gone
        response = await client.get(f"/sessions/{session_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_add_message(self, client, tmp_path, monkeypatch):
        """Test adding a message to a session."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        # Create a session
        create_response = await client.post("/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Add a message
        response = await client.post(
            f"/sessions/{session_id}/messages",
            json={"role": "user", "content": "Hello, world!"},
        )
        assert response.status_code == 201
        assert response.json()["role"] == "user"
        assert response.json()["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_get_messages(self, client, tmp_path, monkeypatch):
        """Test getting messages from a session."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        # Create a session and add messages
        create_response = await client.post("/sessions", json={})
        session_id = create_response.json()["session_id"]

        await client.post(
            f"/sessions/{session_id}/messages",
            json={"role": "user", "content": "Hello"},
        )
        await client.post(
            f"/sessions/{session_id}/messages",
            json={"role": "assistant", "content": "Hi there!"},
        )

        # Get messages
        response = await client.get(f"/sessions/{session_id}/messages")
        assert response.status_code == 200
        messages = response.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi there!"
