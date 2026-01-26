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
def temp_sessions_dir(tmp_path):
    """Create a temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return sessions_dir


@pytest.fixture
def session_service_with_data(temp_sessions_file, temp_sessions_dir):
    """Create a session service with test data."""
    service = SessionService(
        sessions_file=temp_sessions_file, sessions_dir=temp_sessions_dir
    )
    data = service.load()

    # Create a test session
    session_id, data = service.create(
        modules=["pytorch"], params={"model": "test-model"}, data=data
    )
    data = service.update_title(session_id, "Test Session", data)
    data = service.add_message(session_id, {"role": "user", "content": "Hello"}, data)
    service.save(data)

    return service, session_id


@pytest.fixture
def mock_session_paths(tmp_path, monkeypatch):
    """Patch session paths to use temp directory."""
    sessions_file = tmp_path / "chat_sessions.json"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    monkeypatch.setattr("tensortruth.api.deps.get_sessions_file", lambda: sessions_file)
    monkeypatch.setattr(
        "tensortruth.api.deps.get_sessions_data_dir", lambda: sessions_dir
    )

    from tensortruth.api.deps import get_session_service

    get_session_service.cache_clear()

    return sessions_file, sessions_dir


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
    async def test_list_sessions_empty(self, client, mock_session_paths):
        """Test listing sessions when none exist."""
        response = await client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["current_id"] is None

    @pytest.mark.asyncio
    async def test_create_session(self, client, mock_session_paths):
        """Test creating a new session."""
        response = await client.post(
            "/api/sessions",
            json={"modules": ["pytorch"], "params": {"model": "test-model"}},
        )
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["title"] == "New Session"
        assert data["modules"] == ["pytorch"]
        assert data["params"]["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_get_session(self, client, mock_session_paths):
        """Test getting a session by ID."""
        # Create a session first
        create_response = await client.post("/api/sessions", json={"modules": ["test"]})
        session_id = create_response.json()["session_id"]

        # Get the session
        response = await client.get(f"/api/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client, mock_session_paths):
        """Test getting a non-existent session."""
        response = await client.get("/api/sessions/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_session_title(self, client, mock_session_paths):
        """Test updating a session title."""
        # Create a session
        create_response = await client.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Update title
        response = await client.patch(
            f"/api/sessions/{session_id}", json={"title": "Updated Title"}
        )
        assert response.status_code == 200
        assert response.json()["title"] == "Updated Title"

    @pytest.mark.asyncio
    async def test_delete_session(self, client, mock_session_paths):
        """Test deleting a session."""
        # Create a session
        create_response = await client.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Delete it
        response = await client.delete(f"/api/sessions/{session_id}")
        assert response.status_code == 204

        # Verify it's gone
        response = await client.get(f"/api/sessions/{session_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_add_message(self, client, mock_session_paths):
        """Test adding a message to a session."""
        # Create a session
        create_response = await client.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Add a message
        response = await client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "user", "content": "Hello, world!"},
        )
        assert response.status_code == 201
        assert response.json()["role"] == "user"
        assert response.json()["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_get_messages(self, client, mock_session_paths):
        """Test getting messages from a session."""
        # Create a session and add messages
        create_response = await client.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        await client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "user", "content": "Hello"},
        )
        await client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "assistant", "content": "Hi there!"},
        )

        # Get messages
        response = await client.get(f"/api/sessions/{session_id}/messages")
        assert response.status_code == 200
        messages = response.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi there!"


class TestSessionStatsEndpoint:
    """Tests for GET /api/sessions/{session_id}/stats endpoint."""

    @pytest.fixture
    async def session_with_messages(self, client, mock_session_paths):
        """Create session with sample messages for stats testing."""
        # Create a session with a specific model
        create_response = await client.post(
            "/api/sessions",
            json={"modules": ["pytorch"], "params": {"model": "test-model:7b"}},
        )
        session_id = create_response.json()["session_id"]

        # Add some messages
        await client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "user", "content": "Hello, world!"},
        )
        await client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "assistant", "content": "Hi there! How can I help?"},
        )
        await client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "user", "content": "What is PyTorch?"},
        )

        return session_id

    @pytest.fixture
    async def empty_session(self, client, mock_session_paths):
        """Create session with no messages."""
        create_response = await client.post(
            "/api/sessions",
            json={"params": {"model": "test-model:7b"}},
        )
        return create_response.json()["session_id"]

    @pytest.mark.asyncio
    async def test_get_session_stats_returns_200(self, client, session_with_messages):
        """Stats endpoint returns 200 for valid session."""
        response = await client.get(f"/api/sessions/{session_with_messages}/stats")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_session_stats_not_found(self, client, mock_session_paths):
        """Stats endpoint returns 404 for non-existent session."""
        response = await client.get("/api/sessions/nonexistent-id/stats")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_session_stats_schema(self, client, session_with_messages):
        """Response matches expected schema."""
        response = await client.get(f"/api/sessions/{session_with_messages}/stats")
        data = response.json()

        # Check all expected fields are present
        assert "history_messages" in data
        assert "history_chars" in data
        assert "history_tokens_estimate" in data
        assert "model_name" in data
        assert "context_length" in data

        # Check types
        assert isinstance(data["history_messages"], int)
        assert isinstance(data["history_chars"], int)
        assert isinstance(data["history_tokens_estimate"], int)
        assert isinstance(data["context_length"], int)
        # model_name can be string or null
        assert data["model_name"] is None or isinstance(data["model_name"], str)

    @pytest.mark.asyncio
    async def test_get_session_stats_empty_session(self, client, empty_session):
        """Stats for session with no messages returns zeros."""
        response = await client.get(f"/api/sessions/{empty_session}/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["history_messages"] == 0
        assert data["history_chars"] == 0
        assert data["history_tokens_estimate"] == 0

    @pytest.mark.asyncio
    async def test_get_session_stats_calculates_chars(
        self, client, session_with_messages
    ):
        """history_chars correctly sums message content lengths."""
        response = await client.get(f"/api/sessions/{session_with_messages}/stats")
        data = response.json()

        # We added 3 messages with content:
        # "Hello, world!" (13 chars)
        # "Hi there! How can I help?" (25 chars)
        # "What is PyTorch?" (16 chars)
        expected_chars = 13 + 25 + 16  # 54 chars
        expected_tokens = expected_chars // 4  # 13 tokens (rough approximation)
        assert data["history_messages"] == 3
        assert data["history_chars"] == expected_chars
        assert data["history_tokens_estimate"] == expected_tokens

    @pytest.mark.asyncio
    async def test_get_session_stats_includes_model_name(
        self, client, session_with_messages
    ):
        """Stats include the model name from session params."""
        response = await client.get(f"/api/sessions/{session_with_messages}/stats")
        data = response.json()

        assert data["model_name"] == "test-model:7b"
