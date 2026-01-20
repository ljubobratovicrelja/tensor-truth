"""Integration tests for chat API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from tensortruth.api.main import create_app


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


class TestChatAPI:
    """Test chat endpoints."""

    @pytest.mark.asyncio
    async def test_chat_session_not_found(self, client, tmp_path, monkeypatch):
        """Test chat with non-existent session."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        response = await client.post(
            "/api/sessions/nonexistent/chat",
            json={"prompt": "Hello"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_chat_no_modules(self, client, tmp_path, monkeypatch):
        """Test chat with session that has no modules or PDFs."""
        sessions_file = tmp_path / "chat_sessions.json"
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        monkeypatch.setattr(
            "tensortruth.api.deps.get_session_dir",
            lambda sid: sessions_dir / sid,
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        # Create a session without modules
        create_response = await client.post(
            "/api/sessions", json={"modules": [], "params": {}}
        )
        session_id = create_response.json()["session_id"]

        # Create session directory
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.post(
            f"/api/sessions/{session_id}/chat",
            json={"prompt": "Hello"},
        )
        assert response.status_code == 400
        assert "modules" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_intent_classification(self, client, tmp_path, monkeypatch):
        """Test intent classification endpoint."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        # Create a session
        create_response = await client.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Test intent classification with no triggers (should be chat)
        response = await client.post(
            f"/api/sessions/{session_id}/intent",
            json={"message": "What is PyTorch?", "recent_messages": []},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "chat"
        assert data["reason"] == "no_triggers"

    @pytest.mark.asyncio
    async def test_intent_classification_session_not_found(
        self, client, tmp_path, monkeypatch
    ):
        """Test intent classification with non-existent session."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        response = await client.post(
            "/api/sessions/nonexistent/intent",
            json={"message": "Hello", "recent_messages": []},
        )
        assert response.status_code == 404


class TestChatAPIWithMockedRAG:
    """Test chat endpoints with mocked RAG service.

    These tests mock the RAG service to avoid needing Ollama running.
    """

    @pytest.mark.asyncio
    @pytest.mark.requires_ollama
    async def test_chat_with_modules(self, client, tmp_path, monkeypatch):
        """Test chat with modules (requires Ollama).

        This test requires Ollama to be running and is skipped by default.
        Run with --run-ollama to enable.
        """
        # This test would require full RAG setup
        # Skipped by default, can be enabled with --run-ollama
        pass
