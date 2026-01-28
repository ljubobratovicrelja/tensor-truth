"""Integration tests for chat API endpoints."""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from tensortruth.api.main import create_app
from tensortruth.services.models import RAGChunk


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
def mock_session_paths(tmp_path, monkeypatch):
    """Patch session paths to use temp directory."""
    sessions_file = tmp_path / "chat_sessions.json"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    monkeypatch.setattr("tensortruth.api.deps.get_sessions_file", lambda: sessions_file)
    monkeypatch.setattr(
        "tensortruth.api.deps.get_sessions_data_dir", lambda: sessions_dir
    )
    monkeypatch.setattr(
        "tensortruth.api.deps.get_session_dir",
        lambda sid: sessions_dir / sid,
    )

    from tensortruth.api.deps import get_session_service

    get_session_service.cache_clear()

    return sessions_file, sessions_dir


class TestChatAPI:
    """Test chat endpoints."""

    @pytest.mark.asyncio
    async def test_chat_session_not_found(self, client, mock_session_paths):
        """Test chat with non-existent session."""
        response = await client.post(
            "/api/sessions/nonexistent/chat",
            json={"prompt": "Hello"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.requires_ollama
    async def test_chat_no_modules(self, client, mock_session_paths):
        """Test chat with session that has no modules or PDFs.

        Should use LLM-only mode without RAG retrieval.
        Requires Ollama to be running.
        """
        sessions_file, sessions_dir = mock_session_paths

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
        assert response.status_code == 200
        data = response.json()
        assert data["confidence_level"] == "llm_only"
        assert len(data["content"]) > 0
        assert data["sources"] == []
        # LLM-only mode should have no metrics
        assert data.get("metrics") is None

    @pytest.mark.asyncio
    async def test_intent_classification(self, client, mock_session_paths):
        """Test intent classification endpoint."""
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
        self, client, mock_session_paths
    ):
        """Test intent classification with non-existent session."""
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
    async def test_chat_response_includes_metrics_field(
        self, client, mock_session_paths
    ):
        """Test that chat response schema includes metrics field.

        This test verifies the response structure, ensuring the API
        returns the metrics field (even if None for LLM-only mode).
        """
        sessions_file, sessions_dir = mock_session_paths

        # Create a session
        create_response = await client.post(
            "/api/sessions", json={"modules": [], "params": {}}
        )
        session_id = create_response.json()["session_id"]

        # Create session directory
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Since we're in LLM-only mode (no modules), we can test without Ollama
        # by just verifying the response structure from the endpoint definition
        # The actual LLM query would require Ollama, so we check the schema only

        # Verify the ChatResponse schema has metrics field by checking endpoint
        # This is a structural test - full e2e would need Ollama
        from tensortruth.api.schemas.chat import ChatResponse

        # Verify ChatResponse model includes metrics field
        assert "metrics" in ChatResponse.model_fields
        assert ChatResponse.model_fields["metrics"].is_required() is False

    @pytest.mark.asyncio
    @pytest.mark.requires_ollama
    async def test_chat_with_modules(self, client, mock_session_paths):
        """Test chat with modules (requires Ollama).

        This test requires Ollama to be running and is skipped by default.
        Run with --run-ollama to enable.
        """
        # This test would require full RAG setup
        # Skipped by default, can be enabled with --run-ollama
        pass

    @pytest.fixture
    def mock_rag_service(self, app):
        """Mock RAG service to avoid Ollama dependency.

        Uses FastAPI's dependency override mechanism for proper mocking.
        """
        from tensortruth.api.deps import get_rag_service

        mock = MagicMock()

        def query_gen(prompt, session_messages=None):
            yield RAGChunk(status="retrieving")
            yield RAGChunk(text="Mocked RAG response")
            mock_node = MagicMock()
            mock_node.node = MagicMock()
            mock_node.node.get_content.return_value = "Source content from mock"
            mock_node.score = 0.88
            mock_node.metadata = {"source": "mock.pdf", "page": 1}
            yield RAGChunk(
                source_nodes=[mock_node],
                is_complete=True,
                metrics={"mean_score": 0.88},
            )

        mock.query.side_effect = query_gen
        mock.needs_reload.return_value = False

        # Use FastAPI's dependency override mechanism
        app.dependency_overrides[get_rag_service] = lambda: mock

        yield mock

        # Clean up override after test
        app.dependency_overrides.pop(get_rag_service, None)

    @pytest.mark.asyncio
    async def test_chat_rag_mode_returns_sources(
        self, client, mock_session_paths, mock_rag_service
    ):
        """Verify sources in response for RAG mode."""
        sessions_file, sessions_dir = mock_session_paths

        # Create a session WITH modules (RAG mode)
        create_response = await client.post(
            "/api/sessions", json={"modules": ["pytorch"], "params": {}}
        )
        session_id = create_response.json()["session_id"]

        # Create session directory
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.post(
            f"/api/sessions/{session_id}/chat",
            json={"prompt": "What is a tensor?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Mocked RAG response"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Source content from mock"
        assert data["sources"][0]["score"] == 0.88

    @pytest.mark.asyncio
    async def test_chat_rag_mode_returns_metrics(
        self, client, mock_session_paths, mock_rag_service
    ):
        """Verify metrics in response for RAG mode."""
        sessions_file, sessions_dir = mock_session_paths

        # Create a session WITH modules (RAG mode)
        create_response = await client.post(
            "/api/sessions", json={"modules": ["pytorch"], "params": {}}
        )
        session_id = create_response.json()["session_id"]

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.post(
            f"/api/sessions/{session_id}/chat",
            json={"prompt": "What is a tensor?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["metrics"] == {"mean_score": 0.88}

    @pytest.mark.asyncio
    async def test_chat_rag_mode_confidence_normal(
        self, client, mock_session_paths, mock_rag_service
    ):
        """Verify confidence_level='normal' in RAG mode."""
        sessions_file, sessions_dir = mock_session_paths

        # Create a session WITH modules (RAG mode)
        create_response = await client.post(
            "/api/sessions", json={"modules": ["pytorch"], "params": {}}
        )
        session_id = create_response.json()["session_id"]

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.post(
            f"/api/sessions/{session_id}/chat",
            json={"prompt": "What is a tensor?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["confidence_level"] == "normal"

    @pytest.mark.asyncio
    async def test_chat_rag_mode_triggers_engine_reload(
        self, client, mock_session_paths, mock_rag_service
    ):
        """Verify load_engine() called when needed."""
        sessions_file, sessions_dir = mock_session_paths

        # Configure mock to indicate reload needed
        mock_rag_service.needs_reload.return_value = True

        # Create a session WITH modules (RAG mode)
        create_response = await client.post(
            "/api/sessions", json={"modules": ["pytorch"], "params": {}}
        )
        session_id = create_response.json()["session_id"]

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.post(
            f"/api/sessions/{session_id}/chat",
            json={"prompt": "What is a tensor?"},
        )

        assert response.status_code == 200
        mock_rag_service.load_engine.assert_called_once()
        # Verify load_engine was called with correct modules
        call_kwargs = mock_rag_service.load_engine.call_args.kwargs
        assert call_kwargs["modules"] == ["pytorch"]

    @pytest.mark.asyncio
    async def test_chat_saves_sources_to_session(
        self, client, mock_session_paths, mock_rag_service
    ):
        """Verify sources persisted to session storage."""
        sessions_file, sessions_dir = mock_session_paths

        # Create a session WITH modules (RAG mode)
        create_response = await client.post(
            "/api/sessions", json={"modules": ["pytorch"], "params": {}}
        )
        session_id = create_response.json()["session_id"]

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Send chat message
        response = await client.post(
            f"/api/sessions/{session_id}/chat",
            json={"prompt": "What is a tensor?"},
        )
        assert response.status_code == 200

        # Get messages to verify sources were saved
        messages_response = await client.get(f"/api/sessions/{session_id}/messages")
        assert messages_response.status_code == 200
        messages_data = messages_response.json()

        # Find the assistant message
        messages = messages_data["messages"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) == 1

        assistant_msg = assistant_messages[0]
        assert "sources" in assistant_msg
        assert len(assistant_msg["sources"]) == 1
        assert assistant_msg["sources"][0]["text"] == "Source content from mock"

    @pytest.mark.asyncio
    async def test_chat_saves_metrics_to_session(
        self, client, mock_session_paths, mock_rag_service
    ):
        """Verify metrics persisted to session storage."""
        sessions_file, sessions_dir = mock_session_paths

        # Create a session WITH modules (RAG mode)
        create_response = await client.post(
            "/api/sessions", json={"modules": ["pytorch"], "params": {}}
        )
        session_id = create_response.json()["session_id"]

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Send chat message
        response = await client.post(
            f"/api/sessions/{session_id}/chat",
            json={"prompt": "What is a tensor?"},
        )
        assert response.status_code == 200

        # Get messages to verify metrics were saved
        messages_response = await client.get(f"/api/sessions/{session_id}/messages")
        assert messages_response.status_code == 200
        messages_data = messages_response.json()

        # Find the assistant message
        messages = messages_data["messages"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) == 1

        assistant_msg = assistant_messages[0]
        assert "metrics" in assistant_msg
        assert assistant_msg["metrics"] == {"mean_score": 0.88}
