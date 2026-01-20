"""Integration tests for PDFs API endpoints."""

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


class TestPDFsAPI:
    """Test PDF management endpoints."""

    @pytest.mark.asyncio
    async def test_list_pdfs_session_not_found(self, client, tmp_path, monkeypatch):
        """Test listing PDFs for non-existent session."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        response = await client.get("/api/sessions/nonexistent/pdfs")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_pdfs_empty(self, client, tmp_path, monkeypatch):
        """Test listing PDFs when none exist."""
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

        # Create a session first
        create_response = await client.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Create the session directory
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.get(f"/api/sessions/{session_id}/pdfs")
        assert response.status_code == 200
        data = response.json()
        assert data["pdfs"] == []
        assert data["has_index"] is False

    @pytest.mark.asyncio
    async def test_upload_pdf_invalid_file(self, client, tmp_path, monkeypatch):
        """Test uploading a non-PDF file."""
        sessions_file = tmp_path / "chat_sessions.json"
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        # Create a session
        create_response = await client.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Try to upload a non-PDF file
        response = await client.post(
            f"/api/sessions/{session_id}/pdfs",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_pdf_not_found(self, client, tmp_path, monkeypatch):
        """Test deleting a non-existent PDF."""
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

        # Create a session
        create_response = await client.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Create session directory
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.delete(f"/api/sessions/{session_id}/pdfs/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_reindex_no_pdfs(self, client, tmp_path, monkeypatch):
        """Test reindexing when no PDFs exist."""
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

        # Create a session
        create_response = await client.post("/api/sessions", json={})
        session_id = create_response.json()["session_id"]

        # Create session directory
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.post(f"/api/sessions/{session_id}/pdfs/reindex")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["pdf_count"] == 0
