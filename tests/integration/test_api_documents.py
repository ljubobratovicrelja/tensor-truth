"""Integration tests for document management API endpoints."""

import asyncio
from unittest.mock import MagicMock, patch

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


@pytest.fixture
def mock_project_paths(tmp_path, monkeypatch):
    """Patch project and session paths to use temp directory."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()

    sessions_file = tmp_path / "chat_sessions.json"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    monkeypatch.setattr(
        "tensortruth.api.deps.get_projects_data_dir", lambda: projects_dir
    )
    monkeypatch.setattr("tensortruth.api.deps.get_sessions_file", lambda: sessions_file)
    monkeypatch.setattr(
        "tensortruth.api.deps.get_sessions_data_dir", lambda: sessions_dir
    )
    monkeypatch.setattr(
        "tensortruth.api.deps.get_session_dir",
        lambda sid: sessions_dir / sid,
    )
    monkeypatch.setattr(
        "tensortruth.api.deps.get_project_dir",
        lambda pid: projects_dir / pid,
    )

    from tensortruth.api.deps import get_project_service, get_session_service

    get_project_service.cache_clear()
    get_session_service.cache_clear()

    return projects_dir, sessions_dir


async def _create_session(client):
    """Helper to create a session and return its ID."""
    response = await client.post("/api/sessions", json={})
    return response.json()["session_id"]


async def _create_project(client, name="Test Project"):
    """Helper to create a project and return its ID."""
    response = await client.post("/api/projects", json={"name": name})
    return response.json()["project_id"]


class TestSessionDocumentsCRUD:
    """Test session document management endpoints."""

    @pytest.mark.asyncio
    async def test_list_documents_session_not_found(self, client, mock_session_paths):
        """Test listing documents for non-existent session."""
        response = await client.get("/api/sessions/nonexistent/documents")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, client, mock_session_paths):
        """Test listing documents when none exist."""
        sessions_file, sessions_dir = mock_session_paths
        session_id = await _create_session(client)

        # Create session directory
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.get(f"/api/sessions/{session_id}/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["has_index"] is False

    @pytest.mark.asyncio
    async def test_upload_pdf_invalid_file(self, client, mock_session_paths):
        """Test uploading a non-PDF file."""
        session_id = await _create_session(client)

        response = await client.post(
            f"/api/sessions/{session_id}/documents/upload",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_text(self, client, mock_session_paths):
        """Test uploading text content to a session."""
        sessions_file, sessions_dir = mock_session_paths
        session_id = await _create_session(client)

        # Create session directory
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.post(
            f"/api/sessions/{session_id}/documents/upload-text",
            json={"content": "Hello world", "filename": "notes.txt"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["doc_id"].startswith("doc_")
        assert data["filename"] == "notes.txt"
        assert data["file_size"] > 0
        assert data["page_count"] == 0

    @pytest.mark.asyncio
    async def test_upload_text_session_not_found(self, client, mock_session_paths):
        """Test uploading text to non-existent session."""
        response = await client.post(
            "/api/sessions/nonexistent/documents/upload-text",
            json={"content": "Hello", "filename": "notes.txt"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    async def test_upload_url(self, mock_fetch, client, mock_session_paths):
        """Test uploading content from a URL to a session."""
        mock_fetch.return_value = ("# Page Title\n\nSome content.", "Page Title")

        sessions_file, sessions_dir = mock_session_paths
        session_id = await _create_session(client)

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.post(
            f"/api/sessions/{session_id}/documents/upload-url",
            json={"url": "https://example.com/docs", "context": "Focus on API"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["doc_id"].startswith("url_")
        assert data["filename"] == "Page Title"
        assert data["file_size"] > 0

    @pytest.mark.asyncio
    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    async def test_upload_url_error(self, mock_fetch, client, mock_session_paths):
        """Test URL upload with fetch error."""
        mock_fetch.side_effect = ConnectionError("HTTP 404")

        session_id = await _create_session(client)

        response = await client.post(
            f"/api/sessions/{session_id}/documents/upload-url",
            json={"url": "https://example.com/missing"},
        )
        assert response.status_code == 400
        assert "404" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, client, mock_session_paths):
        """Test deleting a non-existent document."""
        sessions_file, sessions_dir = mock_session_paths
        session_id = await _create_session(client)

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.delete(
            f"/api/sessions/{session_id}/documents/nonexistent"
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_text_document(self, client, mock_session_paths):
        """Test uploading then deleting a text document."""
        sessions_file, sessions_dir = mock_session_paths
        session_id = await _create_session(client)

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Upload
        upload_response = await client.post(
            f"/api/sessions/{session_id}/documents/upload-text",
            json={"content": "Hello world", "filename": "notes.txt"},
        )
        doc_id = upload_response.json()["doc_id"]

        # Verify it exists
        list_response = await client.get(f"/api/sessions/{session_id}/documents")
        assert len(list_response.json()["documents"]) == 1

        # Delete
        delete_response = await client.delete(
            f"/api/sessions/{session_id}/documents/{doc_id}"
        )
        assert delete_response.status_code == 204

    @pytest.mark.asyncio
    async def test_reindex_no_documents(self, client, mock_session_paths):
        """Test reindexing when no documents exist."""
        sessions_file, sessions_dir = mock_session_paths
        session_id = await _create_session(client)

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.post(f"/api/sessions/{session_id}/documents/reindex")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["pdf_count"] == 0


class TestProjectDocumentsCRUD:
    """Test project document management endpoints."""

    @pytest.mark.asyncio
    async def test_list_documents_project_not_found(self, client, mock_project_paths):
        """Test listing documents for non-existent project."""
        response = await client.get("/api/projects/nonexistent/documents")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, client, mock_project_paths):
        """Test listing documents when none exist in project."""
        project_id = await _create_project(client)

        response = await client.get(f"/api/projects/{project_id}/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["has_index"] is False

    @pytest.mark.asyncio
    async def test_upload_pdf_invalid_file(self, client, mock_project_paths):
        """Test uploading a non-PDF file to project."""
        project_id = await _create_project(client)

        response = await client.post(
            f"/api/projects/{project_id}/documents/upload",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_text(self, client, mock_project_paths):
        """Test uploading text content to a project."""
        project_id = await _create_project(client)

        response = await client.post(
            f"/api/projects/{project_id}/documents/upload-text",
            json={"content": "Project documentation", "filename": "readme.md"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["doc_id"].startswith("doc_")
        assert data["filename"] == "readme.md"
        assert data["file_size"] > 0

    @pytest.mark.asyncio
    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    async def test_upload_url(self, mock_fetch, client, mock_project_paths):
        """Test uploading content from a URL to a project."""
        mock_fetch.return_value = ("# Docs\n\nContent here.", "API Docs")

        project_id = await _create_project(client)

        response = await client.post(
            f"/api/projects/{project_id}/documents/upload-url",
            json={"url": "https://example.com/api-docs"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["doc_id"].startswith("url_")
        assert data["filename"] == "API Docs"

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, client, mock_project_paths):
        """Test deleting a non-existent document from project."""
        project_id = await _create_project(client)

        response = await client.delete(
            f"/api/projects/{project_id}/documents/nonexistent"
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_upload_and_delete_text(self, client, mock_project_paths):
        """Test uploading then deleting a text document from project."""
        project_id = await _create_project(client)

        # Upload
        upload_response = await client.post(
            f"/api/projects/{project_id}/documents/upload-text",
            json={"content": "Some content", "filename": "doc.txt"},
        )
        doc_id = upload_response.json()["doc_id"]

        # Verify it exists
        list_response = await client.get(f"/api/projects/{project_id}/documents")
        assert len(list_response.json()["documents"]) == 1

        # Delete
        delete_response = await client.delete(
            f"/api/projects/{project_id}/documents/{doc_id}"
        )
        assert delete_response.status_code == 204

    @pytest.mark.asyncio
    async def test_reindex_no_documents(self, client, mock_project_paths):
        """Test reindexing when no documents exist in project."""
        project_id = await _create_project(client)

        response = await client.post(f"/api/projects/{project_id}/documents/reindex")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["pdf_count"] == 0

    @pytest.mark.asyncio
    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    async def test_upload_url_error(self, mock_fetch, client, mock_project_paths):
        """Test URL upload with fetch error for project."""
        mock_fetch.side_effect = ConnectionError("HTTP 500")

        project_id = await _create_project(client)

        response = await client.post(
            f"/api/projects/{project_id}/documents/upload-url",
            json={"url": "https://example.com/broken"},
        )
        assert response.status_code == 400
        assert "500" in response.json()["detail"]


class TestCatalogModuleAdd:
    """Test catalog module add endpoint."""

    @pytest.fixture
    def mock_indexes(self, tmp_path, monkeypatch):
        """Create a fake indexes dir with a built module."""
        indexes_dir = tmp_path / "indexes"
        model_dir = indexes_dir / "bge-m3"
        # Create a built module "pytorch"
        pytorch_dir = model_dir / "pytorch"
        pytorch_dir.mkdir(parents=True)
        (pytorch_dir / "chroma.sqlite3").touch()

        monkeypatch.setattr(
            "tensortruth.api.routes.documents.get_indexes_dir", lambda: indexes_dir
        )
        monkeypatch.setattr(
            "tensortruth.api.routes.documents.sanitize_model_id",
            lambda _model: "bge-m3",
        )
        return indexes_dir

    @pytest.mark.asyncio
    async def test_add_module_project_not_found(
        self, client, mock_project_paths, mock_indexes
    ):
        """Test adding module to non-existent project."""
        response = await client.post(
            "/api/projects/nonexistent/catalog-modules",
            json={"module_name": "pytorch"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_add_module_not_found(self, client, mock_project_paths, mock_indexes):
        """Test adding a module that has no built index."""
        project_id = await _create_project(client)

        response = await client.post(
            f"/api/projects/{project_id}/catalog-modules",
            json={"module_name": "nonexistent_module"},
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_add_module_success(self, client, mock_project_paths, mock_indexes):
        """Test successfully adding a catalog module that exists on disk."""
        project_id = await _create_project(client)

        response = await client.post(
            f"/api/projects/{project_id}/catalog-modules",
            json={"module_name": "pytorch"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["module_name"] == "pytorch"
        assert data["status"] == "indexed"

        # Verify project was updated
        project_response = await client.get(f"/api/projects/{project_id}")
        catalog = project_response.json()["catalog_modules"]
        assert "pytorch" in catalog
        assert catalog["pytorch"]["status"] == "indexed"

    @pytest.mark.asyncio
    async def test_add_module_already_indexed(
        self, client, mock_project_paths, mock_indexes
    ):
        """Test adding a module that is already indexed in the project."""
        project_id = await _create_project(client)

        # First add should succeed
        response1 = await client.post(
            f"/api/projects/{project_id}/catalog-modules",
            json={"module_name": "pytorch"},
        )
        assert response1.status_code == 201

        # Second add should conflict
        response2 = await client.post(
            f"/api/projects/{project_id}/catalog-modules",
            json={"module_name": "pytorch"},
        )
        assert response2.status_code == 409
        assert "already indexed" in response2.json()["detail"]


class TestCatalogModuleRemove:
    """Test catalog module remove endpoint."""

    @pytest.mark.asyncio
    async def test_remove_module_project_not_found(self, client, mock_project_paths):
        """Test removing module from non-existent project."""
        response = await client.delete(
            "/api/projects/nonexistent/catalog-modules/pytorch"
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_remove_module_not_in_project(self, client, mock_project_paths):
        """Test removing a module that isn't in the project."""
        project_id = await _create_project(client)

        response = await client.delete(
            f"/api/projects/{project_id}/catalog-modules/pytorch"
        )
        assert response.status_code == 404
        assert "not found in project" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_remove_module_while_building(self, client, mock_project_paths):
        """Test removing a module that is currently being built."""
        from tensortruth.api.deps import get_project_service

        project_id = await _create_project(client)

        # Manually set a module as building
        project_service = get_project_service()
        data = project_service.load()
        data.projects[project_id]["catalog_modules"] = {
            "pytorch": {"status": "building", "task_id": "fake-task"}
        }
        project_service.save(data)

        # Try to remove while building
        response = await client.delete(
            f"/api/projects/{project_id}/catalog-modules/pytorch"
        )
        assert response.status_code == 409
        assert "being built" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_remove_indexed_module(self, client, mock_project_paths):
        """Test removing a module with 'indexed' status."""
        project_id = await _create_project(client)

        # Manually set a module as indexed by updating project
        from tensortruth.api.deps import get_project_service

        project_service = get_project_service()
        data = project_service.load()
        data.projects[project_id]["catalog_modules"] = {
            "pytorch": {"status": "indexed"}
        }
        project_service.save(data)

        # Remove it
        response = await client.delete(
            f"/api/projects/{project_id}/catalog-modules/pytorch"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["module_name"] == "pytorch"
        assert data["status"] == "removed"

        # Verify it's gone from project
        project_response = await client.get(f"/api/projects/{project_id}")
        assert "pytorch" not in project_response.json()["catalog_modules"]
