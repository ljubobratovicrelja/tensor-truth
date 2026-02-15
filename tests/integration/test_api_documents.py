"""Integration tests for document management API endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests
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
        assert data["index_updated_at"] is None

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
    async def test_list_documents_returns_short_doc_id_and_timestamps(
        self, client, mock_session_paths
    ):
        """Test that listed documents have short doc_ids and uploaded_at."""
        sessions_file, sessions_dir = mock_session_paths
        session_id = await _create_session(client)

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Upload a text doc
        await client.post(
            f"/api/sessions/{session_id}/documents/upload-text",
            json={"content": "Hello world", "filename": "notes.txt"},
        )

        # List documents
        list_response = await client.get(f"/api/sessions/{session_id}/documents")
        data = list_response.json()
        assert len(data["documents"]) == 1

        doc = data["documents"][0]
        # doc_id should be the short form (e.g. "doc_abcdef12"), not the full stem
        assert doc["doc_id"].startswith("doc_")
        assert "_" not in doc["doc_id"][4:]  # no underscore after the hash
        assert doc["uploaded_at"] is not None
        assert data["index_updated_at"] is None  # no index yet

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

        # Delete — the short doc_id from upload should work
        delete_response = await client.delete(
            f"/api/sessions/{session_id}/documents/{doc_id}"
        )
        assert delete_response.status_code == 204

        # Verify it's gone
        list_response = await client.get(f"/api/sessions/{session_id}/documents")
        assert len(list_response.json()["documents"]) == 0

    @pytest.mark.asyncio
    async def test_reindex_no_documents(self, client, mock_session_paths):
        """Test reindexing when no documents exist."""
        sessions_file, sessions_dir = mock_session_paths
        session_id = await _create_session(client)

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        response = await client.post(f"/api/sessions/{session_id}/pdfs/reindex")
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

        # Verify it's gone
        list_response = await client.get(f"/api/projects/{project_id}/documents")
        assert len(list_response.json()["documents"]) == 0

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


class TestFileUrlEndpoints:
    """Test file URL probe and upload endpoints."""

    @pytest.mark.asyncio
    @patch("tensortruth.api.routes.documents.requests.head")
    async def test_probe_file_url_pdf(self, mock_head, client, mock_session_paths):
        """Test probing a PDF file URL returns supported=True."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {
            "Content-Type": "application/pdf",
            "Content-Length": "2100000",
        }
        mock_head.return_value = mock_resp

        response = await client.get(
            "/api/file-url-info",
            params={"url": "https://example.com/paper.pdf"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "paper.pdf"
        assert data["content_type"] == "application/pdf"
        assert data["file_size"] == 2100000
        assert data["supported"] is True

    @pytest.mark.asyncio
    @patch("tensortruth.api.routes.documents.requests.head")
    async def test_probe_file_url_unsupported(
        self, mock_head, client, mock_session_paths
    ):
        """Test probing an unsupported file type returns supported=False."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {
            "Content-Type": "image/png",
            "Content-Length": "50000",
        }
        mock_head.return_value = mock_resp

        response = await client.get(
            "/api/file-url-info",
            params={"url": "https://example.com/image.png"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["supported"] is False
        assert data["content_type"] == "image/png"

    @pytest.mark.asyncio
    @patch("tensortruth.api.routes.documents.requests.get")
    @patch("tensortruth.api.routes.documents.requests.head")
    async def test_probe_file_url_unreachable(
        self, mock_head, mock_get, client, mock_session_paths
    ):
        """Test probing an unreachable URL returns 400."""
        mock_head.side_effect = requests.ConnectionError("Connection refused")
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        response = await client.get(
            "/api/file-url-info",
            params={"url": "https://unreachable.example.com/file.pdf"},
        )
        assert response.status_code == 400
        assert "Could not reach" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_probe_file_url_invalid(self, client, mock_session_paths):
        """Test probing an invalid URL returns 400."""
        response = await client.get(
            "/api/file-url-info",
            params={"url": "not-a-url"},
        )
        assert response.status_code == 400
        assert "Invalid URL" in response.json()["detail"]

    @pytest.mark.asyncio
    @patch("tensortruth.api.routes.documents.requests.get")
    async def test_upload_file_url_session(self, mock_get, client, mock_session_paths):
        """Test downloading a PDF from URL and uploading to session."""
        sessions_file, sessions_dir = mock_session_paths
        session_id = await _create_session(client)

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"%PDF-1.4 fake pdf content"
        mock_resp.headers = {
            "Content-Type": "application/pdf",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        response = await client.post(
            f"/api/sessions/{session_id}/documents/upload-file-url",
            json={"url": "https://example.com/paper.pdf"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["doc_id"].startswith("pdf_")
        assert "paper" in data["filename"].lower()

    @pytest.mark.asyncio
    @patch("tensortruth.api.routes.documents.requests.get")
    async def test_upload_file_url_project(self, mock_get, client, mock_project_paths):
        """Test downloading a PDF from URL and uploading to project."""
        project_id = await _create_project(client)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"%PDF-1.4 fake pdf content"
        mock_resp.headers = {
            "Content-Type": "application/pdf",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        response = await client.post(
            f"/api/projects/{project_id}/documents/upload-file-url",
            json={"url": "https://example.com/doc.pdf"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["doc_id"].startswith("pdf_")

    @pytest.mark.asyncio
    @patch("tensortruth.api.routes.documents.requests.get")
    async def test_upload_file_url_unsupported(
        self, mock_get, client, mock_session_paths
    ):
        """Test uploading an unsupported file type returns 400."""
        session_id = await _create_session(client)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\x89PNG fake image"
        mock_resp.headers = {
            "Content-Type": "image/png",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        response = await client.post(
            f"/api/sessions/{session_id}/documents/upload-file-url",
            json={"url": "https://example.com/image.png"},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]


def _make_mock_paper(arxiv_id="2301.12345", title="Attention Is All You Need"):
    """Create a mock arxiv paper result."""
    paper = MagicMock()
    paper.title = title
    paper.authors = [MagicMock(name="Author One"), MagicMock(name="Author Two")]
    # Make author .name return a string (MagicMock's name kwarg is special)
    paper.authors[0].name = "Author One"
    paper.authors[1].name = "Author Two"
    paper.published = datetime(2023, 1, 15, tzinfo=timezone.utc)
    paper.categories = ["cs.CL", "cs.AI"]
    paper.summary = "This is the abstract."
    paper.pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    return paper


class TestArxivEndpoints:
    """Test arXiv lookup and upload endpoints."""

    @pytest.mark.asyncio
    @patch("tensortruth.api.routes.arxiv.arxiv.Search")
    async def test_lookup_valid_id(self, mock_search, client, mock_session_paths):
        """Test arXiv metadata lookup with a valid ID."""
        paper = _make_mock_paper()
        mock_search.return_value.results.return_value = iter([paper])

        response = await client.get("/api/arxiv/2301.12345")
        assert response.status_code == 200
        data = response.json()
        assert data["arxiv_id"] == "2301.12345"
        assert data["title"] == "Attention Is All You Need"
        assert data["authors"] == ["Author One", "Author Two"]
        assert data["published"] == "2023-01-15"
        assert data["categories"] == ["cs.CL", "cs.AI"]
        assert data["abstract"] == "This is the abstract."
        assert "arxiv.org/pdf" in data["pdf_url"]

    @pytest.mark.asyncio
    async def test_lookup_invalid_id(self, client, mock_session_paths):
        """Test arXiv lookup with an invalid ID returns 400."""
        response = await client.get("/api/arxiv/not-a-valid-id")
        assert response.status_code == 400
        assert "Invalid" in response.json()["detail"]

    @pytest.mark.asyncio
    @patch("tensortruth.api.routes.arxiv.arxiv.Search")
    async def test_lookup_not_found(self, mock_search, client, mock_session_paths):
        """Test arXiv lookup when paper doesn't exist returns 404."""
        mock_search.return_value.results.return_value = iter([])

        response = await client.get("/api/arxiv/2301.99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    @patch("tensortruth.api.routes.documents.arxiv.Search")
    async def test_upload_arxiv_session(self, mock_search, client, mock_session_paths):
        """Test uploading an arXiv paper to a session."""
        sessions_file, sessions_dir = mock_session_paths
        session_id = await _create_session(client)

        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        paper = _make_mock_paper()

        # Make download_pdf write a minimal PDF
        def fake_download(dirpath, filename):
            pdf_bytes = b"%PDF-1.4 fake pdf content"
            (
                dirpath / filename
                if hasattr(dirpath, "__truediv__")
                else __import__("pathlib").Path(dirpath) / filename
            ).write_bytes(pdf_bytes)

        paper.download_pdf = fake_download
        mock_search.return_value.results.return_value = iter([paper])

        response = await client.post(
            f"/api/sessions/{session_id}/documents/upload-arxiv",
            json={"arxiv_id": "2301.12345"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["doc_id"].startswith("pdf_")
        assert "Attention Is All You Need" in data["filename"]

    @pytest.mark.asyncio
    @patch("tensortruth.api.routes.documents.arxiv.Search")
    async def test_upload_arxiv_project(self, mock_search, client, mock_project_paths):
        """Test uploading an arXiv paper to a project."""
        project_id = await _create_project(client)

        paper = _make_mock_paper()

        def fake_download(dirpath, filename):
            pdf_bytes = b"%PDF-1.4 fake pdf content"
            (
                dirpath / filename
                if hasattr(dirpath, "__truediv__")
                else __import__("pathlib").Path(dirpath) / filename
            ).write_bytes(pdf_bytes)

        paper.download_pdf = fake_download
        mock_search.return_value.results.return_value = iter([paper])

        response = await client.post(
            f"/api/projects/{project_id}/documents/upload-arxiv",
            json={"arxiv_id": "2301.12345"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["doc_id"].startswith("pdf_")
        assert "Attention Is All You Need" in data["filename"]
