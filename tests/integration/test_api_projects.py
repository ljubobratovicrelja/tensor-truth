"""Integration tests for projects API endpoints."""

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

    from tensortruth.api.deps import get_project_service, get_session_service

    get_project_service.cache_clear()
    get_session_service.cache_clear()

    return projects_dir, sessions_dir


class TestProjectsCRUD:
    """Test project CRUD endpoints."""

    @pytest.mark.asyncio
    async def test_list_projects_empty(self, client, mock_project_paths):
        """Test listing projects when none exist."""
        response = await client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert data["projects"] == []

    @pytest.mark.asyncio
    async def test_create_project(self, client, mock_project_paths):
        """Test creating a new project."""
        response = await client.post(
            "/api/projects",
            json={"name": "My Project", "description": "A test project"},
        )
        assert response.status_code == 201
        data = response.json()
        assert "project_id" in data
        assert data["name"] == "My Project"
        assert data["description"] == "A test project"
        assert data["catalog_modules"] == {}
        assert data["documents"] == []
        assert data["session_ids"] == []
        assert data["config"] == {}

    @pytest.mark.asyncio
    async def test_create_project_minimal(self, client, mock_project_paths):
        """Test creating a project with just a name."""
        response = await client.post(
            "/api/projects",
            json={"name": "Minimal"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal"
        assert data["description"] == ""

    @pytest.mark.asyncio
    async def test_get_project(self, client, mock_project_paths):
        """Test getting a project by ID."""
        create_response = await client.post("/api/projects", json={"name": "Test"})
        project_id = create_response.json()["project_id"]

        response = await client.get(f"/api/projects/{project_id}")
        assert response.status_code == 200
        assert response.json()["project_id"] == project_id
        assert response.json()["name"] == "Test"

    @pytest.mark.asyncio
    async def test_get_project_not_found(self, client, mock_project_paths):
        """Test getting a non-existent project."""
        response = await client.get("/api/projects/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_project(self, client, mock_project_paths):
        """Test updating a project."""
        create_response = await client.post("/api/projects", json={"name": "Original"})
        project_id = create_response.json()["project_id"]

        response = await client.patch(
            f"/api/projects/{project_id}",
            json={"name": "Updated", "description": "New desc"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Updated"
        assert response.json()["description"] == "New desc"

    @pytest.mark.asyncio
    async def test_update_project_config(self, client, mock_project_paths):
        """Test updating project config."""
        create_response = await client.post(
            "/api/projects", json={"name": "Configurable"}
        )
        project_id = create_response.json()["project_id"]

        response = await client.patch(
            f"/api/projects/{project_id}",
            json={"config": {"temperature": 0.3, "model": "custom-model"}},
        )
        assert response.status_code == 200
        assert response.json()["config"]["temperature"] == 0.3
        assert response.json()["config"]["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_update_project_not_found(self, client, mock_project_paths):
        """Test updating a non-existent project."""
        response = await client.patch(
            "/api/projects/nonexistent-id",
            json={"name": "Updated"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_project(self, client, mock_project_paths):
        """Test deleting a project."""
        create_response = await client.post("/api/projects", json={"name": "To Delete"})
        project_id = create_response.json()["project_id"]

        response = await client.delete(f"/api/projects/{project_id}")
        assert response.status_code == 204

        # Verify it's gone
        response = await client.get(f"/api/projects/{project_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_project_not_found(self, client, mock_project_paths):
        """Test deleting a non-existent project."""
        response = await client.delete("/api/projects/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_projects_sorted_by_updated_at(self, client, mock_project_paths):
        """Test that projects are sorted by updated_at descending."""
        await client.post("/api/projects", json={"name": "First"})
        await client.post("/api/projects", json={"name": "Second"})
        await client.post("/api/projects", json={"name": "Third"})

        response = await client.get("/api/projects")
        projects = response.json()["projects"]
        assert len(projects) == 3
        # Most recently created should be first
        assert projects[0]["name"] == "Third"


class TestProjectSessions:
    """Test project session endpoints."""

    @pytest.mark.asyncio
    async def test_create_project_session(self, client, mock_project_paths):
        """Test creating a session within a project."""
        # Create project
        project_response = await client.post(
            "/api/projects", json={"name": "Test Project"}
        )
        project_id = project_response.json()["project_id"]

        # Create session in project
        response = await client.post(
            f"/api/projects/{project_id}/sessions",
            json={"modules": ["pytorch"], "params": {"model": "test-model"}},
        )
        assert response.status_code == 201
        session_data = response.json()
        assert "session_id" in session_data
        assert session_data["project_id"] == project_id

        # Verify session is in project's session_ids
        project_response = await client.get(f"/api/projects/{project_id}")
        assert session_data["session_id"] in project_response.json()["session_ids"]

    @pytest.mark.asyncio
    async def test_create_project_session_not_found(self, client, mock_project_paths):
        """Test creating a session in a non-existent project."""
        response = await client.post(
            "/api/projects/nonexistent-id/sessions",
            json={},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_project_sessions(self, client, mock_project_paths):
        """Test listing sessions belonging to a project."""
        # Create project
        project_response = await client.post("/api/projects", json={"name": "Test"})
        project_id = project_response.json()["project_id"]

        # Create two sessions and add messages so they aren't filtered as empty
        s1 = await client.post(f"/api/projects/{project_id}/sessions", json={})
        s1_id = s1.json()["session_id"]
        await client.post(
            f"/api/sessions/{s1_id}/messages",
            json={"role": "user", "content": "hi"},
        )

        s2 = await client.post(f"/api/projects/{project_id}/sessions", json={})
        s2_id = s2.json()["session_id"]
        await client.post(
            f"/api/sessions/{s2_id}/messages",
            json={"role": "user", "content": "hello"},
        )

        # List sessions
        response = await client.get(f"/api/projects/{project_id}/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_project_sessions_not_found(self, client, mock_project_paths):
        """Test listing sessions for a non-existent project."""
        response = await client.get("/api/projects/nonexistent-id/sessions")
        assert response.status_code == 404


class TestCascadeDelete:
    """Test cascade delete of project and its sessions."""

    @pytest.mark.asyncio
    async def test_delete_project_cascades_sessions(self, client, mock_project_paths):
        """Deleting a project also deletes its sessions."""
        # Create project with sessions
        project_response = await client.post(
            "/api/projects", json={"name": "Cascade Test"}
        )
        project_id = project_response.json()["project_id"]

        session1 = await client.post(f"/api/projects/{project_id}/sessions", json={})
        session1_id = session1.json()["session_id"]

        session2 = await client.post(f"/api/projects/{project_id}/sessions", json={})
        session2_id = session2.json()["session_id"]

        # Delete project
        response = await client.delete(f"/api/projects/{project_id}")
        assert response.status_code == 204

        # Verify sessions are gone
        assert (await client.get(f"/api/sessions/{session1_id}")).status_code == 404
        assert (await client.get(f"/api/sessions/{session2_id}")).status_code == 404


class TestSessionDeleteBookkeeping:
    """Test that deleting a session removes it from its project."""

    @pytest.mark.asyncio
    async def test_session_delete_removes_from_project(
        self, client, mock_project_paths
    ):
        """Deleting a session via /api/sessions removes it from project's session_ids."""
        # Create project with a session
        project_response = await client.post(
            "/api/projects", json={"name": "Bookkeeping Test"}
        )
        project_id = project_response.json()["project_id"]

        session_response = await client.post(
            f"/api/projects/{project_id}/sessions", json={}
        )
        session_id = session_response.json()["session_id"]

        # Delete session via session endpoint (not project endpoint)
        response = await client.delete(f"/api/sessions/{session_id}")
        assert response.status_code == 204

        # Verify session removed from project's session_ids
        project_response = await client.get(f"/api/projects/{project_id}")
        assert session_id not in project_response.json()["session_ids"]


class TestConfigInheritanceAPI:
    """Test config inheritance through the API."""

    @pytest.mark.asyncio
    async def test_project_config_applied_to_session(self, client, mock_project_paths):
        """Project config is inherited by sessions created within it."""
        # Create project with config
        project_response = await client.post(
            "/api/projects", json={"name": "Config Test"}
        )
        project_id = project_response.json()["project_id"]

        # Set project config
        await client.patch(
            f"/api/projects/{project_id}",
            json={"config": {"temperature": 0.2}},
        )

        # Create session in project
        session_response = await client.post(
            f"/api/projects/{project_id}/sessions",
            json={"params": {}},
        )
        session_data = session_response.json()

        # Session should have project's temperature
        assert session_data["params"]["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_user_params_override_project_config(
        self, client, mock_project_paths
    ):
        """User params override project config when creating session."""
        # Create project with config
        project_response = await client.post(
            "/api/projects", json={"name": "Override Test"}
        )
        project_id = project_response.json()["project_id"]

        await client.patch(
            f"/api/projects/{project_id}",
            json={"config": {"temperature": 0.2}},
        )

        # Create session with user override
        session_response = await client.post(
            f"/api/projects/{project_id}/sessions",
            json={"params": {"temperature": 0.9}},
        )
        session_data = session_response.json()

        # User param should win
        assert session_data["params"]["temperature"] == 0.9
