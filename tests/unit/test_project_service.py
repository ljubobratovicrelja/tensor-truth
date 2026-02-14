"""Unit tests for ProjectService."""

import json
from pathlib import Path

import pytest

from tensortruth.services.models import ProjectData
from tensortruth.services.project_service import ProjectService


@pytest.fixture
def temp_projects_dir(tmp_path: Path) -> Path:
    """Create a temporary projects directory."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    return projects_dir


@pytest.fixture
def project_service(temp_projects_dir: Path) -> ProjectService:
    """Create a ProjectService instance with temp paths."""
    return ProjectService(projects_dir=temp_projects_dir)


@pytest.fixture
def sample_project_data() -> ProjectData:
    """Create sample project data for testing."""
    return ProjectData(
        projects={
            "project-1": {
                "project_id": "project-1",
                "name": "Test Project",
                "description": "A test project",
                "created_at": "2024-01-01 12:00:00",
                "updated_at": "2024-01-01 12:00:00",
                "catalog_modules": {},
                "documents": [],
                "session_ids": ["session-a", "session-b"],
                "config": {"temperature": 0.5},
            },
            "project-2": {
                "project_id": "project-2",
                "name": "Another Project",
                "description": "",
                "created_at": "2024-01-02 12:00:00",
                "updated_at": "2024-01-02 12:00:00",
                "catalog_modules": {},
                "documents": [],
                "session_ids": [],
                "config": {},
            },
        }
    )


class TestProjectServiceLoad:
    """Tests for ProjectService.load()."""

    def test_load_empty_returns_empty_data(self, project_service: ProjectService):
        """Load returns empty data when no projects exist."""
        data = project_service.load()

        assert data.projects == {}

    def test_load_from_per_project_files(
        self, project_service: ProjectService, temp_projects_dir: Path
    ):
        """Load returns data from per-project files."""
        project_id = "test-project"
        project_dir = temp_projects_dir / project_id
        project_dir.mkdir()
        project_data = {
            "project_id": project_id,
            "name": "Test Project",
            "description": "desc",
            "created_at": "2024-01-01 12:00:00",
            "updated_at": "2024-01-01 12:00:00",
            "catalog_modules": {},
            "documents": [],
            "session_ids": [],
            "config": {},
        }
        (project_dir / "project.json").write_text(json.dumps(project_data))

        # Create index file
        index = {
            "projects": {
                project_id: {
                    "name": "Test Project",
                    "updated_at": "2024-01-01 12:00:00",
                }
            }
        }
        (temp_projects_dir / "projects_index.json").write_text(json.dumps(index))

        data = project_service.load()

        assert project_id in data.projects
        assert data.projects[project_id]["name"] == "Test Project"

    def test_load_removes_stale_index_entries(
        self, project_service: ProjectService, temp_projects_dir: Path
    ):
        """Load removes index entries for missing project files."""
        index = {
            "projects": {
                "missing-project": {
                    "name": "Missing",
                    "updated_at": "2024-01-01 12:00:00",
                }
            }
        }
        (temp_projects_dir / "projects_index.json").write_text(json.dumps(index))

        data = project_service.load()

        assert data.projects == {}

        # Index should be updated
        updated_index = json.loads(
            (temp_projects_dir / "projects_index.json").read_text()
        )
        assert updated_index["projects"] == {}


class TestProjectServiceSave:
    """Tests for ProjectService.save()."""

    def test_save_creates_per_project_files(
        self, project_service: ProjectService, temp_projects_dir: Path
    ):
        """Save creates individual project files."""
        data = ProjectData(
            projects={
                "new-id": {
                    "project_id": "new-id",
                    "name": "Test Project",
                    "description": "",
                    "created_at": "2024-01-01 12:00:00",
                    "updated_at": "2024-01-01 12:00:00",
                    "catalog_modules": {},
                    "documents": [],
                    "session_ids": [],
                    "config": {},
                }
            }
        )

        project_service.save(data)

        # Verify project file created
        project_file = temp_projects_dir / "new-id" / "project.json"
        assert project_file.exists()
        saved_project = json.loads(project_file.read_text())
        assert saved_project["name"] == "Test Project"

        # Verify index created
        index_file = temp_projects_dir / "projects_index.json"
        assert index_file.exists()
        index = json.loads(index_file.read_text())
        assert "new-id" in index["projects"]

    def test_save_deletes_removed_projects(
        self, project_service: ProjectService, temp_projects_dir: Path
    ):
        """Save deletes project directories for removed projects."""
        # Create existing project
        project_dir = temp_projects_dir / "to-delete"
        project_dir.mkdir()
        (project_dir / "project.json").write_text('{"name": "To Delete"}')

        # Create index with the project
        index = {
            "projects": {"to-delete": {"name": "To Delete", "updated_at": "2024-01-01"}}
        }
        (temp_projects_dir / "projects_index.json").write_text(json.dumps(index))

        # Save with empty data (project removed)
        project_service.save(ProjectData(projects={}))

        # Project directory should be deleted
        assert not project_dir.exists()


class TestProjectServiceCreate:
    """Tests for ProjectService.create()."""

    def test_create_new_project(self, project_service: ProjectService):
        """Create generates new project with correct properties."""
        data = ProjectData(projects={})

        new_id, new_data = project_service.create(
            name="My Project",
            description="A description",
            data=data,
        )

        assert new_id is not None
        assert new_id in new_data.projects

        project = new_data.projects[new_id]
        assert project["name"] == "My Project"
        assert project["description"] == "A description"
        assert project["project_id"] == new_id
        assert project["catalog_modules"] == {}
        assert project["documents"] == []
        assert project["session_ids"] == []
        assert project["config"] == {}
        assert "created_at" in project
        assert "updated_at" in project

    def test_create_preserves_existing_projects(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Create preserves existing projects in the data."""
        new_id, new_data = project_service.create(
            name="New", description="", data=sample_project_data
        )

        assert "project-1" in new_data.projects
        assert "project-2" in new_data.projects
        assert new_id in new_data.projects


class TestProjectServiceUpdate:
    """Tests for ProjectService.update()."""

    def test_update_name(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Update changes the project name."""
        new_data = project_service.update(
            "project-1", sample_project_data, name="Updated Name"
        )

        assert new_data.projects["project-1"]["name"] == "Updated Name"

    def test_update_description(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Update changes the project description."""
        new_data = project_service.update(
            "project-1", sample_project_data, description="New desc"
        )

        assert new_data.projects["project-1"]["description"] == "New desc"

    def test_update_config(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Update changes the project config."""
        new_data = project_service.update(
            "project-1", sample_project_data, config={"model": "llama3"}
        )

        assert new_data.projects["project-1"]["config"] == {"model": "llama3"}

    def test_update_bumps_updated_at(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Update bumps the updated_at timestamp."""
        old_updated_at = sample_project_data.projects["project-1"]["updated_at"]

        new_data = project_service.update(
            "project-1", sample_project_data, name="New Name"
        )

        assert new_data.projects["project-1"]["updated_at"] != old_updated_at

    def test_update_nonexistent_returns_unchanged(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Update returns unchanged data for nonexistent project."""
        new_data = project_service.update(
            "nonexistent", sample_project_data, name="New Name"
        )

        assert new_data == sample_project_data

    def test_update_partial_preserves_other_fields(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Partial update preserves fields not being updated."""
        new_data = project_service.update(
            "project-1", sample_project_data, name="Updated"
        )

        assert new_data.projects["project-1"]["description"] == "A test project"
        assert new_data.projects["project-1"]["config"] == {"temperature": 0.5}


class TestProjectServiceDelete:
    """Tests for ProjectService.delete()."""

    def test_delete_project(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Delete removes project from data."""
        new_data = project_service.delete("project-2", sample_project_data)

        assert "project-2" not in new_data.projects
        assert "project-1" in new_data.projects

    def test_delete_nonexistent_returns_unchanged(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Delete returns unchanged data for nonexistent project."""
        new_data = project_service.delete("nonexistent", sample_project_data)

        assert new_data == sample_project_data


class TestProjectServiceSessionManagement:
    """Tests for session management methods."""

    def test_add_session(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Add session appends to project session_ids."""
        new_data = project_service.add_session(
            "project-1", "session-c", sample_project_data
        )

        assert "session-c" in new_data.projects["project-1"]["session_ids"]
        assert len(new_data.projects["project-1"]["session_ids"]) == 3

    def test_add_session_idempotent(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Add session is idempotent - doesn't duplicate existing session_id."""
        new_data = project_service.add_session(
            "project-1", "session-a", sample_project_data
        )

        session_ids = new_data.projects["project-1"]["session_ids"]
        assert session_ids.count("session-a") == 1
        assert len(session_ids) == 2

    def test_add_session_bumps_updated_at(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Add session bumps the updated_at timestamp."""
        old_updated_at = sample_project_data.projects["project-1"]["updated_at"]

        new_data = project_service.add_session(
            "project-1", "session-c", sample_project_data
        )

        assert new_data.projects["project-1"]["updated_at"] != old_updated_at

    def test_remove_session(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Remove session removes from project session_ids."""
        new_data = project_service.remove_session(
            "project-1", "session-a", sample_project_data
        )

        assert "session-a" not in new_data.projects["project-1"]["session_ids"]
        assert len(new_data.projects["project-1"]["session_ids"]) == 1

    def test_remove_session_not_present(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Remove session for non-present session_id is a no-op."""
        new_data = project_service.remove_session(
            "project-1", "nonexistent", sample_project_data
        )

        assert len(new_data.projects["project-1"]["session_ids"]) == 2

    def test_remove_session_bumps_updated_at(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Remove session bumps the updated_at timestamp."""
        old_updated_at = sample_project_data.projects["project-1"]["updated_at"]

        new_data = project_service.remove_session(
            "project-1", "session-a", sample_project_data
        )

        assert new_data.projects["project-1"]["updated_at"] != old_updated_at

    def test_add_session_nonexistent_project(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Add session to nonexistent project returns unchanged data."""
        new_data = project_service.add_session(
            "nonexistent", "session-c", sample_project_data
        )

        assert new_data == sample_project_data

    def test_remove_session_nonexistent_project(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Remove session from nonexistent project returns unchanged data."""
        new_data = project_service.remove_session(
            "nonexistent", "session-a", sample_project_data
        )

        assert new_data == sample_project_data


class TestProjectServiceGetProject:
    """Tests for ProjectService.get_project()."""

    def test_get_project(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Get project returns the project dict."""
        project = project_service.get_project("project-1", sample_project_data)

        assert project is not None
        assert project["name"] == "Test Project"

    def test_get_project_nonexistent(
        self, project_service: ProjectService, sample_project_data: ProjectData
    ):
        """Get project returns None for nonexistent project."""
        project = project_service.get_project("nonexistent", sample_project_data)

        assert project is None
