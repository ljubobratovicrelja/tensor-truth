"""Project management service - pure business logic.

This service handles all project CRUD operations, returning new state objects
instead of mutating global state.
"""

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tensortruth.app_utils.file_utils import atomic_write_json

from .models import ProjectData

logger = logging.getLogger(__name__)


class ProjectService:
    """Service for managing projects.

    All methods are pure - they accept state as input and return new state
    as output.

    Storage model:
    - Per-project files: ~/.tensortruth/projects/{project_id}/project.json
    - Index file: ~/.tensortruth/projects/projects_index.json (cache for fast listing)
    - Per-project files are authoritative; index is a cache
    """

    def __init__(self, projects_dir: Path):
        self.projects_dir = Path(projects_dir)
        self.index_file = self.projects_dir / "projects_index.json"

    # -------------------------------------------------------------------------
    # Internal file operations
    # -------------------------------------------------------------------------

    def _load_index(self) -> Dict[str, Any]:
        """Load the projects index file."""
        if not self.index_file.exists():
            return {"projects": {}}

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load projects index: {e}")
            return {"projects": {}}

    def _save_index(self, index: Dict[str, Any]) -> None:
        """Save the projects index file atomically."""
        atomic_write_json(self.index_file, index)

    def _load_project_file(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Load a single project's data file."""
        project_file = self.projects_dir / project_id / "project.json"
        if not project_file.exists():
            return None

        try:
            with open(project_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load project {project_id}: {e}")
            return None

    def _save_project_file(self, project_id: str, data: Dict[str, Any]) -> None:
        """Save a single project's data file atomically."""
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        project_file = project_dir / "project.json"
        atomic_write_json(project_file, data)

    def _delete_project_dir(self, project_id: str) -> None:
        """Delete a project's directory."""
        project_dir = self.projects_dir / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load(self) -> ProjectData:
        """Load projects from per-project files.

        Returns:
            ProjectData with projects dict.
            Returns empty data if no projects exist.
        """
        index = self._load_index()
        index_projects = index.get("projects", {})

        projects: Dict[str, Dict[str, Any]] = {}
        stale_ids: List[str] = []

        for project_id in index_projects:
            project_data = self._load_project_file(project_id)
            if project_data is not None:
                projects[project_id] = project_data
            else:
                logger.warning(
                    f"Removing stale index entry for missing project: {project_id}"
                )
                stale_ids.append(project_id)

        if stale_ids:
            for project_id in stale_ids:
                del index_projects[project_id]
            self._save_index(index)

        return ProjectData(projects=projects)

    def save(self, data: ProjectData) -> None:
        """Save projects to per-project files.

        Args:
            data: ProjectData to save.
        """
        old_index = self._load_index()
        old_project_ids = set(old_index.get("projects", {}).keys())

        new_index: Dict[str, Any] = {"projects": {}}

        for project_id, project_data in data.projects.items():
            self._save_project_file(project_id, project_data)
            new_index["projects"][project_id] = {
                "name": project_data.get("name", ""),
                "updated_at": project_data.get("updated_at", ""),
            }

        self._save_index(new_index)

        # Delete removed project directories
        new_project_ids = set(data.projects.keys())
        removed_ids = old_project_ids - new_project_ids
        for project_id in removed_ids:
            self._delete_project_dir(project_id)

    def create(
        self,
        name: str,
        description: str,
        data: ProjectData,
    ) -> Tuple[str, ProjectData]:
        """Create a new project.

        Args:
            name: Project name.
            description: Project description.
            data: Current project data.

        Returns:
            Tuple of (new_project_id, updated_ProjectData).
        """
        new_id = str(uuid.uuid4())
        now = str(datetime.now())

        new_project = {
            "project_id": new_id,
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
            "catalog_modules": {},
            "documents": [],
            "session_ids": [],
            "config": {},
        }

        new_projects = dict(data.projects)
        new_projects[new_id] = new_project

        return new_id, ProjectData(projects=new_projects)

    def update(
        self,
        project_id: str,
        data: ProjectData,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ProjectData:
        """Update a project (partial update).

        Args:
            project_id: Project ID to update.
            data: Current project data.
            name: New name (optional).
            description: New description (optional).
            config: New config (optional).

        Returns:
            Updated ProjectData.
        """
        if project_id not in data.projects:
            return data

        new_projects = dict(data.projects)
        new_projects[project_id] = dict(new_projects[project_id])

        if name is not None:
            new_projects[project_id]["name"] = name
        if description is not None:
            new_projects[project_id]["description"] = description
        if config is not None:
            new_projects[project_id]["config"] = config

        new_projects[project_id]["updated_at"] = str(datetime.now())

        return ProjectData(projects=new_projects)

    def delete(self, project_id: str, data: ProjectData) -> ProjectData:
        """Delete a project from data.

        The project directory is deleted when save() detects the removal.

        Args:
            project_id: Project ID to delete.
            data: Current project data.

        Returns:
            Updated ProjectData without the deleted project.
        """
        if project_id not in data.projects:
            return data

        new_projects = {
            pid: proj for pid, proj in data.projects.items() if pid != project_id
        }

        return ProjectData(projects=new_projects)

    def add_session(
        self, project_id: str, session_id: str, data: ProjectData
    ) -> ProjectData:
        """Add a session to a project (idempotent).

        Args:
            project_id: Project ID.
            session_id: Session ID to add.
            data: Current project data.

        Returns:
            Updated ProjectData.
        """
        if project_id not in data.projects:
            return data

        new_projects = dict(data.projects)
        new_projects[project_id] = dict(new_projects[project_id])
        session_ids = list(new_projects[project_id].get("session_ids", []))

        if session_id not in session_ids:
            session_ids.append(session_id)

        new_projects[project_id]["session_ids"] = session_ids
        new_projects[project_id]["updated_at"] = str(datetime.now())

        return ProjectData(projects=new_projects)

    def remove_session(
        self, project_id: str, session_id: str, data: ProjectData
    ) -> ProjectData:
        """Remove a session from a project.

        Args:
            project_id: Project ID.
            session_id: Session ID to remove.
            data: Current project data.

        Returns:
            Updated ProjectData.
        """
        if project_id not in data.projects:
            return data

        new_projects = dict(data.projects)
        new_projects[project_id] = dict(new_projects[project_id])
        session_ids = list(new_projects[project_id].get("session_ids", []))

        if session_id in session_ids:
            session_ids.remove(session_id)

        new_projects[project_id]["session_ids"] = session_ids
        new_projects[project_id]["updated_at"] = str(datetime.now())

        return ProjectData(projects=new_projects)

    def get_project(
        self, project_id: str, data: ProjectData
    ) -> Optional[Dict[str, Any]]:
        """Get a project by ID.

        Args:
            project_id: Project ID.
            data: Current project data.

        Returns:
            Project dict or None if not found.
        """
        return data.projects.get(project_id)
