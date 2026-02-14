"""Project-related schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CatalogModuleStatus(BaseModel):
    """Status of a catalog module within a project."""

    status: str
    task_id: Optional[str] = None


class DocumentInfo(BaseModel):
    """Information about a document within a project."""

    doc_id: str
    type: str
    filename: Optional[str] = None
    url: Optional[str] = None
    context: Optional[str] = None
    status: str


class ProjectCreate(BaseModel):
    """Request body for creating a new project."""

    name: str = Field(..., description="Project name")
    description: str = Field(default="", description="Project description")


class ProjectUpdate(BaseModel):
    """Request body for updating a project."""

    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class ProjectResponse(BaseModel):
    """Response for a single project."""

    project_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    catalog_modules: Dict[str, CatalogModuleStatus] = Field(default_factory=dict)
    documents: List[DocumentInfo] = Field(default_factory=list)
    session_ids: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


class ProjectListResponse(BaseModel):
    """Response for listing all projects."""

    projects: List[ProjectResponse]


class ProjectSessionCreate(BaseModel):
    """Request body for creating a session within a project."""

    modules: Optional[List[str]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
