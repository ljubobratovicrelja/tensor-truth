"""Project management endpoints."""

from fastapi import APIRouter, HTTPException

from tensortruth.api.deps import (
    ConfigServiceDep,
    ProjectServiceDep,
    SessionServiceDep,
    get_document_service,
)
from tensortruth.api.routes.documents import extract_doc_id, get_display_name
from tensortruth.api.routes.sessions import _session_to_response
from tensortruth.api.schemas import (
    CatalogModuleStatus,
    DocumentInfo,
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectSessionCreate,
    ProjectUpdate,
    SessionResponse,
)
from tensortruth.app_utils.paths import get_session_dir

router = APIRouter()


def _project_to_response(project_id: str, project: dict) -> ProjectResponse:
    """Convert internal project dict to response schema."""
    # Convert catalog_modules dict values to CatalogModuleStatus
    raw_modules = project.get("catalog_modules", {})
    catalog_modules = {}
    for mod_name, mod_data in raw_modules.items():
        if isinstance(mod_data, dict):
            catalog_modules[mod_name] = CatalogModuleStatus(**mod_data)
        else:
            catalog_modules[mod_name] = CatalogModuleStatus(status=str(mod_data))

    # Scan disk for documents instead of reading from stored (always-empty) JSON
    documents = []
    try:
        with get_document_service(project_id, "project") as doc_service:
            pdf_doc_ids: set[str] = set()
            for pdf_path in doc_service.get_all_pdf_files():
                doc_id = extract_doc_id(pdf_path.stem)
                pdf_doc_ids.add(doc_id)
                doc_type = "pdf" if doc_id.startswith("pdf_") else "text"
                documents.append(
                    DocumentInfo(
                        doc_id=doc_id,
                        type=doc_type,
                        filename=get_display_name(pdf_path),
                        status="uploaded",
                    )
                )
            # Add standalone markdown files, skip PDF conversion artifacts
            for md_path in doc_service.get_all_markdown_files():
                doc_id = extract_doc_id(md_path.stem)
                if doc_id in pdf_doc_ids:
                    continue
                if doc_id.startswith("url_"):
                    doc_type = "url"
                elif doc_id.startswith("doc_"):
                    doc_type = "text"
                else:
                    doc_type = "text"
                documents.append(
                    DocumentInfo(
                        doc_id=doc_id,
                        type=doc_type,
                        filename=get_display_name(md_path),
                        status="uploaded",
                    )
                )
    except Exception:
        pass  # Project dir may not exist yet for newly created projects

    return ProjectResponse(
        project_id=project_id,
        name=project.get("name", ""),
        description=project.get("description", ""),
        created_at=project.get("created_at", ""),
        updated_at=project.get("updated_at", ""),
        catalog_modules=catalog_modules,
        documents=documents,
        session_ids=project.get("session_ids", []),
        config=project.get("config", {}),
    )


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    project_service: ProjectServiceDep,
) -> ProjectListResponse:
    """List all projects, sorted by updated_at descending."""
    data = project_service.load()
    projects = [_project_to_response(pid, proj) for pid, proj in data.projects.items()]
    projects.sort(key=lambda p: p.updated_at, reverse=True)
    return ProjectListResponse(projects=projects)


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(
    body: ProjectCreate,
    project_service: ProjectServiceDep,
) -> ProjectResponse:
    """Create a new project."""
    data = project_service.load()
    new_id, new_data = project_service.create(
        name=body.name,
        description=body.description,
        data=data,
    )
    project_service.save(new_data)
    return _project_to_response(new_id, new_data.projects[new_id])


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    project_service: ProjectServiceDep,
) -> ProjectResponse:
    """Get a project by ID."""
    data = project_service.load()
    project = project_service.get_project(project_id, data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _project_to_response(project_id, project)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    body: ProjectUpdate,
    project_service: ProjectServiceDep,
) -> ProjectResponse:
    """Update a project (name, description, config)."""
    data = project_service.load()
    project = project_service.get_project(project_id, data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    data = project_service.update(
        project_id,
        data,
        name=body.name,
        description=body.description,
        config=body.config,
    )
    project_service.save(data)
    return _project_to_response(project_id, data.projects[project_id])


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: str,
    project_service: ProjectServiceDep,
    session_service: SessionServiceDep,
) -> None:
    """Delete a project and cascade-delete its sessions."""
    project_data = project_service.load()
    project = project_service.get_project(project_id, project_data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Cascade: delete all sessions owned by this project
    session_ids = project.get("session_ids", [])
    if session_ids:
        session_data = session_service.load()
        for sid in session_ids:
            if sid in session_data.sessions:
                session_dir = get_session_dir(sid)
                session_data = session_service.delete(
                    sid, session_data, session_dir=session_dir
                )
        session_service.save(session_data)

    # Delete the project
    project_data = project_service.delete(project_id, project_data)
    project_service.save(project_data)


@router.post("/{project_id}/sessions", response_model=SessionResponse, status_code=201)
async def create_project_session(
    project_id: str,
    body: ProjectSessionCreate,
    project_service: ProjectServiceDep,
    session_service: SessionServiceDep,
    config_service: ConfigServiceDep,
) -> SessionResponse:
    """Create a new session within a project with config inheritance."""
    project_data = project_service.load()
    project = project_service.get_project(project_id, project_data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Config inheritance: global defaults < project config < user params
    merged_params = {**project.get("config", {}), **body.params}

    session_data = session_service.load()
    new_id, session_data = session_service.create(
        modules=body.modules,
        params=merged_params,
        data=session_data,
        config_service=config_service,
    )

    # Tag the session with project_id
    session_data.sessions[new_id]["project_id"] = project_id
    session_service.save(session_data)

    # Add session to project
    project_data = project_service.add_session(project_id, new_id, project_data)
    project_service.save(project_data)

    return _session_to_response(new_id, session_data.sessions[new_id])


@router.get("/{project_id}/sessions", response_model=list[SessionResponse])
async def list_project_sessions(
    project_id: str,
    project_service: ProjectServiceDep,
    session_service: SessionServiceDep,
) -> list[SessionResponse]:
    """List all sessions belonging to a project."""
    project_data = project_service.load()
    project = project_service.get_project(project_id, project_data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    session_ids = project.get("session_ids", [])
    session_data = session_service.load()

    sessions = []
    for sid in session_ids:
        session = session_data.sessions.get(sid)
        if session:
            sessions.append(_session_to_response(sid, session))

    sessions.sort(key=lambda s: s.created_at, reverse=True)
    return sessions
