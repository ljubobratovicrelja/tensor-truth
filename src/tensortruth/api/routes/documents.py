"""Document management endpoints for session and project scopes."""

import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from tensortruth.api.deps import (
    ProjectServiceDep,
    SessionServiceDep,
    TaskRunnerDep,
    get_document_service,
    get_project_service,
    get_task_runner,
)
from tensortruth.api.schemas import (
    CatalogModuleAddRequest,
    CatalogModuleAddResponse,
    CatalogModuleRemoveResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
    ReindexResponse,
    TextUploadRequest,
    UrlUploadRequest,
)
from tensortruth.app_utils.paths import (
    get_base_indexes_dir,
    get_library_docs_dir,
    get_sources_config_path,
)
from tensortruth.indexing.builder import build_module
from tensortruth.services import ProjectService
from tensortruth.services.task_runner import TaskStatus
from tensortruth.utils.sources_config import load_config as load_sources_config

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared helpers (avoid duplication between session/project routes)
# ---------------------------------------------------------------------------


def _list_documents(scope_id: str, scope_type: str) -> DocumentListResponse:
    """List all documents in a scope."""
    with get_document_service(scope_id, scope_type) as doc_service:
        pdf_files = doc_service.get_all_pdf_files()
        md_files = doc_service.get_all_markdown_files()
        has_index = doc_service.index_exists()

    documents: List[DocumentListItem] = []

    # Add PDF files
    for pdf_path in pdf_files:
        import fitz  # PyMuPDF

        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
        except Exception:
            page_count = 0

        documents.append(
            DocumentListItem(
                doc_id=pdf_path.stem,
                filename=pdf_path.name,
                file_size=pdf_path.stat().st_size if pdf_path.exists() else 0,
                page_count=page_count,
            )
        )

    # Add markdown files (text/url uploads)
    for md_path in md_files:
        documents.append(
            DocumentListItem(
                doc_id=md_path.stem,
                filename=md_path.name,
                file_size=md_path.stat().st_size if md_path.exists() else 0,
                page_count=0,
            )
        )

    return DocumentListResponse(documents=documents, has_index=has_index)


def _upload_pdf(
    scope_id: str, scope_type: str, content: bytes, filename: str
) -> DocumentUploadResponse:
    """Upload a PDF document to a scope."""
    with get_document_service(scope_id, scope_type) as doc_service:
        metadata = doc_service.upload(content, filename)

    return DocumentUploadResponse(
        doc_id=metadata.pdf_id,
        filename=metadata.filename,
        file_size=metadata.file_size,
        page_count=metadata.page_count,
    )


def _upload_text(
    scope_id: str, scope_type: str, content: str, filename: str
) -> DocumentUploadResponse:
    """Upload a text/markdown document to a scope."""
    with get_document_service(scope_id, scope_type) as doc_service:
        metadata = doc_service.upload_text(content.encode("utf-8"), filename)

    return DocumentUploadResponse(
        doc_id=metadata.pdf_id,
        filename=metadata.filename,
        file_size=metadata.file_size,
        page_count=metadata.page_count,
    )


def _upload_url(
    scope_id: str, scope_type: str, url: str, context: str = ""
) -> DocumentUploadResponse:
    """Upload content from a URL to a scope."""
    with get_document_service(scope_id, scope_type) as doc_service:
        metadata = doc_service.upload_url(url, context=context)

    return DocumentUploadResponse(
        doc_id=metadata.pdf_id,
        filename=metadata.filename,
        file_size=metadata.file_size,
        page_count=metadata.page_count,
    )


def _delete_document(scope_id: str, scope_type: str, doc_id: str) -> None:
    """Delete a document from a scope."""
    with get_document_service(scope_id, scope_type) as doc_service:
        pdf_files = doc_service.get_all_pdf_files()
        md_files = doc_service.get_all_markdown_files()

        # Check if doc exists in PDFs or markdown files
        found = any(f.stem.startswith(doc_id) for f in pdf_files) or any(
            f.stem.startswith(doc_id) for f in md_files
        )
        if not found:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_service.delete(doc_id)


def _reindex_documents(scope_id: str, scope_type: str) -> ReindexResponse:
    """Rebuild the vector index for a scope."""
    with get_document_service(scope_id, scope_type) as doc_service:
        pdf_count = doc_service.get_pdf_count()

        if pdf_count == 0:
            return ReindexResponse(
                success=False,
                message="No documents to index",
                pdf_count=0,
            )

        try:
            doc_service.rebuild_index()
            return ReindexResponse(
                success=True,
                message=f"Successfully indexed {pdf_count} documents",
                pdf_count=pdf_count,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to rebuild index: {str(e)}",
            )


# ---------------------------------------------------------------------------
# Session document routes
# ---------------------------------------------------------------------------


@router.get("/sessions/{session_id}/documents", response_model=DocumentListResponse)
async def list_session_documents(
    session_id: str,
    session_service: SessionServiceDep,
) -> DocumentListResponse:
    """List all documents in a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return _list_documents(session_id, "session")


@router.post(
    "/sessions/{session_id}/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_session_document(
    session_id: str,
    session_service: SessionServiceDep,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Upload a PDF document to a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    return _upload_pdf(session_id, "session", content, file.filename)


@router.post(
    "/sessions/{session_id}/documents/upload-text",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_session_text(
    session_id: str,
    body: TextUploadRequest,
    session_service: SessionServiceDep,
) -> DocumentUploadResponse:
    """Upload text/markdown content to a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return _upload_text(session_id, "session", body.content, body.filename)


@router.post(
    "/sessions/{session_id}/documents/upload-url",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_session_url(
    session_id: str,
    body: UrlUploadRequest,
    session_service: SessionServiceDep,
) -> DocumentUploadResponse:
    """Upload content from a URL to a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        return _upload_url(session_id, "session", body.url, body.context)
    except (ValueError, ConnectionError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/sessions/{session_id}/documents/{doc_id}", status_code=204)
async def delete_session_document(
    session_id: str,
    doc_id: str,
    session_service: SessionServiceDep,
) -> None:
    """Delete a document from a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    _delete_document(session_id, "session", doc_id)


@router.post(
    "/sessions/{session_id}/documents/reindex",
    response_model=ReindexResponse,
)
async def reindex_session_documents(
    session_id: str,
    session_service: SessionServiceDep,
) -> ReindexResponse:
    """Rebuild the vector index for session documents."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return _reindex_documents(session_id, "session")


# ---------------------------------------------------------------------------
# Project document routes
# ---------------------------------------------------------------------------


def _get_project_or_404(project_id: str, project_service: ProjectService) -> dict:
    """Load a project or raise 404."""
    data = project_service.load()
    project = project_service.get_project(project_id, data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("/projects/{project_id}/documents", response_model=DocumentListResponse)
async def list_project_documents(
    project_id: str,
    project_service: ProjectServiceDep,
) -> DocumentListResponse:
    """List all documents in a project."""
    _get_project_or_404(project_id, project_service)
    return _list_documents(project_id, "project")


@router.post(
    "/projects/{project_id}/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_project_document(
    project_id: str,
    project_service: ProjectServiceDep,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Upload a PDF document to a project."""
    _get_project_or_404(project_id, project_service)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    return _upload_pdf(project_id, "project", content, file.filename)


@router.post(
    "/projects/{project_id}/documents/upload-text",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_project_text(
    project_id: str,
    body: TextUploadRequest,
    project_service: ProjectServiceDep,
) -> DocumentUploadResponse:
    """Upload text/markdown content to a project."""
    _get_project_or_404(project_id, project_service)
    return _upload_text(project_id, "project", body.content, body.filename)


@router.post(
    "/projects/{project_id}/documents/upload-url",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_project_url(
    project_id: str,
    body: UrlUploadRequest,
    project_service: ProjectServiceDep,
) -> DocumentUploadResponse:
    """Upload content from a URL to a project."""
    _get_project_or_404(project_id, project_service)

    try:
        return _upload_url(project_id, "project", body.url, body.context)
    except (ValueError, ConnectionError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/projects/{project_id}/documents/{doc_id}", status_code=204)
async def delete_project_document(
    project_id: str,
    doc_id: str,
    project_service: ProjectServiceDep,
) -> None:
    """Delete a document from a project."""
    _get_project_or_404(project_id, project_service)
    _delete_document(project_id, "project", doc_id)


@router.post(
    "/projects/{project_id}/documents/reindex",
    response_model=ReindexResponse,
)
async def reindex_project_documents(
    project_id: str,
    project_service: ProjectServiceDep,
) -> ReindexResponse:
    """Rebuild the vector index for project documents."""
    _get_project_or_404(project_id, project_service)
    return _reindex_documents(project_id, "project")


# ---------------------------------------------------------------------------
# Catalog module endpoints (project-only)
# ---------------------------------------------------------------------------


def _make_on_complete(
    project_id: str,
    module_name: str,
    project_service: ProjectService,
):
    """Create an on_complete callback for module build tasks."""

    def _on_complete(task_info):
        data = project_service.load()
        project = project_service.get_project(project_id, data)
        if not project:
            return

        if task_info.status == TaskStatus.COMPLETED:
            project["catalog_modules"][module_name] = {"status": "indexed"}
        else:
            project["catalog_modules"][module_name] = {
                "status": "error",
                "error": task_info.error,
            }

        project_service.save(data)

    return _on_complete


@router.post(
    "/projects/{project_id}/catalog-modules",
    response_model=CatalogModuleAddResponse,
    status_code=201,
)
async def add_catalog_module(
    project_id: str,
    body: CatalogModuleAddRequest,
    project_service: ProjectServiceDep,
    task_runner: TaskRunnerDep,
) -> CatalogModuleAddResponse:
    """Add a catalog module to a project (triggers background build)."""
    data = project_service.load()
    project = project_service.get_project(project_id, data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    module_name = body.module_name

    # Validate module_name against available modules from sources config
    sources_config_path = get_sources_config_path()
    sources_config = load_sources_config(sources_config_path)

    all_sources = {
        **sources_config.get("libraries", {}),
        **sources_config.get("papers", {}),
        **sources_config.get("books", {}),
    }

    if module_name not in all_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown module: {module_name}. Not found in sources config.",
        )

    # Check current status of this module in the project
    catalog = project.get("catalog_modules", {})
    existing = catalog.get(module_name)
    if existing and isinstance(existing, dict):
        if existing.get("status") == "building":
            raise HTTPException(
                status_code=409,
                detail=f"Module '{module_name}' is already being built.",
            )
        if existing.get("status") == "indexed":
            raise HTTPException(
                status_code=409,
                detail=f"Module '{module_name}' is already indexed.",
            )

    # Submit build task
    library_docs_dir = get_library_docs_dir()
    indexes_dir = get_base_indexes_dir()

    def _build_fn(progress_callback):
        return build_module(
            module_name=module_name,
            library_docs_dir=library_docs_dir,
            indexes_dir=indexes_dir,
            sources_config=sources_config,
            progress_callback=progress_callback,
        )

    task_id = await task_runner.submit(
        "build_module",
        _build_fn,
        on_complete=_make_on_complete(project_id, module_name, project_service),
        metadata={"project_id": project_id, "module_name": module_name},
    )

    # Update project status to "building"
    if "catalog_modules" not in project:
        project["catalog_modules"] = {}
    project["catalog_modules"][module_name] = {
        "status": "building",
        "task_id": task_id,
    }
    project_service.save(data)

    return CatalogModuleAddResponse(
        task_id=task_id,
        module_name=module_name,
        status="building",
    )


@router.delete(
    "/projects/{project_id}/catalog-modules/{module_name}",
    response_model=CatalogModuleRemoveResponse,
)
async def remove_catalog_module(
    project_id: str,
    module_name: str,
    project_service: ProjectServiceDep,
) -> CatalogModuleRemoveResponse:
    """Remove a catalog module from a project."""
    data = project_service.load()
    project = project_service.get_project(project_id, data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    catalog = project.get("catalog_modules", {})
    if module_name not in catalog:
        raise HTTPException(
            status_code=404,
            detail=f"Module '{module_name}' not found in project.",
        )

    existing = catalog[module_name]
    if isinstance(existing, dict) and existing.get("status") == "building":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot remove module '{module_name}' while it is being built.",
        )

    del project["catalog_modules"][module_name]
    project_service.save(data)

    return CatalogModuleRemoveResponse(
        module_name=module_name,
        status="removed",
    )
