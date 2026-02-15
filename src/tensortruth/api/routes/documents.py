"""Document management endpoints for session and project scopes."""

import asyncio
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import arxiv
from fastapi import APIRouter, File, HTTPException, UploadFile

from tensortruth.api.deps import (
    ConfigServiceDep,
    ProjectServiceDep,
    SessionServiceDep,
    get_document_service,
)
from tensortruth.api.schemas import (
    ArxivUploadRequest,
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
from tensortruth.app_utils.paths import get_indexes_dir
from tensortruth.indexing.metadata import sanitize_model_id
from tensortruth.services import ProjectService
from tensortruth.utils.validation import validate_arxiv_id

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared helpers (avoid duplication between session/project routes)
# ---------------------------------------------------------------------------


_DOC_PREFIX_RE = re.compile(r"^(pdf|doc|url)_[a-f0-9]{7,8}_")
_MD_HEADER_RE = re.compile(r"^# Document:\s*(.+)")


def extract_doc_id(stem: str) -> str:
    """Extract the short doc ID prefix from a full file stem.

    Examples:
        pdf_544414c_my_paper -> pdf_544414c
        doc_abcd1234_notes   -> doc_abcd1234
        url_12345678_slug    -> url_12345678

    Returns the original stem unchanged if no prefix pattern matches.
    """
    m = _DOC_PREFIX_RE.match(stem)
    if m:
        # m.group(0) includes the trailing underscore, strip it
        return m.group(0).rstrip("_")
    return stem


def strip_doc_prefix(filename: str) -> str:
    """Strip internal ID prefix from a document filename.

    Works well for PDFs where the original name is preserved in the stem:
        pdf_544414c_my_paper.pdf  -> my_paper.pdf

    For markdown files, prefer get_display_name() which reads the stored header.
    """
    return _DOC_PREFIX_RE.sub("", filename)


def get_display_name(file_path: Path) -> str:
    """Extract the user-facing display name for a document.

    - For PDFs: strips the internal ID prefix (original name is in the stem).
    - For markdown files (text/url uploads): reads the ``# Document: ...``
      header written at upload time, which contains the original filename
      or page title.
    - Falls back to prefix-stripping if the header can't be read.
    """
    if file_path.suffix == ".pdf":
        return strip_doc_prefix(file_path.name)

    # Markdown files — read the first line for the original name
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        m = _MD_HEADER_RE.match(first_line)
        if m:
            return m.group(1).strip()
    except OSError:
        pass

    return strip_doc_prefix(file_path.name)


def _mtime_iso(path: Path) -> Optional[str]:
    """Return file mtime as ISO-8601 UTC string, or None on error."""
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _list_documents(scope_id: str, scope_type: str) -> DocumentListResponse:
    """List all documents in a scope."""
    with get_document_service(scope_id, scope_type) as doc_service:
        pdf_files = doc_service.get_all_pdf_files()
        md_files = doc_service.get_all_markdown_files()
        has_index = doc_service.index_exists()
        index_dir = doc_service.get_index_path()

    # Compute index mtime if an index exists
    index_updated_at: Optional[str] = None
    if has_index and index_dir is not None:
        chroma_file = index_dir / "chroma.sqlite3"
        if chroma_file.exists():
            index_updated_at = _mtime_iso(chroma_file)

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
                doc_id=extract_doc_id(pdf_path.stem),
                filename=get_display_name(pdf_path),
                file_size=pdf_path.stat().st_size if pdf_path.exists() else 0,
                page_count=page_count,
                uploaded_at=_mtime_iso(pdf_path),
            )
        )

    # Add markdown files (text/url uploads)
    for md_path in md_files:
        documents.append(
            DocumentListItem(
                doc_id=extract_doc_id(md_path.stem),
                filename=get_display_name(md_path),
                file_size=md_path.stat().st_size if md_path.exists() else 0,
                page_count=0,
                uploaded_at=_mtime_iso(md_path),
            )
        )

    return DocumentListResponse(
        documents=documents,
        has_index=has_index,
        index_updated_at=index_updated_at,
    )


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


async def _upload_arxiv(
    scope_id: str, scope_type: str, arxiv_id: str
) -> DocumentUploadResponse:
    """Download an arXiv paper PDF and upload it to a scope."""
    normalized = validate_arxiv_id(arxiv_id)
    if normalized is None:
        raise HTTPException(status_code=400, detail="Invalid arXiv ID")

    def _fetch_and_download():
        search = arxiv.Search(id_list=[normalized])
        try:
            paper = next(search.results())
        except StopIteration:
            return None, None, None

        with tempfile.TemporaryDirectory() as tmpdir:
            paper.download_pdf(dirpath=tmpdir, filename="paper.pdf")
            pdf_bytes = Path(tmpdir, "paper.pdf").read_bytes()

        return paper.title, pdf_bytes, paper

    try:
        loop = asyncio.get_event_loop()
        title, pdf_bytes, paper = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch_and_download),
            timeout=60,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="arXiv download timeout")
    except Exception as e:
        logger.error(f"arXiv download failed for {arxiv_id}: {e}")
        raise HTTPException(status_code=502, detail="arXiv download error")

    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found on arXiv")

    filename = f"{title}.pdf"
    return _upload_pdf(scope_id, scope_type, pdf_bytes, filename)


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


@router.post(
    "/sessions/{session_id}/documents/upload-arxiv",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_session_arxiv(
    session_id: str,
    body: ArxivUploadRequest,
    session_service: SessionServiceDep,
) -> DocumentUploadResponse:
    """Download an arXiv paper and add it to a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return await _upload_arxiv(session_id, "session", body.arxiv_id)


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


@router.post(
    "/projects/{project_id}/documents/upload-arxiv",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_project_arxiv(
    project_id: str,
    body: ArxivUploadRequest,
    project_service: ProjectServiceDep,
) -> DocumentUploadResponse:
    """Download an arXiv paper and add it to a project."""
    _get_project_or_404(project_id, project_service)
    return await _upload_arxiv(project_id, "project", body.arxiv_id)


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


@router.post(
    "/projects/{project_id}/catalog-modules",
    response_model=CatalogModuleAddResponse,
    status_code=201,
)
async def add_catalog_module(
    project_id: str,
    body: CatalogModuleAddRequest,
    project_service: ProjectServiceDep,
    config_service: ConfigServiceDep,
) -> CatalogModuleAddResponse:
    """Add an existing catalog module (built index) to a project."""
    data = project_service.load()
    project = project_service.get_project(project_id, data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    module_name = body.module_name

    # Validate that the module exists as a built index on disk
    config = config_service.load()
    model_id = sanitize_model_id(config.rag.default_embedding_model)
    indexes_dir = get_indexes_dir()
    module_dir = indexes_dir / model_id / module_name

    if not module_dir.exists() or not (module_dir / "chroma.sqlite3").exists():
        raise HTTPException(
            status_code=400,
            detail=f"Module '{module_name}' not found. No built index exists.",
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

    # Module already exists on disk — just reference it
    if "catalog_modules" not in project:
        project["catalog_modules"] = {}
    project["catalog_modules"][module_name] = {"status": "indexed"}
    project_service.save(data)

    return CatalogModuleAddResponse(
        module_name=module_name,
        status="indexed",
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
