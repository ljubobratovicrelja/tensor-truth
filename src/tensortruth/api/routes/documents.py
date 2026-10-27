"""Document management endpoints for session and project scopes."""

import asyncio
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import urlparse

import arxiv
import requests
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from tensortruth.api.deps import (
    ConfigServiceDep,
    ProjectServiceDep,
    SessionServiceDep,
    TaskRunnerDep,
    get_document_service,
)
from tensortruth.api.schemas import (
    ArxivUploadRequest,
    CatalogModuleAddRequest,
    CatalogModuleAddResponse,
    CatalogModuleRemoveResponse,
    DocumentContextRequest,
    DocumentContextResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
    FileUrlInfoResponse,
    FileUrlUploadRequest,
    IndexingConfig,
    IndexingConfigUpdate,
    ReindexTaskResponse,
    TextUploadRequest,
    UrlUploadRequest,
)
from tensortruth.app_utils.paths import get_indexes_dir
from tensortruth.indexing.metadata import sanitize_model_id
from tensortruth.services import ProjectService, TaskRunner
from tensortruth.services.context_documents import CHARS_PER_TOKEN
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


def _context_tokens(chars: Optional[int]) -> Optional[int]:
    """Approximate token cost of in-context text (matches orchestrator)."""
    if chars is None:
        return None
    return chars // CHARS_PER_TOKEN


def _mark_in_context(
    scope_id: str, scope_type: str, doc_id: str, in_context: bool = True
) -> Optional[int]:
    """Attach/detach a document to the conversation context.

    Returns the approximate token cost when attaching, else None.

    Raises:
        HTTPException: 400 when the document has no extractable text.
    """
    try:
        with get_document_service(scope_id, scope_type) as doc_service:
            chars = doc_service.set_in_context(doc_id, in_context)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _context_tokens(chars)


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
        builder = doc_service._get_index_builder()
        indexed_doc_ids = builder.get_indexed_doc_ids() if has_index else set()
        # Snapshot in-context flags so document listings can badge them.
        context_docs = {
            doc_id: metadata.get("context_chars")
            for doc_id, metadata in doc_service.get_metadata_cache().items()
            if (metadata or {}).get("in_context")
        }

    # Compute index mtime if an index exists
    index_updated_at: Optional[str] = None
    if has_index and index_dir is not None:
        chroma_file = index_dir / "chroma.sqlite3"
        if chroma_file.exists():
            index_updated_at = _mtime_iso(chroma_file)

    documents: List[DocumentListItem] = []
    unindexed_count = 0

    # Add PDF files and collect their doc_ids
    pdf_doc_ids: set[str] = set()
    for pdf_path in pdf_files:
        import fitz  # PyMuPDF

        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
        except Exception:
            page_count = 0

        doc_id = extract_doc_id(pdf_path.stem)
        pdf_doc_ids.add(doc_id)
        is_indexed = doc_id in indexed_doc_ids
        if not is_indexed:
            unindexed_count += 1
        documents.append(
            DocumentListItem(
                doc_id=doc_id,
                filename=get_display_name(pdf_path),
                file_size=pdf_path.stat().st_size if pdf_path.exists() else 0,
                page_count=page_count,
                uploaded_at=_mtime_iso(pdf_path),
                is_indexed=is_indexed,
                in_context=doc_id in context_docs,
                context_tokens=_context_tokens(context_docs.get(doc_id)),
            )
        )

    # Add markdown files (text/url uploads), excluding conversion artifacts
    for md_path in md_files:
        md_doc_id = extract_doc_id(md_path.stem)
        # Skip markdown files that are conversion artifacts of existing PDFs
        if md_doc_id in pdf_doc_ids:
            continue
        is_indexed = md_doc_id in indexed_doc_ids
        if not is_indexed:
            unindexed_count += 1
        documents.append(
            DocumentListItem(
                doc_id=md_doc_id,
                filename=get_display_name(md_path),
                file_size=md_path.stat().st_size if md_path.exists() else 0,
                page_count=0,
                uploaded_at=_mtime_iso(md_path),
                is_indexed=is_indexed,
                in_context=md_doc_id in context_docs,
                context_tokens=_context_tokens(context_docs.get(md_doc_id)),
            )
        )

    return DocumentListResponse(
        documents=documents,
        has_index=has_index,
        index_updated_at=index_updated_at,
        unindexed_count=unindexed_count,
    )


def _upload_pdf(
    scope_id: str,
    scope_type: str,
    content: bytes,
    filename: str,
    in_context: bool = False,
) -> DocumentUploadResponse:
    """Upload a PDF document to a scope."""
    with get_document_service(scope_id, scope_type) as doc_service:
        metadata = doc_service.upload(content, filename)

    tokens = (
        _mark_in_context(scope_id, scope_type, metadata.pdf_id) if in_context else None
    )
    return DocumentUploadResponse(
        doc_id=metadata.pdf_id,
        filename=metadata.filename,
        file_size=metadata.file_size,
        page_count=metadata.page_count,
        in_context=in_context,
        context_tokens=tokens,
    )


def _upload_text(
    scope_id: str,
    scope_type: str,
    content: str,
    filename: str,
    in_context: bool = False,
) -> DocumentUploadResponse:
    """Upload a text/markdown document to a scope."""
    with get_document_service(scope_id, scope_type) as doc_service:
        metadata = doc_service.upload_text(content.encode("utf-8"), filename)

    tokens = (
        _mark_in_context(scope_id, scope_type, metadata.pdf_id) if in_context else None
    )
    return DocumentUploadResponse(
        doc_id=metadata.pdf_id,
        filename=metadata.filename,
        file_size=metadata.file_size,
        page_count=metadata.page_count,
        in_context=in_context,
        context_tokens=tokens,
    )


def _upload_url(
    scope_id: str,
    scope_type: str,
    url: str,
    context: str = "",
    in_context: bool = False,
) -> DocumentUploadResponse:
    """Upload content from a URL to a scope."""
    with get_document_service(scope_id, scope_type) as doc_service:
        metadata = doc_service.upload_url(url, context=context)

    tokens = (
        _mark_in_context(scope_id, scope_type, metadata.pdf_id) if in_context else None
    )
    return DocumentUploadResponse(
        doc_id=metadata.pdf_id,
        filename=metadata.filename,
        file_size=metadata.file_size,
        page_count=metadata.page_count,
        in_context=in_context,
        context_tokens=tokens,
    )


def _apply_arxiv_metadata(
    scope_id: str,
    scope_type: str,
    doc_id: str,
    paper,
    normalized: str,
) -> None:
    """Persist arXiv metadata on a freshly uploaded paper (blocking worker)."""
    from tensortruth.utils.metadata import format_authors as _fmt_authors

    authors_str = ", ".join(a.name for a in paper.authors)
    formatted = _fmt_authors(authors_str)
    display = f"{paper.title}, {formatted}" if formatted else paper.title
    arxiv_metadata = {
        "title": paper.title,
        "authors": authors_str,
        "display_name": display,
        "source_url": f"https://arxiv.org/abs/{normalized}",
        "doc_type": "arxiv_paper",
    }
    # The service is created and closed on the worker thread so its stores
    # keep thread affinity.
    with get_document_service(scope_id, scope_type) as doc_service:
        doc_service.set_metadata(doc_id, arxiv_metadata)


async def _upload_arxiv(
    scope_id: str, scope_type: str, arxiv_id: str, in_context: bool = False
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

    if paper is None or pdf_bytes is None:
        raise HTTPException(status_code=404, detail="Paper not found on arXiv")

    filename = f"{title}.pdf"
    # PDF storage touches the filesystem and metadata store — keep it off the
    # event loop. _upload_pdf opens and closes its own DocumentService.
    result = await asyncio.to_thread(
        _upload_pdf, scope_id, scope_type, pdf_bytes, filename
    )
    if in_context:
        result = result.model_copy(update={"in_context": False, "context_tokens": None})

    # Persist arXiv metadata so the indexer skips LLM extraction.
    # _apply_arxiv_metadata creates its DocumentService on the worker thread.
    await asyncio.to_thread(
        _apply_arxiv_metadata, scope_id, scope_type, result.doc_id, paper, normalized
    )

    if in_context:
        # _mark_in_context runs pymupdf4llm when the markdown does not exist
        # yet — the expensive blocking call this whole module must avoid.
        tokens = await asyncio.to_thread(
            _mark_in_context, scope_id, scope_type, result.doc_id
        )
        result = result.model_copy(
            update={"in_context": True, "context_tokens": tokens}
        )

    return result


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


def _get_indexing_config(
    scope_id: str,
    scope_type: str,
    project_service: Optional[ProjectService] = None,
) -> IndexingConfig:
    """Read indexing config from project config or return defaults."""
    if scope_type == "project" and project_service is not None:
        data = project_service.load()
        project = project_service.get_project(scope_id, data)
        if project is not None:
            cfg = project.get("config", {}).get("indexing", {})
            return IndexingConfig(**cfg)
    return IndexingConfig()


def _count_index_chunks(doc_service) -> Optional[int]:
    """Best-effort count of vectors in the scope's index.

    Returns None when the index does not exist or its store cannot be read.
    """
    try:
        builder = doc_service._get_index_builder()
        if not builder.index_exists():
            return None
        collection = builder._get_chroma_client().get_or_create_collection("data")
        return int(collection.count())
    except Exception as e:  # pragma: no cover - defensive, chroma internals
        logger.debug("Could not count index chunks: %s", e)
        return None


async def _build_index(
    scope_id: str,
    scope_type: str,
    task_runner: TaskRunner,
    project_service: Optional[ProjectService] = None,
    on_complete: Optional[Callable] = None,
) -> ReindexTaskResponse:
    """Submit an incremental build-index job to the TaskRunner.

    Shared by the REST endpoints and the ``build_index`` agent tool so both
    paths resolve indexing config, progress reporting and completion
    bookkeeping identically.

    Args:
        on_complete: Optional TaskRunner completion callback (receives the
            TaskInfo). Used to clear the RAG engine cache after a build.
    """

    def _count_scope() -> tuple[int, int, list]:
        # Directory walks + metadata reads; service opened/closed on the
        # worker thread so its stores keep thread affinity.
        with get_document_service(scope_id, scope_type) as doc_service:
            pdf_count = doc_service.get_pdf_count()
            md_count = len(doc_service.get_all_markdown_files())
            unindexed = doc_service.get_unindexed_doc_ids()
        return pdf_count, md_count, unindexed

    pdf_count, md_count, unindexed = await asyncio.to_thread(_count_scope)

    if pdf_count == 0 and md_count == 0:
        raise HTTPException(status_code=400, detail="No documents to index")

    if not unindexed:
        raise HTTPException(status_code=400, detail="All documents already indexed")

    indexing_cfg = _get_indexing_config(scope_id, scope_type, project_service)

    def _do_build(progress_callback):
        with get_document_service(scope_id, scope_type) as doc_service:
            doc_service.build_index(
                chunk_sizes=indexing_cfg.chunk_sizes,
                progress_callback=progress_callback,
                conversion_method=indexing_cfg.conversion_method,
            )
            chunk_count = _count_index_chunks(doc_service)
        return {
            "scope_id": scope_id,
            "scope_type": scope_type,
            "documents_indexed": len(unindexed),
            "chunks": chunk_count,
        }

    task_id = await task_runner.submit(
        "build-index",
        _do_build,
        on_complete=on_complete,
        metadata={"scope_id": scope_id, "scope_type": scope_type},
    )
    return ReindexTaskResponse(task_id=task_id, pdf_count=len(unindexed))


_SUPPORTED_FILE_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
}


def _validate_url(url: str) -> None:
    """Validate that a URL is http(s) with a non-empty netloc."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL")


def _filename_from_response(resp: requests.Response, url: str) -> str:
    """Extract a filename from Content-Disposition header, URL path, or fallback."""
    cd = resp.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        # Try to parse filename from Content-Disposition
        for part in cd.split(";"):
            part = part.strip()
            if part.startswith("filename="):
                name = part[len("filename=") :].strip().strip('"').strip("'")
                if name:
                    return name

    # Fall back to URL path
    path = urlparse(url).path
    if path and "/" in path:
        name = path.rsplit("/", 1)[-1]
        if name:
            return name

    return "download"


@router.get("/file-url-info", response_model=FileUrlInfoResponse)
async def probe_file_url(
    url: str = Query(..., description="URL to probe"),
) -> FileUrlInfoResponse:
    """Probe a file URL via HEAD to get metadata before downloading."""
    _validate_url(url)

    loop = asyncio.get_event_loop()

    def _probe():
        try:
            resp = requests.head(url, allow_redirects=True, timeout=10)
        except requests.RequestException:
            # HEAD might be blocked; try ranged GET
            try:
                resp = requests.get(
                    url,
                    headers={"Range": "bytes=0-0"},
                    allow_redirects=True,
                    timeout=10,
                )
            except requests.RequestException as e:
                raise ConnectionError(str(e))

        if resp.status_code == 405:
            # HEAD not allowed, try ranged GET
            try:
                resp = requests.get(
                    url,
                    headers={"Range": "bytes=0-0"},
                    allow_redirects=True,
                    timeout=10,
                )
            except requests.RequestException as e:
                raise ConnectionError(str(e))

        content_type = resp.headers.get("Content-Type", "application/octet-stream")
        # Strip charset etc.
        content_type = content_type.split(";")[0].strip().lower()

        content_length = resp.headers.get("Content-Length")
        file_size = int(content_length) if content_length else None

        filename = _filename_from_response(resp, url)
        supported = content_type in _SUPPORTED_FILE_TYPES

        return FileUrlInfoResponse(
            url=url,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            supported=supported,
        )

    try:
        result = await loop.run_in_executor(None, _probe)
    except ConnectionError as e:
        raise HTTPException(status_code=400, detail=f"Could not reach URL: {e}")

    return result


async def _upload_file_url(
    scope_id: str, scope_type: str, url: str, in_context: bool = False
) -> DocumentUploadResponse:
    """Download a file from a URL and upload it to a scope."""
    _validate_url(url)

    loop = asyncio.get_event_loop()

    def _download():
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp

    try:
        resp = await asyncio.wait_for(
            loop.run_in_executor(None, _download),
            timeout=60,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="File download timeout")
    except Exception as e:
        logger.error(f"File URL download failed for {url}: {e}")
        raise HTTPException(status_code=502, detail="File download error")

    content_type = resp.headers.get("Content-Type", "application/octet-stream")
    content_type = content_type.split(";")[0].strip().lower()

    filename = _filename_from_response(resp, url)

    if content_type == "application/pdf":
        # Upload + optional context attach touch disk and can trigger PDF
        # conversion — run the whole helper on a worker thread.
        return await asyncio.to_thread(
            _upload_pdf,
            scope_id,
            scope_type,
            resp.content,
            filename,
            in_context=in_context,
        )
    elif content_type in ("text/plain", "text/markdown"):
        return await asyncio.to_thread(
            _upload_text,
            scope_id,
            scope_type,
            resp.content.decode("utf-8", errors="replace"),
            filename,
            in_context=in_context,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}",
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

    # _list_documents walks the scope directory and opens every PDF to count
    # pages — real filesystem I/O that must not run on the event loop.
    return await asyncio.to_thread(_list_documents, session_id, "session")


@router.post(
    "/sessions/{session_id}/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_session_document(
    session_id: str,
    session_service: SessionServiceDep,
    file: UploadFile = File(...),
    in_context: bool = Query(
        False, description="Attach the document text to the chat context (no index)"
    ),
) -> DocumentUploadResponse:
    """Upload a PDF document to a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()
    # Upload stores the file and may convert the PDF to markdown — blocking.
    return await asyncio.to_thread(
        _upload_pdf,
        session_id,
        "session",
        content,
        file.filename,
        in_context=in_context,
    )


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

    # Writes the markdown file and updates the metadata store — blocking I/O.
    return await asyncio.to_thread(
        _upload_text,
        session_id,
        "session",
        body.content,
        body.filename,
        in_context=body.in_context,
    )


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

    # Network download + storage — run on a worker thread.
    try:
        return await asyncio.to_thread(
            _upload_url,
            session_id,
            "session",
            body.url,
            body.context,
            in_context=body.in_context,
        )
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

    return await _upload_arxiv(
        session_id, "session", body.arxiv_id, in_context=body.in_context
    )


@router.post(
    "/sessions/{session_id}/documents/upload-file-url",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_session_file_url(
    session_id: str,
    body: FileUrlUploadRequest,
    session_service: SessionServiceDep,
) -> DocumentUploadResponse:
    """Download a file from a URL and add it to a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return await _upload_file_url(
        session_id, "session", body.url, in_context=body.in_context
    )


@router.patch(
    "/sessions/{session_id}/documents/{doc_id}/context",
    response_model=DocumentContextResponse,
)
async def set_session_document_context(
    session_id: str,
    doc_id: str,
    body: DocumentContextRequest,
    session_service: SessionServiceDep,
) -> DocumentContextResponse:
    """Attach or detach a document from the session's chat context.

    Attached documents are pasted verbatim into the LLM context on every
    turn — no vector index is built or needed.
    """
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # _mark_in_context -> DocumentService.set_in_context -> ensure_markdown ->
    # pymupdf4llm: ~16s for a 50-page paper. Never on the event loop.
    tokens = await asyncio.to_thread(
        _mark_in_context, session_id, "session", doc_id, body.in_context
    )
    return DocumentContextResponse(
        doc_id=doc_id, in_context=body.in_context, context_tokens=tokens
    )


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

    # Directory scan + file deletion — blocking filesystem I/O.
    await asyncio.to_thread(_delete_document, session_id, "session", doc_id)


@router.post(
    "/sessions/{session_id}/documents/build-index",
    response_model=ReindexTaskResponse,
)
async def build_session_index(
    session_id: str,
    session_service: SessionServiceDep,
    task_runner: TaskRunnerDep,
) -> ReindexTaskResponse:
    """Build the vector index for session documents (incremental)."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    from tensortruth.api.routes.index_management import _make_on_complete

    return await _build_index(
        session_id, "session", task_runner, on_complete=_make_on_complete()
    )


# ---------------------------------------------------------------------------
# Project helpers
# ---------------------------------------------------------------------------


def _get_project_or_404(project_id: str, project_service: ProjectService) -> dict:
    """Load a project or raise 404."""
    data = project_service.load()
    project = project_service.get_project(project_id, data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


# ---------------------------------------------------------------------------
# Project indexing config
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/indexing-config",
    response_model=IndexingConfig,
)
async def get_project_indexing_config(
    project_id: str,
    project_service: ProjectServiceDep,
) -> IndexingConfig:
    """Get the indexing configuration for a project."""
    _get_project_or_404(project_id, project_service)
    return _get_indexing_config(project_id, "project", project_service)


@router.patch(
    "/projects/{project_id}/indexing-config",
    response_model=IndexingConfig,
)
async def update_project_indexing_config(
    project_id: str,
    body: IndexingConfigUpdate,
    project_service: ProjectServiceDep,
) -> IndexingConfig:
    """Update the indexing configuration for a project."""
    data = project_service.load()
    project = project_service.get_project(project_id, data)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    config = project.setdefault("config", {})
    indexing = config.setdefault("indexing", {})

    old_chunk_sizes = indexing.get("chunk_sizes")
    old_conversion = indexing.get("conversion_method")

    if body.chunk_sizes is not None:
        indexing["chunk_sizes"] = body.chunk_sizes
    if body.conversion_method is not None:
        if body.conversion_method not in ("marker", "direct"):
            raise HTTPException(
                status_code=400, detail="conversion_method must be 'marker' or 'direct'"
            )
        indexing["conversion_method"] = body.conversion_method

    # If settings actually changed and index exists, delete it so next build
    # does a full rebuild
    settings_changed = (
        indexing.get("chunk_sizes") != old_chunk_sizes
        or indexing.get("conversion_method") != old_conversion
    )

    def _persist_and_invalidate() -> None:
        # Config write + possible index directory deletion: both blocking.
        # Service is created and closed on the worker thread.
        project_service.save(data)
        if settings_changed:
            with get_document_service(project_id, "project") as doc_service:
                if doc_service.index_exists():
                    logger.info(
                        f"Indexing settings changed for project {project_id}, "
                        "deleting index"
                    )
                    doc_service.delete_index()

    await asyncio.to_thread(_persist_and_invalidate)

    return IndexingConfig(**indexing)


# ---------------------------------------------------------------------------
# Project document routes
# ---------------------------------------------------------------------------


@router.get("/projects/{project_id}/documents", response_model=DocumentListResponse)
async def list_project_documents(
    project_id: str,
    project_service: ProjectServiceDep,
) -> DocumentListResponse:
    """List all documents in a project."""
    _get_project_or_404(project_id, project_service)
    return await asyncio.to_thread(_list_documents, project_id, "project")


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
    return await asyncio.to_thread(
        _upload_pdf, project_id, "project", content, file.filename
    )


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
    return await asyncio.to_thread(
        _upload_text, project_id, "project", body.content, body.filename
    )


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
        return await asyncio.to_thread(
            _upload_url, project_id, "project", body.url, body.context
        )
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


@router.post(
    "/projects/{project_id}/documents/upload-file-url",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_project_file_url(
    project_id: str,
    body: FileUrlUploadRequest,
    project_service: ProjectServiceDep,
) -> DocumentUploadResponse:
    """Download a file from a URL and add it to a project."""
    _get_project_or_404(project_id, project_service)
    return await _upload_file_url(project_id, "project", body.url)


@router.delete("/projects/{project_id}/documents/{doc_id}", status_code=204)
async def delete_project_document(
    project_id: str,
    doc_id: str,
    project_service: ProjectServiceDep,
) -> None:
    """Delete a document from a project."""
    _get_project_or_404(project_id, project_service)
    await asyncio.to_thread(_delete_document, project_id, "project", doc_id)


@router.post(
    "/projects/{project_id}/documents/build-index",
    response_model=ReindexTaskResponse,
)
async def build_project_index(
    project_id: str,
    project_service: ProjectServiceDep,
    task_runner: TaskRunnerDep,
) -> ReindexTaskResponse:
    """Build the vector index for project documents (incremental)."""
    _get_project_or_404(project_id, project_service)

    from tensortruth.api.routes.index_management import _make_on_complete

    return await _build_index(
        project_id,
        "project",
        task_runner,
        project_service,
        on_complete=_make_on_complete(),
    )


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
    # Project config write (JSON on disk) — keep it off the event loop.
    await asyncio.to_thread(project_service.save, data)

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
    await asyncio.to_thread(project_service.save, data)

    return CatalogModuleRemoveResponse(
        module_name=module_name,
        status="removed",
    )
