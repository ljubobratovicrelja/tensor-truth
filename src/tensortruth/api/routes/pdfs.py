"""PDF management endpoints."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from tensortruth.api.deps import SessionServiceDep, get_pdf_service
from tensortruth.api.schemas import (
    PDFListResponse,
    PDFMetadataResponse,
    ReindexResponse,
)

router = APIRouter()


@router.get("/sessions/{session_id}/pdfs", response_model=PDFListResponse)
async def list_pdfs(
    session_id: str, session_service: SessionServiceDep
) -> PDFListResponse:
    """List all PDFs in a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    with get_pdf_service(session_id) as pdf_service:
        pdf_files = pdf_service.get_all_pdf_files()
        has_index = pdf_service.index_exists()

        pdfs = []
        for pdf_path in pdf_files:
            # Extract basic metadata from file
            import fitz  # PyMuPDF

            try:
                doc = fitz.open(pdf_path)
                page_count = len(doc)
                doc.close()
            except Exception:
                page_count = 0

            pdfs.append(
                PDFMetadataResponse(
                    pdf_id=pdf_path.stem,
                    filename=pdf_path.name,
                    path=str(pdf_path),
                    file_size=pdf_path.stat().st_size if pdf_path.exists() else 0,
                    page_count=page_count,
                )
            )

        return PDFListResponse(pdfs=pdfs, has_index=has_index)


@router.post(
    "/sessions/{session_id}/pdfs",
    response_model=PDFMetadataResponse,
    status_code=201,
)
async def upload_pdf(
    session_id: str,
    session_service: SessionServiceDep,
    file: UploadFile = File(...),
) -> PDFMetadataResponse:
    """Upload a PDF to a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()

    with get_pdf_service(session_id) as pdf_service:
        metadata = pdf_service.upload(content, file.filename)

        return PDFMetadataResponse(
            pdf_id=metadata.pdf_id,
            filename=metadata.filename,
            path=metadata.path,
            file_size=metadata.file_size,
            page_count=metadata.page_count,
        )


@router.delete("/sessions/{session_id}/pdfs/{pdf_id}", status_code=204)
async def delete_pdf(
    session_id: str, pdf_id: str, session_service: SessionServiceDep
) -> None:
    """Delete a PDF from a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    with get_pdf_service(session_id) as pdf_service:
        # Check if PDF exists before deleting
        pdf_files = pdf_service.get_all_pdf_files()
        if not any(f.stem.startswith(pdf_id) for f in pdf_files):
            raise HTTPException(status_code=404, detail="PDF not found")
        pdf_service.delete(pdf_id)


@router.post("/sessions/{session_id}/pdfs/reindex", response_model=ReindexResponse)
async def reindex_pdfs(
    session_id: str, session_service: SessionServiceDep
) -> ReindexResponse:
    """Rebuild the vector index for session PDFs."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    with get_pdf_service(session_id) as pdf_service:
        pdf_count = pdf_service.get_pdf_count()

        if pdf_count == 0:
            return ReindexResponse(
                success=False,
                message="No PDFs to index",
                pdf_count=0,
            )

        try:
            pdf_service.rebuild_index()
            return ReindexResponse(
                success=True,
                message=f"Successfully indexed {pdf_count} PDFs",
                pdf_count=pdf_count,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to rebuild index: {str(e)}",
            )
