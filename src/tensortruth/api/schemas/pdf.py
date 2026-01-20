"""PDF-related schemas."""

from typing import List

from pydantic import BaseModel


class PDFMetadataResponse(BaseModel):
    """Response for PDF metadata."""

    pdf_id: str
    filename: str
    path: str
    file_size: int
    page_count: int


class PDFListResponse(BaseModel):
    """Response for listing session PDFs."""

    pdfs: List[PDFMetadataResponse]
    has_index: bool = False


class ReindexResponse(BaseModel):
    """Response for reindex operation."""

    success: bool
    message: str
    pdf_count: int
