"""Document management schemas for session and project scopes."""

from typing import List, Optional

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    """Response for a document upload operation."""

    doc_id: str
    filename: str
    file_size: int
    page_count: int


class DocumentListItem(BaseModel):
    """A single document in a listing."""

    doc_id: str
    filename: str
    file_size: int
    page_count: int
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for listing documents in a scope."""

    documents: List[DocumentListItem]
    has_index: bool = False
    index_updated_at: Optional[str] = None


class UrlUploadRequest(BaseModel):
    """Request body for uploading content from a URL."""

    url: str
    context: str = ""


class TextUploadRequest(BaseModel):
    """Request body for uploading text/markdown content."""

    content: str
    filename: str


class CatalogModuleAddRequest(BaseModel):
    """Request body for adding a catalog module to a project."""

    module_name: str


class CatalogModuleAddResponse(BaseModel):
    """Response for adding a catalog module."""

    task_id: Optional[str] = None
    module_name: str
    status: str


class ArxivLookupResponse(BaseModel):
    """Response for an arXiv metadata lookup."""

    arxiv_id: str
    title: str
    authors: List[str]
    published: str
    categories: List[str]
    abstract: str
    pdf_url: str


class ArxivUploadRequest(BaseModel):
    """Request body for uploading an arXiv paper as PDF."""

    arxiv_id: str


class CatalogModuleRemoveResponse(BaseModel):
    """Response for removing a catalog module."""

    module_name: str
    status: str
