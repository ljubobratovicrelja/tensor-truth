"""Schemas for the Index Management API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceCreateRequest(BaseModel):
    """Request to add a new source to sources.json."""

    source_type: str = Field(description="One of: libraries, papers, books")
    name: str = Field(description="Source key name (e.g. 'pytorch', 'dl_foundations')")
    config: Dict[str, Any] = Field(description="Source configuration dictionary")


class SourceUpdateRequest(BaseModel):
    """Request to update an existing source."""

    config: Dict[str, Any] = Field(description="Updated source configuration")


class SourcesResponse(BaseModel):
    """Full contents of sources.json."""

    libraries: Dict[str, Any] = Field(default_factory=dict)
    papers: Dict[str, Any] = Field(default_factory=dict)
    books: Dict[str, Any] = Field(default_factory=dict)


class ModuleOverview(BaseModel):
    """Enriched module info combining source config + fetch status + index status."""

    name: str
    source_type: str
    display_name: str
    doc_type: str = ""
    source_config: Dict[str, Any] = Field(default_factory=dict)
    in_sources_config: bool = False
    has_fetched_docs: bool = False
    doc_dir_size_mb: Optional[float] = None
    has_index: bool = False
    index_size_mb: Optional[float] = None
    index_metadata: Optional[Dict[str, Any]] = None


class ModulesOverviewResponse(BaseModel):
    """List of enriched module overviews."""

    modules: List[ModuleOverview]


class FetchRequest(BaseModel):
    """Request to fetch documentation for a source."""

    source_type: str
    name: str
    workers: int = 20
    converter: str = "marker"
    output_format: str = "markdown"
    enable_cleanup: bool = False
    min_size: int = 0
    pages_per_chunk: int = 15


class BuildRequest(BaseModel):
    """Request to build a vector index for a module."""

    module_name: str
    chunking_strategy: str = "structure"
    chunk_sizes: List[int] = Field(default_factory=lambda: [2048, 512, 256])
    chunk_overlap: int = 64
    embedding_model: Optional[str] = None
    semantic_buffer_size: int = 1
    semantic_breakpoint_threshold: int = 95


class BatchBuildRequest(BaseModel):
    """Request to build indexes for multiple modules."""

    module_names: List[str]
    chunking_strategy: str = "structure"
    chunk_sizes: List[int] = Field(default_factory=lambda: [2048, 512, 256])
    chunk_overlap: int = 64
    embedding_model: Optional[str] = None
    semantic_buffer_size: int = 1
    semantic_breakpoint_threshold: int = 95


class ValidateResponse(BaseModel):
    """Response from source validation."""

    valid: bool
    issues: List[str] = Field(default_factory=list)


class SourceDocumentInfo(BaseModel):
    """Info about a single source document in an index's docstore."""

    doc_id: str
    file_name: str
    title: Optional[str] = None
    display_name: Optional[str] = None
    doc_type: Optional[str] = None
    source_url: Optional[str] = None
    file_size: Optional[int] = None
    chunk_count: int


class SourceDocumentsResponse(BaseModel):
    """List of source documents for a module."""

    module_name: str
    documents: List[SourceDocumentInfo]


class ChunkBoundary(BaseModel):
    """A chunk's position within reconstructed document text."""

    chunk_id: str
    start: int
    end: int
    level: int = 0
    # Section path of the node, so the viewer can name a chunk instead of
    # showing a bare id; also the only heading context a nested leaf can show.
    heading_path: Optional[str] = None


class DocumentContentResponse(BaseModel):
    """Reconstructed document text with chunk boundaries."""

    doc_id: str
    metadata: Dict[str, Any]
    text: str
    chunks: List[ChunkBoundary]
