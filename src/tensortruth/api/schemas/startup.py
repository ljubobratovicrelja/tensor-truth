"""Startup-related schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field


class AvailableEmbeddingModelSchema(BaseModel):
    """Information about an available embedding model with indexes."""

    model_id: str = Field(..., description="Sanitized model ID (e.g., 'bge-m3')")
    model_name: Optional[str] = Field(
        None, description="Full HuggingFace model name if available"
    )
    index_count: int = Field(..., description="Number of module indexes available")
    modules: List[str] = Field(..., description="List of module names with indexes")


class IndexesStatusSchema(BaseModel):
    """Indexes availability status."""

    exists: bool = Field(..., description="Indexes directory exists")
    has_content: bool = Field(
        ..., description="Indexes contain valid ChromaDB databases"
    )
    available_models: Optional[List[AvailableEmbeddingModelSchema]] = Field(
        None, description="Embedding models with available indexes"
    )


class EmbeddingMismatchSchema(BaseModel):
    """Embedding model configuration mismatch information."""

    config_model: str = Field(
        ..., description="Embedding model configured in config.yaml"
    )
    config_model_id: str = Field(..., description="Sanitized config model ID")
    available_model_ids: List[str] = Field(
        ..., description="Model IDs with available indexes"
    )
    message: str = Field(..., description="Human-readable warning message")


class ModelsStatusSchema(BaseModel):
    """Ollama models availability status."""

    required: List[str] = Field(..., description="Required model names")
    available: List[str] = Field(..., description="All available models")
    missing: List[str] = Field(..., description="Missing required models")


class StartupStatusResponse(BaseModel):
    """Comprehensive startup status."""

    directories_ok: bool = Field(..., description="Required directories exist")
    config_ok: bool = Field(..., description="Configuration loaded successfully")
    indexes_ok: bool = Field(..., description="Indexes available")
    models_ok: bool = Field(..., description="All required models available")
    indexes_status: IndexesStatusSchema
    models_status: ModelsStatusSchema
    embedding_mismatch: Optional[EmbeddingMismatchSchema] = Field(
        None, description="Embedding model mismatch warning if applicable"
    )
    ready: bool = Field(..., description="App ready to run (critical checks passed)")
    warnings: List[str] = Field(..., description="Non-critical warnings")


class IndexDownloadRequest(BaseModel):
    """Request to download indexes from HuggingFace Hub."""

    repo_id: str = Field(
        "ljubobratovicrelja/tensor-truth-indexes",
        description="HuggingFace repository ID",
    )
    filename: Optional[str] = Field(
        None,
        description="Tarball filename to download (auto-selected if embedding_model provided)",
    )
    embedding_model: Optional[str] = Field(
        None,
        description="Embedding model to download indexes for (e.g., 'BAAI/bge-m3')",
    )


class IndexDownloadResponse(BaseModel):
    """Response for index download request."""

    status: str = Field(..., description="Download status (started, completed, error)")
    message: str = Field(..., description="Human-readable status message")


class ModelPullRequest(BaseModel):
    """Request to pull an Ollama model."""

    model_name: str = Field(..., description="Model name to pull (e.g., llama3.2:3b)")


class ModelPullResponse(BaseModel):
    """Response for model pull request."""

    status: str = Field(..., description="Pull status (started, completed, error)")
    message: str = Field(..., description="Human-readable status message")


class ReinitializeIndexesResponse(BaseModel):
    """Response for reinitialize indexes request."""

    status: str = Field(..., description="Operation status (started, error)")
    message: str = Field(..., description="Human-readable status message")
