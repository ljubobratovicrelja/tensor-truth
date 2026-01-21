"""Startup-related schemas."""

from typing import List

from pydantic import BaseModel, Field


class IndexesStatusSchema(BaseModel):
    """Indexes availability status."""

    exists: bool = Field(..., description="Indexes directory exists")
    has_content: bool = Field(
        ..., description="Indexes contain valid ChromaDB databases"
    )


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
    ready: bool = Field(..., description="App ready to run (critical checks passed)")
    warnings: List[str] = Field(..., description="Non-critical warnings")


class IndexDownloadRequest(BaseModel):
    """Request to download indexes from HuggingFace Hub."""

    repo_id: str = Field(
        "ljubobratovicrelja/tensor-truth-indexes",
        description="HuggingFace repository ID",
    )
    filename: str = Field(
        "indexes_v0.1.14.tar",
        description="Tarball filename to download",
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
