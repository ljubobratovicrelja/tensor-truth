"""Startup and initialization endpoints."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from tensortruth.api.deps import StartupServiceDep
from tensortruth.api.schemas import (
    IndexDownloadRequest,
    IndexDownloadResponse,
    ModelPullRequest,
    ModelPullResponse,
    ReinitializeIndexesResponse,
    StartupStatusResponse,
)
from tensortruth.app_utils.helpers import (
    download_and_extract_indexes,
    get_available_hf_embedding_indexes,
)
from tensortruth.app_utils.paths import get_indexes_dir, get_user_data_dir
from tensortruth.core.ollama import pull_model

logger = logging.getLogger(__name__)

router = APIRouter()

# Thread pool for blocking operations
_thread_pool = ThreadPoolExecutor(max_workers=2)


# ============================================================================
# Embedding Model Suggestion/Selection Schemas
# ============================================================================


class EmbeddingModelSuggestion(BaseModel):
    """A suggested embedding model."""

    model_name: str = Field(..., description="Full HuggingFace model path")
    model_id: str = Field(..., description="Sanitized model ID")
    description: str = Field(..., description="Brief description of the model")


class EmbeddingModelSuggestionsResponse(BaseModel):
    """Response with suggested embedding models."""

    suggestions: List[EmbeddingModelSuggestion]
    default: str = Field(..., description="Recommended default model")


class AvailableHFIndexEntry(BaseModel):
    """An available index entry from HuggingFace Hub."""

    embedding_model: str
    embedding_model_id: str
    filename: str
    version: str


class AvailableHFIndexesResponse(BaseModel):
    """Response with available indexes from HuggingFace Hub."""

    available_indexes: List[AvailableHFIndexEntry]
    default_model: str = Field(..., description="Default embedding model")


@router.get("/status", response_model=StartupStatusResponse)
async def get_startup_status(
    startup_service: StartupServiceDep,
) -> StartupStatusResponse:
    """Get comprehensive startup status.

    Checks:
    - Directory initialization
    - Configuration loading
    - Vector indexes availability
    - Ollama models availability
    - Embedding model mismatch

    Returns status, warnings, and readiness flag.
    """
    status = startup_service.check_startup_status()

    return StartupStatusResponse(
        directories_ok=status["directories_ok"],
        config_ok=status["config_ok"],
        indexes_ok=status["indexes_ok"],
        models_ok=status["models_ok"],
        indexes_status=status["indexes_status"],
        models_status=status["models_status"],
        embedding_mismatch=status.get("embedding_mismatch"),
        ready=status["ready"],
        warnings=status["warnings"],
    )


@router.get(
    "/embedding-models/suggestions", response_model=EmbeddingModelSuggestionsResponse
)
async def get_embedding_model_suggestions() -> EmbeddingModelSuggestionsResponse:
    """Get suggested embedding models for index building.

    Returns a list of recommended embedding models with descriptions.
    Users can use ANY HuggingFace model, but these are good starting points.
    """
    suggestions = [
        EmbeddingModelSuggestion(
            model_name="BAAI/bge-m3",
            model_id="bge-m3",
            description="Multilingual, high quality, good balance of size/performance",
        ),
        EmbeddingModelSuggestion(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_id="qwen3-embedding-0.6b",
            description="Small, fast, efficient for English text",
        ),
        EmbeddingModelSuggestion(
            model_name="Qwen/Qwen3-Embedding-4B",
            model_id="qwen3-embedding-4b",
            description="Large, high quality, best for complex documents",
        ),
        EmbeddingModelSuggestion(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_id="all-minilm-l6-v2",
            description="Very fast, lightweight, English only",
        ),
    ]

    return EmbeddingModelSuggestionsResponse(
        suggestions=suggestions,
        default="BAAI/bge-m3",
    )


@router.get(
    "/embedding-models/available-indexes", response_model=AvailableHFIndexesResponse
)
async def get_available_embedding_indexes() -> AvailableHFIndexesResponse:
    """Get embedding models with pre-built indexes available on HuggingFace Hub.

    Returns a list of embedding models for which pre-built indexes are available
    for download from the HuggingFace repository.
    """
    available = get_available_hf_embedding_indexes()

    return AvailableHFIndexesResponse(
        available_indexes=[
            AvailableHFIndexEntry(
                embedding_model=entry.get("embedding_model", ""),
                embedding_model_id=entry.get("embedding_model_id", ""),
                filename=entry.get("filename", ""),
                version=entry.get("version", ""),
            )
            for entry in available
        ],
        default_model="BAAI/bge-m3",
    )


def _download_indexes_sync(
    repo_id: str,
    filename: str | None = None,
    embedding_model: str | None = None,
) -> bool:
    """Synchronous wrapper for index download (runs in thread pool)."""
    try:
        user_dir = get_user_data_dir()
        success = download_and_extract_indexes(
            user_dir,
            repo_id=repo_id,
            filename=filename,
            embedding_model=embedding_model,
        )
        if success:
            model_info = f" for {embedding_model}" if embedding_model else ""
            logger.info(f"✓ Indexes{model_info} downloaded and extracted successfully")
        else:
            logger.error("✗ Index download failed")
        return success
    except Exception as e:
        logger.error(f"Error downloading indexes: {e}", exc_info=True)
        return False


async def _download_indexes_background(
    repo_id: str,
    filename: str | None = None,
    embedding_model: str | None = None,
):
    """Background task to download indexes."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _thread_pool, _download_indexes_sync, repo_id, filename, embedding_model
    )


@router.post("/download-indexes", response_model=IndexDownloadResponse)
async def download_indexes(
    request: IndexDownloadRequest,
    background_tasks: BackgroundTasks,
) -> IndexDownloadResponse:
    """Trigger background download of vector indexes from HuggingFace Hub.

    If embedding_model is provided, downloads indexes for that specific model.
    Otherwise falls back to filename or default.

    This endpoint returns immediately and runs the download in the background.
    Poll /startup/status to check when indexes become available.
    """
    try:
        # Add background task
        background_tasks.add_task(
            _download_indexes_background,
            request.repo_id,
            request.filename,
            request.embedding_model,
        )

        model_info = (
            f" for {request.embedding_model}" if request.embedding_model else ""
        )
        logger.info(f"Started index download{model_info}: {request.repo_id}")

        return IndexDownloadResponse(
            status="started",
            message=(
                f"Downloading indexes{model_info} from {request.repo_id}. "
                f"This may take a few minutes. Poll /startup/status to check progress."
            ),
        )

    except Exception as e:
        logger.error(f"Failed to start index download: {e}", exc_info=True)
        return IndexDownloadResponse(
            status="error",
            message=f"Failed to start download: {str(e)}",
        )


def _pull_model_sync(model_name: str) -> bool:
    """Synchronous wrapper for model pull (runs in thread pool)."""
    try:
        logger.info(f"Pulling model: {model_name}")
        success = pull_model(model_name)
        if success:
            logger.info(f"✓ Model pulled successfully: {model_name}")
        else:
            logger.error(f"✗ Model pull failed: {model_name}")
        return success
    except Exception as e:
        logger.error(f"Error pulling model {model_name}: {e}", exc_info=True)
        return False


async def _pull_model_background(model_name: str):
    """Background task to pull Ollama model."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_thread_pool, _pull_model_sync, model_name)


@router.post("/pull-model", response_model=ModelPullResponse)
async def pull_ollama_model(
    request: ModelPullRequest,
    background_tasks: BackgroundTasks,
) -> ModelPullResponse:
    """Trigger background pull of Ollama model.

    This endpoint returns immediately and runs the pull in the background.
    Poll /startup/status to check when model becomes available.
    """
    try:
        # Add background task
        background_tasks.add_task(_pull_model_background, request.model_name)

        logger.info(f"Started model pull: {request.model_name}")

        return ModelPullResponse(
            status="started",
            message=(
                f"Pulling model {request.model_name}. "
                f"This may take several minutes. Poll /startup/status to check progress."
            ),
        )

    except Exception as e:
        logger.error(f"Failed to start model pull: {e}", exc_info=True)
        return ModelPullResponse(
            status="error",
            message=f"Failed to start pull: {str(e)}",
        )


def _reinitialize_indexes_sync() -> bool:
    """Delete indexes directory and redownload (runs in thread pool)."""
    import shutil

    try:
        indexes_dir = get_indexes_dir()

        # Delete existing indexes directory
        if indexes_dir.exists():
            logger.info(f"Deleting indexes directory: {indexes_dir}")
            shutil.rmtree(indexes_dir)

        # Recreate empty directory
        indexes_dir.mkdir(parents=True, exist_ok=True)

        # Download fresh indexes
        logger.info("Downloading fresh indexes from HuggingFace Hub")
        user_dir = get_user_data_dir()
        success = download_and_extract_indexes(user_dir)

        if success:
            logger.info("✓ Indexes reinitialized successfully")
        else:
            logger.error("✗ Index reinitialization failed")

        return success

    except Exception as e:
        logger.error(f"Error reinitializing indexes: {e}", exc_info=True)
        return False


async def _reinitialize_indexes_background():
    """Background task to reinitialize indexes."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_thread_pool, _reinitialize_indexes_sync)


@router.delete("/reinitialize-indexes", response_model=ReinitializeIndexesResponse)
async def reinitialize_indexes(
    background_tasks: BackgroundTasks,
) -> ReinitializeIndexesResponse:
    """Delete existing indexes and download fresh copies from HuggingFace Hub.

    This endpoint:
    1. Deletes the ~/.tensortruth/indexes/ directory
    2. Downloads fresh indexes from HuggingFace Hub
    3. Extracts them to ~/.tensortruth/indexes/

    Useful for:
    - Fixing corrupted indexes
    - Updating to latest index version
    - Resetting to default state

    This endpoint returns immediately and runs in the background.
    Poll /startup/status to check when indexes become available.
    """
    try:
        # Add background task
        background_tasks.add_task(_reinitialize_indexes_background)

        logger.info("Started index reinitialization")

        return ReinitializeIndexesResponse(
            status="started",
            message=(
                "Deleting existing indexes and downloading fresh copies. "
                "This may take a few minutes. Poll /startup/status to check progress."
            ),
        )

    except Exception as e:
        logger.error(f"Failed to start index reinitialization: {e}", exc_info=True)
        return ReinitializeIndexesResponse(
            status="error",
            message=f"Failed to start reinitialization: {str(e)}",
        )
