"""Startup and initialization endpoints."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks

from tensortruth.api.deps import StartupServiceDep
from tensortruth.api.schemas import (
    IndexDownloadRequest,
    IndexDownloadResponse,
    ModelPullRequest,
    ModelPullResponse,
    ReinitializeIndexesResponse,
    StartupStatusResponse,
)
from tensortruth.app_utils.helpers import download_and_extract_indexes
from tensortruth.app_utils.paths import get_indexes_dir, get_user_data_dir
from tensortruth.core.ollama import pull_model

logger = logging.getLogger(__name__)

router = APIRouter()

# Thread pool for blocking operations
_thread_pool = ThreadPoolExecutor(max_workers=2)


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
        ready=status["ready"],
        warnings=status["warnings"],
    )


def _download_indexes_sync(repo_id: str, filename: str) -> bool:
    """Synchronous wrapper for index download (runs in thread pool)."""
    try:
        user_dir = get_user_data_dir()
        success = download_and_extract_indexes(
            user_dir, repo_id=repo_id, filename=filename
        )
        if success:
            logger.info("✓ Indexes downloaded and extracted successfully")
        else:
            logger.error("✗ Index download failed")
        return success
    except Exception as e:
        logger.error(f"Error downloading indexes: {e}", exc_info=True)
        return False


async def _download_indexes_background(repo_id: str, filename: str):
    """Background task to download indexes."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_thread_pool, _download_indexes_sync, repo_id, filename)


@router.post("/download-indexes", response_model=IndexDownloadResponse)
async def download_indexes(
    request: IndexDownloadRequest,
    background_tasks: BackgroundTasks,
) -> IndexDownloadResponse:
    """Trigger background download of vector indexes from HuggingFace Hub.

    This endpoint returns immediately and runs the download in the background.
    Poll /startup/status to check when indexes become available.
    """
    try:
        # Add background task
        background_tasks.add_task(
            _download_indexes_background,
            request.repo_id,
            request.filename,
        )

        logger.info(f"Started index download: {request.repo_id}/{request.filename}")

        return IndexDownloadResponse(
            status="started",
            message=(
                f"Downloading indexes from {request.repo_id}. "
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
