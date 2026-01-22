"""System information endpoints."""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from tensortruth.app_utils.helpers import (
    format_ollama_runtime_info,
    free_memory,
    get_ollama_ps,
    get_system_devices,
)
from tensortruth.core.system import MemoryInfo as CoreMemoryInfo
from tensortruth.core.system import get_all_memory_info

router = APIRouter()


class MemoryInfo(BaseModel):
    """Memory usage information for a component."""

    name: str
    allocated_gb: float
    total_gb: Optional[float] = None
    details: Optional[str] = None


class MemoryResponse(BaseModel):
    """Response schema for memory endpoint."""

    memory: List[MemoryInfo]


class DevicesResponse(BaseModel):
    """Response schema for devices endpoint."""

    devices: List[str]


class OllamaModelInfo(BaseModel):
    """Information about a running Ollama model."""

    name: str
    size_vram_gb: float
    size_gb: float
    parameters: Optional[str] = None


class OllamaStatusResponse(BaseModel):
    """Response schema for Ollama status endpoint."""

    running: bool
    models: List[OllamaModelInfo]
    info_lines: List[str]


def _convert_memory_info(core_info: CoreMemoryInfo) -> MemoryInfo:
    """Convert core MemoryInfo to API schema."""
    return MemoryInfo(
        name=core_info.name,
        allocated_gb=core_info.allocated_gb,
        total_gb=core_info.total_gb,
        details=core_info.details,
    )


@router.get("/memory", response_model=MemoryResponse)
async def get_memory() -> MemoryResponse:
    """Get comprehensive memory usage across all components.

    Returns memory info for:
    - CUDA VRAM (if available)
    - MPS unified memory (if available)
    - Ollama VRAM (if models running)
    - System RAM
    """
    memory_info = get_all_memory_info()
    return MemoryResponse(memory=[_convert_memory_info(info) for info in memory_info])


@router.get("/devices", response_model=DevicesResponse)
async def get_devices() -> DevicesResponse:
    """Get list of available compute devices for this system.

    Returns list in order of preference: cuda > mps > cpu
    """
    devices = get_system_devices()
    return DevicesResponse(devices=devices)


@router.get("/ollama/status", response_model=OllamaStatusResponse)
async def get_ollama_status() -> OllamaStatusResponse:
    """Get Ollama runtime status and running models information.

    Returns:
    - running: Whether any models are currently running
    - models: List of running model details
    - info_lines: Formatted information strings for display
    """
    try:
        running_models = get_ollama_ps()
        models = []

        if running_models:
            for model_info in running_models:
                models.append(
                    OllamaModelInfo(
                        name=model_info.get("name", "Unknown"),
                        size_vram_gb=model_info.get("size_vram", 0) / (1024**3),
                        size_gb=model_info.get("size", 0) / (1024**3),
                        parameters=model_info.get("details", {}).get("parameter_size"),
                    )
                )

        info_lines = format_ollama_runtime_info()

        return OllamaStatusResponse(
            running=len(models) > 0, models=models, info_lines=info_lines
        )
    except Exception:
        # Ollama not available or error occurred
        return OllamaStatusResponse(running=False, models=[], info_lines=[])


class RestartEngineResponse(BaseModel):
    """Response schema for restart engine endpoint."""

    success: bool
    message: str


@router.post("/restart-engine", response_model=RestartEngineResponse)
async def restart_engine() -> RestartEngineResponse:
    """Restart the RAG engine by clearing memory and caches.

    Frees GPU/MPS memory by:
    - Clearing retriever LRU cache
    - Deleting engine instance
    - Clearing LlamaIndex embedding model
    - Clearing PDF processing models (MARKER_CONVERTER)
    - Running garbage collection
    - Emptying CUDA/MPS cache

    Returns:
    - success: Whether the operation succeeded
    - message: Status message
    """
    try:
        free_memory()
        return RestartEngineResponse(
            success=True, message="Engine restarted successfully"
        )
    except Exception as e:
        return RestartEngineResponse(
            success=False, message=f"Failed to restart engine: {str(e)}"
        )
