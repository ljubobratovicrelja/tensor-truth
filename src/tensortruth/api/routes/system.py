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
from tensortruth.core.system import get_all_memory_info, get_rag_status

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
    context_length: Optional[int] = None


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
                        context_length=model_info.get("context_length"),
                    )
                )

        info_lines = format_ollama_runtime_info()

        return OllamaStatusResponse(
            running=len(models) > 0, models=models, info_lines=info_lines
        )
    except Exception:
        # Ollama not available or error occurred
        return OllamaStatusResponse(running=False, models=[], info_lines=[])


class LlamaCppModelInfo(BaseModel):
    """Information about a llama.cpp model."""

    name: str
    display_name: str
    status: str  # "loaded" | "loading" | "unloaded"


class LlamaCppStatusResponse(BaseModel):
    """Response schema for llama.cpp status endpoint."""

    running: bool
    models: List[LlamaCppModelInfo]
    base_url: str


class LlamaCppActionRequest(BaseModel):
    """Request body for llama.cpp load/unload."""

    model: str
    provider_id: Optional[str] = None


class LlamaCppActionResponse(BaseModel):
    """Response for llama.cpp load/unload actions."""

    success: bool
    message: str


@router.get("/llama-cpp/status", response_model=LlamaCppStatusResponse)
async def get_llama_cpp_status() -> LlamaCppStatusResponse:
    """Get llama.cpp runtime status and available models."""
    from tensortruth.core.llama_cpp import (
        check_health,
        format_display_name,
        get_available_models,
    )
    from tensortruth.core.providers import ProviderRegistry

    registry = ProviderRegistry.get_instance()

    # Find the first llama_cpp provider
    provider = None
    for p in registry.providers:
        if p.type == "llama_cpp":
            provider = p
            break

    if provider is None:
        return LlamaCppStatusResponse(running=False, models=[], base_url="")

    base = provider.base_url.rstrip("/")
    healthy = check_health(base)

    if not healthy:
        return LlamaCppStatusResponse(running=False, models=[], base_url=base)

    # Build static lookup for display names
    static_lookup = {}
    for m in provider.models:
        name = m.get("name", "")
        if name:
            static_lookup[name] = m

    server_models = get_available_models(base)
    models = []
    for sm in server_models:
        model_id = sm.get("id", "")
        if not model_id:
            continue
        static = static_lookup.get(model_id, {})
        models.append(
            LlamaCppModelInfo(
                name=model_id,
                display_name=static.get("display_name") or format_display_name(model_id),
                status=sm.get("status", "unloaded"),
            )
        )

    return LlamaCppStatusResponse(running=True, models=models, base_url=base)


@router.post("/llama-cpp/load", response_model=LlamaCppActionResponse)
async def load_llama_cpp_model(req: LlamaCppActionRequest) -> LlamaCppActionResponse:
    """Load a model into VRAM on the llama.cpp server."""
    from tensortruth.core.llama_cpp import load_model
    from tensortruth.core.providers import ProviderRegistry

    registry = ProviderRegistry.get_instance()

    provider = None
    if req.provider_id:
        provider = registry.get_provider(req.provider_id)
    else:
        for p in registry.providers:
            if p.type == "llama_cpp":
                provider = p
                break

    if provider is None or provider.type != "llama_cpp":
        return LlamaCppActionResponse(success=False, message="No llama_cpp provider found")

    base = provider.base_url.rstrip("/")
    ok = load_model(base, req.model)
    if ok:
        return LlamaCppActionResponse(success=True, message=f"Model {req.model} loaded")
    return LlamaCppActionResponse(success=False, message=f"Failed to load {req.model}")


@router.post("/llama-cpp/unload", response_model=LlamaCppActionResponse)
async def unload_llama_cpp_model(req: LlamaCppActionRequest) -> LlamaCppActionResponse:
    """Unload a model from VRAM on the llama.cpp server."""
    from tensortruth.core.llama_cpp import unload_model
    from tensortruth.core.providers import ProviderRegistry

    registry = ProviderRegistry.get_instance()

    provider = None
    if req.provider_id:
        provider = registry.get_provider(req.provider_id)
    else:
        for p in registry.providers:
            if p.type == "llama_cpp":
                provider = p
                break

    if provider is None or provider.type != "llama_cpp":
        return LlamaCppActionResponse(success=False, message="No llama_cpp provider found")

    base = provider.base_url.rstrip("/")
    ok = unload_model(base, req.model)
    if ok:
        return LlamaCppActionResponse(success=True, message=f"Model {req.model} unloaded")
    return LlamaCppActionResponse(success=False, message=f"Failed to unload {req.model}")


class RAGModelStatus(BaseModel):
    """Status information for a RAG model."""

    loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None
    memory_gb: Optional[float] = None


class RAGStatusResponse(BaseModel):
    """Response schema for RAG status endpoint."""

    active: bool
    embedder: RAGModelStatus
    reranker: RAGModelStatus
    total_memory_gb: float


@router.get("/rag/status", response_model=RAGStatusResponse)
async def get_rag_status_endpoint() -> RAGStatusResponse:
    """Get RAG system status including embedder and reranker details.

    Returns:
    - active: Whether any RAG models are currently loaded
    - embedder: Embedding model status (model name, device, memory)
    - reranker: Reranker model status (model name, device, memory)
    - total_memory_gb: Combined memory usage of RAG models
    """
    status = get_rag_status()
    return RAGStatusResponse(
        active=status.active,
        embedder=RAGModelStatus(
            loaded=status.embedder.loaded,
            model_name=status.embedder.model_name,
            device=status.embedder.device,
            memory_gb=status.embedder.memory_gb,
        ),
        reranker=RAGModelStatus(
            loaded=status.reranker.loaded,
            model_name=status.reranker.model_name,
            device=status.reranker.device,
            memory_gb=status.reranker.memory_gb,
        ),
        total_memory_gb=status.total_memory_gb,
    )


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
