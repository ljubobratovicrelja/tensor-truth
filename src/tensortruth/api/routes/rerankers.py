"""Reranker model management endpoints."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from tensortruth.api.deps import ConfigServiceDep

router = APIRouter()


class RerankerModelInfo(BaseModel):
    """Information about a reranker model."""

    model: str
    status: str  # "valid" or "unknown" (we don't validate on list, only on add)


class RerankerListResponse(BaseModel):
    """Response for GET /rerankers."""

    models: list[RerankerModelInfo]
    current: str


class RerankerAddRequest(BaseModel):
    """Request body for POST /rerankers."""

    model: str


class RerankerAddResponse(BaseModel):
    """Response for POST /rerankers."""

    status: str  # "added" or "failed"
    model: str | None = None
    error: str | None = None


class RerankerRemoveResponse(BaseModel):
    """Response for DELETE /rerankers/{model_id}."""

    status: str  # "removed" or "failed"
    error: str | None = None


def _validate_hf_model(model_name: str) -> tuple[bool, str | None]:
    """Check if a model exists on HuggingFace Hub.

    Args:
        model_name: HuggingFace model path (e.g., "BAAI/bge-reranker-v2-m3")

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        from huggingface_hub import model_info

        model_info(model_name)
        return True, None
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            return False, "Model not found on HuggingFace Hub"
        return False, f"Failed to validate model: {error_msg}"


@router.get("", response_model=RerankerListResponse)
async def list_rerankers(config_service: ConfigServiceDep) -> RerankerListResponse:
    """List configured reranker models."""
    config = config_service.load()
    models = config.rag.get_reranker_models()

    return RerankerListResponse(
        models=[RerankerModelInfo(model=m, status="valid") for m in models],
        current=config.rag.default_reranker,
    )


@router.post("", response_model=RerankerAddResponse)
async def add_reranker(
    body: RerankerAddRequest, config_service: ConfigServiceDep
) -> RerankerAddResponse:
    """Add a new reranker model.

    Validates that the model exists on HuggingFace Hub before adding.
    The model itself is downloaded lazily on first use.
    """
    model_name = body.model.strip()

    if not model_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model name cannot be empty",
        )

    # Check if already exists
    config = config_service.load()
    existing_models = config.rag.get_reranker_models()
    if model_name in existing_models:
        return RerankerAddResponse(
            status="added",
            model=model_name,
        )

    # Validate model exists on HuggingFace
    is_valid, error = _validate_hf_model(model_name)
    if not is_valid:
        return RerankerAddResponse(
            status="failed",
            error=error,
        )

    # Add to config
    new_models = existing_models + [model_name]
    config_service.update(rag_reranker_models=new_models)

    return RerankerAddResponse(
        status="added",
        model=model_name,
    )


@router.delete("/{model_id:path}", response_model=RerankerRemoveResponse)
async def remove_reranker(
    model_id: str, config_service: ConfigServiceDep
) -> RerankerRemoveResponse:
    """Remove a reranker model from config.

    Cannot remove the currently selected default reranker.
    """
    config = config_service.load()
    existing_models = config.rag.get_reranker_models()

    # Decode URL-encoded model path
    model_name = model_id.replace("%2F", "/")

    if model_name not in existing_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reranker model '{model_name}' not found in config",
        )

    # Cannot remove the current default
    if model_name == config.rag.default_reranker:
        return RerankerRemoveResponse(
            status="failed",
            error="Cannot remove the currently selected default reranker",
        )

    # Remove from config
    new_models = [m for m in existing_models if m != model_name]
    config_service.update(rag_reranker_models=new_models)

    return RerankerRemoveResponse(status="removed")
