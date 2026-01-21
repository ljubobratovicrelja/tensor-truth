"""Modules and models listing endpoints."""

from typing import Any, Dict, List

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tensortruth.api.deps import ConfigServiceDep
from tensortruth.app_utils.paths import get_indexes_dir, get_presets_file

router = APIRouter()


class ModuleInfo(BaseModel):
    """Information about a knowledge module."""

    name: str
    display_name: str = ""
    doc_type: str = "unknown"
    sort_order: int = 4


class ModulesResponse(BaseModel):
    """Response for listing available modules."""

    modules: List[ModuleInfo]


class ModelInfo(BaseModel):
    """Information about an Ollama model."""

    name: str
    size: int = 0
    modified_at: str = ""


class ModelsResponse(BaseModel):
    """Response for listing available models."""

    models: List[ModelInfo]


class PresetInfo(BaseModel):
    """Information about a preset configuration."""

    name: str
    config: Dict[str, Any]


class PresetsResponse(BaseModel):
    """Response for listing available presets."""

    presets: List[PresetInfo]


@router.get("/modules", response_model=ModulesResponse)
async def list_modules(config_service: ConfigServiceDep) -> ModulesResponse:
    """List available knowledge modules for the configured embedding model.

    Modules are vector indexes stored in ~/.tensortruth/indexes/{model_id}/.
    Returns modules sorted by type (Books, Papers, Libraries, Other).
    """
    from tensortruth.app_utils.helpers import get_module_display_name
    from tensortruth.indexing.metadata import sanitize_model_id

    indexes_dir = get_indexes_dir()
    modules = []

    if indexes_dir.exists():
        # Get the configured embedding model
        config = config_service.load()
        model_id = sanitize_model_id(config.rag.default_embedding_model)
        model_indexes_dir = indexes_dir / model_id

        # Only check versioned structure: indexes/{model_id}/{module}/
        if model_indexes_dir.exists():
            for path in model_indexes_dir.iterdir():
                if path.is_dir() and (path / "chroma.sqlite3").exists():
                    display_name, doc_type, _, sort_order = get_module_display_name(
                        model_indexes_dir, path.name
                    )
                    modules.append(
                        ModuleInfo(
                            name=path.name,
                            display_name=display_name,
                            doc_type=doc_type,
                            sort_order=sort_order,
                        )
                    )

    # Sort by type (sort_order) then by display_name
    modules.sort(key=lambda m: (m.sort_order, m.display_name.lower()))

    return ModulesResponse(modules=modules)


class EmbeddingModelInfo(BaseModel):
    """Information about an available embedding model."""

    model_id: str
    model_name: str | None = None
    index_count: int
    modules: List[str]


class EmbeddingModelsResponse(BaseModel):
    """Response for listing available embedding models."""

    models: List[EmbeddingModelInfo]
    current: str


@router.get("/embedding-models", response_model=EmbeddingModelsResponse)
async def list_embedding_models(
    config_service: ConfigServiceDep,
) -> EmbeddingModelsResponse:
    """List available embedding models based on indexes directory.

    Returns embedding models that have indexes built, along with the
    currently configured model.
    """
    from tensortruth.indexing.metadata import (
        get_available_embedding_models,
        sanitize_model_id,
    )

    indexes_dir = get_indexes_dir()
    available = get_available_embedding_models(indexes_dir)

    config = config_service.load()
    current_model_id = sanitize_model_id(config.rag.default_embedding_model)

    return EmbeddingModelsResponse(
        models=[
            EmbeddingModelInfo(
                model_id=m["model_id"],
                model_name=m.get("model_name"),
                index_count=m["index_count"],
                modules=m["modules"],
            )
            for m in available
        ],
        current=current_model_id,
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(config_service: ConfigServiceDep) -> ModelsResponse:
    """List available Ollama models."""
    ollama_url = config_service.get_ollama_url()

    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()

        models = [
            ModelInfo(
                name=model.get("name", ""),
                size=model.get("size", 0),
                modified_at=model.get("modified_at", ""),
            )
            for model in data.get("models", [])
        ]
        return ModelsResponse(models=models)

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?",
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail=f"Timeout connecting to Ollama at {ollama_url}",
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error fetching models from Ollama: {str(e)}",
        )


@router.get("/presets", response_model=PresetsResponse)
async def list_presets() -> PresetsResponse:
    """List available presets."""
    import json

    presets_file = get_presets_file()
    presets = []

    if presets_file.exists():
        try:
            with open(presets_file, "r") as f:
                data = json.load(f)
                for name, config in data.items():
                    presets.append(PresetInfo(name=name, config=config))
        except (json.JSONDecodeError, IOError):
            pass

    return PresetsResponse(presets=presets)


@router.get("/presets/favorites", response_model=PresetsResponse)
async def list_favorite_presets() -> PresetsResponse:
    """List favorite presets sorted by favorite_order."""
    from tensortruth.app_utils.presets import get_favorites

    presets_file = get_presets_file()
    favorites = get_favorites(presets_file)

    return PresetsResponse(
        presets=[
            PresetInfo(name=name, config=config) for name, config in favorites.items()
        ]
    )
