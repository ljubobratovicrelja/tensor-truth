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
    description: str = ""


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
async def list_modules() -> ModulesResponse:
    """List available knowledge modules.

    Modules are vector indexes stored in ~/.tensortruth/indexes/.
    """
    indexes_dir = get_indexes_dir()
    modules = []

    if indexes_dir.exists():
        for path in indexes_dir.iterdir():
            if path.is_dir():
                # Check if it's a valid index (has chroma.sqlite3)
                if (path / "chroma.sqlite3").exists():
                    modules.append(
                        ModuleInfo(
                            name=path.name,
                            description=f"Knowledge module: {path.name}",
                        )
                    )

    return ModulesResponse(modules=modules)


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
