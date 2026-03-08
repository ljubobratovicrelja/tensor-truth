"""Modules and models listing endpoints."""

import logging
from typing import Dict, List

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tensortruth.api.deps import ConfigServiceDep
from tensortruth.app_utils.paths import get_indexes_dir

logger = logging.getLogger(__name__)

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
    """Information about an available LLM model."""

    name: str
    provider_id: str = "ollama"
    provider_type: str = "ollama"
    display_name: str = ""
    size: int = 0
    modified_at: str = ""
    capabilities: List[str] = []
    status: str | None = None


class ModelsResponse(BaseModel):
    """Response for listing available models."""

    models: List[ModelInfo]


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
    from tensortruth.app_utils.config_schema import DEFAULT_EMBEDDING_MODEL_CONFIGS
    from tensortruth.indexing.metadata import (
        get_available_embedding_models,
        sanitize_model_id,
    )

    indexes_dir = get_indexes_dir()
    available = get_available_embedding_models(indexes_dir)

    config = config_service.load()
    current_model_id = sanitize_model_id(config.rag.default_embedding_model)

    # Build lookup from sanitized model_id -> full HuggingFace path
    # so we can fill in model_name when index metadata doesn't have it.
    # Priority: built-in defaults < user config < current default (highest)
    known_full_names: Dict[str, str] = {}
    for full_name in DEFAULT_EMBEDDING_MODEL_CONFIGS:
        known_full_names[sanitize_model_id(full_name)] = full_name
    for full_name in config.rag.embedding_model_configs:
        known_full_names[sanitize_model_id(full_name)] = full_name
    known_full_names[current_model_id] = config.rag.default_embedding_model

    return EmbeddingModelsResponse(
        models=[
            EmbeddingModelInfo(
                model_id=m["model_id"],
                model_name=m.get("model_name") or known_full_names.get(m["model_id"]),
                index_count=m["index_count"],
                modules=m["modules"],
            )
            for m in available
        ],
        current=current_model_id,
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(config_service: ConfigServiceDep) -> ModelsResponse:
    """List available models from all configured providers."""
    from tensortruth.core.ollama import supports_thinking_levels
    from tensortruth.core.providers import ProviderRegistry

    registry = ProviderRegistry.get_instance()

    models = []
    has_any_models = False

    for provider in registry.providers:
        if provider.type == "ollama":
            # Query Ollama API for live model list with extra metadata
            try:
                base = provider.base_url.rstrip("/")
                response = requests.get(f"{base}/api/tags", timeout=10)
                response.raise_for_status()
                data = response.json()

                # Query loaded models for status
                loaded_names: set = set()
                try:
                    ps_resp = requests.get(f"{base}/api/ps", timeout=2)
                    if ps_resp.status_code == 200:
                        for rm in ps_resp.json().get("models", []):
                            loaded_names.add(rm.get("name", ""))
                except Exception:
                    pass  # If ps fails, all models show as unloaded

                for m in data.get("models", []):
                    name = m.get("name", "")
                    if not name:
                        continue
                    from tensortruth.core.ollama import get_model_info

                    info = get_model_info(name)
                    capabilities = list(info.get("capabilities", []))

                    if "thinking" in capabilities and supports_thinking_levels(name):
                        capabilities.append("thinking_levels")

                    models.append(
                        ModelInfo(
                            name=name,
                            provider_id=provider.id,
                            provider_type="ollama",
                            display_name=name,
                            size=m.get("size", 0),
                            modified_at=m.get("modified_at", ""),
                            capabilities=capabilities,
                            status="loaded" if name in loaded_names else "unloaded",
                        )
                    )
                    has_any_models = True

            except Exception as e:
                logger.warning(
                    "Failed to fetch models from Ollama provider '%s': %s",
                    provider.id,
                    e,
                )

        elif provider.type == "openai_compatible":
            # Return statically-configured models
            for m in provider.models:
                name = m.get("name", "")
                if not name:
                    continue
                models.append(
                    ModelInfo(
                        name=name,
                        provider_id=provider.id,
                        provider_type="openai_compatible",
                        display_name=m.get("display_name") or name,
                        capabilities=m.get("capabilities", []),
                    )
                )
                has_any_models = True

        elif provider.type == "llama_cpp":
            # Query llama.cpp router mode for dynamic model list
            try:
                from tensortruth.core.llama_cpp import (
                    format_display_name,
                )
                from tensortruth.core.llama_cpp import (
                    get_available_models as get_llama_cpp_models,
                )

                base = provider.base_url.rstrip("/")
                server_models = get_llama_cpp_models(base)

                # Build lookup from static config for capability enrichment
                static_lookup = {}
                for m in provider.models:
                    name = m.get("name", "")
                    if name:
                        static_lookup[name] = m

                for sm in server_models:
                    model_id = sm.get("id", "")
                    if not model_id:
                        continue
                    static = static_lookup.get(model_id, {})
                    caps = static.get("capabilities", [])
                    models.append(
                        ModelInfo(
                            name=model_id,
                            provider_id=provider.id,
                            provider_type="llama_cpp",
                            display_name=(
                                static.get("display_name")
                                or format_display_name(model_id)
                            ),
                            capabilities=caps,
                            status=sm.get("status", "unloaded"),
                        )
                    )
                    has_any_models = True

            except Exception as e:
                logger.warning(
                    "Failed to fetch models from llama.cpp provider '%s': %s",
                    provider.id,
                    e,
                )

    if not has_any_models and len(registry.providers) == 1:
        # Single Ollama provider that is unreachable — raise for backward compat
        ollama_url = registry.providers[0].base_url
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?",
        )

    return ModelsResponse(models=models)
