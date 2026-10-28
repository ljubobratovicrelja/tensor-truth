"""Modules and models listing endpoints."""

import asyncio
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
    """Information about a knowledge module.

    ``description``/``doc_count`` come from the cached module catalog.
    ``loaded`` is only set when the request names a session.
    """

    name: str
    display_name: str = ""
    doc_type: str = "unknown"
    sort_order: int = 4
    description: str = ""
    doc_count: int = 0
    loaded: bool | None = None


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
    # Reasoning-effort values the endpoint accepted for this model, when it
    # has been probed (see /providers/{id}/models/{name}/reasoning-levels).
    reasoning_levels: List[str] = []


class ModelsResponse(BaseModel):
    """Response for listing available models."""

    models: List[ModelInfo]


@router.get("/modules", response_model=ModulesResponse)
async def list_modules(
    config_service: ConfigServiceDep,
    session_id: str | None = None,
) -> ModulesResponse:
    """List available knowledge modules for the configured embedding model.

    Modules are vector indexes stored in ~/.tensortruth/indexes/{model_id}/.
    Returns modules sorted by type (Books, Papers, Libraries, Other), with
    their catalog description, document count and — when ``session_id`` is
    given — whether the session currently has that module loaded.
    """
    from tensortruth.services.module_catalog import (
        DOC_TYPE_SORT_ORDER,
        load_module_catalog,
    )

    config = config_service.load()
    # indexes_dir is passed explicitly so the route keeps honouring an
    # overridden indexes location (and its existing tests' patch point).
    catalog = load_module_catalog(config, indexes_dir=get_indexes_dir())

    loaded_names: set[str] = set()
    if session_id:
        from tensortruth.api.deps import get_session_service

        try:
            session = get_session_service().get_session(
                session_id, get_session_service().load()
            )
            if session:
                loaded_names = set(session.get("modules") or [])
        except Exception:
            logger.debug("Could not load session %s for loaded flags", session_id)

    modules = [
        ModuleInfo(
            name=entry.name,
            display_name=entry.display_name,
            doc_type=entry.doc_type,
            sort_order=DOC_TYPE_SORT_ORDER.get(entry.doc_type, 4),
            description=entry.description,
            doc_count=entry.doc_count,
            loaded=(entry.name in loaded_names) if session_id else None,
        )
        for entry in catalog
    ]

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


def _fetch_ollama_models_sync(provider) -> List[ModelInfo]:
    """Synchronously fetch models from an Ollama provider (blocking worker)."""
    from tensortruth.core.ollama import get_model_info, supports_thinking_levels

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

    models = []
    for m in data.get("models", []):
        name = m.get("name", "")
        if not name:
            continue
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
    return models


def _fetch_openai_compatible_models_sync(provider) -> List[str]:
    """Synchronously fetch dynamic model names from an OpenAI-compatible provider."""
    base = provider.base_url.rstrip("/")
    headers = {}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"

    # Try /v1/models first, fall back to /models
    resp = None
    for models_path in (f"{base}/v1/models", f"{base}/models"):
        try:
            resp = requests.get(
                models_path,
                headers=headers,
                timeout=5,
                allow_redirects=False,
            )
            if resp.status_code == 200:
                break
        except Exception:
            continue

    if resp is not None and resp.status_code == 200:
        data = resp.json()
        model_list = data.get("data", []) if isinstance(data, dict) else []
        if isinstance(model_list, list):
            return [m.get("id", "") for m in model_list if m.get("id")]
    return []


def _fetch_llama_cpp_models_sync(provider):
    """Synchronously fetch models from a llama.cpp provider."""
    from tensortruth.core.llama_cpp import get_available_models as get_llama_cpp_models

    base = provider.base_url.rstrip("/")
    return get_llama_cpp_models(base)


@router.get("/models", response_model=ModelsResponse)
async def list_models(config_service: ConfigServiceDep) -> ModelsResponse:
    """List available models from all configured providers."""
    from tensortruth.core.providers import ProviderRegistry
    from tensortruth.core.reasoning_levels import levels_for_model

    registry = ProviderRegistry.get_instance()

    models = []
    has_any_models = False

    for provider in registry.providers:
        if provider.type == "ollama":
            # Query Ollama API for live model list with extra metadata
            try:
                provider_models = await asyncio.to_thread(
                    _fetch_ollama_models_sync, provider
                )
                models.extend(provider_models)
                if provider_models:
                    has_any_models = True
            except Exception as e:
                logger.warning(
                    "Failed to fetch models from Ollama provider '%s': %s",
                    provider.id,
                    e,
                )

        elif provider.type == "openai_compatible":
            # Build a lookup from statically-configured models for capability enrichment
            static_lookup = {
                m.get("name", ""): m for m in provider.models if m.get("name")
            }

            # Prefer dynamic discovery; fall back to static-only list if server is unreachable
            dynamic_names: list[str] = []
            try:
                dynamic_names = await asyncio.to_thread(
                    _fetch_openai_compatible_models_sync, provider
                )
            except Exception as e:
                logger.warning(
                    "Failed to fetch models from openai_compatible provider '%s': %s",
                    provider.id,
                    e,
                )

            # Use dynamic list if available, otherwise fall back to static config
            names_to_use = (
                dynamic_names if dynamic_names else list(static_lookup.keys())
            )
            for name in names_to_use:
                if not name:
                    continue
                static = static_lookup.get(name, {})
                default_caps = getattr(provider, "default_capabilities", []) or []
                caps = static.get("capabilities", []) or default_caps
                models.append(
                    ModelInfo(
                        name=name,
                        provider_id=provider.id,
                        provider_type="openai_compatible",
                        display_name=static.get("display_name") or name,
                        capabilities=caps,
                        reasoning_levels=levels_for_model(provider.id, name),
                    )
                )
                has_any_models = True

        elif provider.type == "llama_cpp":
            # Query llama.cpp router mode for dynamic model list
            try:
                from tensortruth.core.llama_cpp import (
                    format_display_name,
                )

                server_models = await asyncio.to_thread(
                    _fetch_llama_cpp_models_sync, provider
                )

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
