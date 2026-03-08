"""Provider management routes."""

import ipaddress
import logging
import socket
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests as _requests
from fastapi import APIRouter, HTTPException

from tensortruth.api.deps import ConfigServiceDep
from tensortruth.api.schemas.provider import (
    DiscoveredServer,
    DiscoverResponse,
    ProviderCreateRequest,
    ProviderListResponse,
    ProviderResponse,
    ProviderTestRequest,
    ProviderTestResponse,
    ProviderUpdateRequest,
)
from tensortruth.app_utils.config_schema import ProviderConfig

logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_provider_url(url: str) -> None:
    """Validate that a provider URL doesn't target internal/dangerous networks."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname")

    # Allow localhost and common local development
    if hostname in ("localhost", "127.0.0.1", "::1"):
        return  # Local development is fine

    try:
        # Resolve hostname and check for private/reserved IPs
        for info in socket.getaddrinfo(hostname, None):
            addr = ipaddress.ip_address(info[4][0])
            if addr.is_private or addr.is_reserved:
                raise ValueError(f"URL targets private/reserved address: {hostname}")
    except socket.gaierror:
        pass  # Let the actual request fail if DNS doesn't resolve


def _mask_api_key(key: str) -> str:
    return "***" if key else ""


def _probe_provider(ptype: str, base_url: str, api_key: str = "") -> dict:
    """Probe a provider URL for connectivity and models.

    Returns dict with keys: connected (bool), models (list[str]).
    """
    _validate_provider_url(base_url)

    base = base_url.rstrip("/")
    result: Dict = {"connected": False, "models": []}

    try:
        if ptype == "ollama":
            resp = _requests.get(f"{base}/api/tags", timeout=1, allow_redirects=False)
            if resp.status_code == 200:
                result["connected"] = True
                data = resp.json()
                result["models"] = sorted(
                    m["name"] for m in data.get("models", []) if m.get("name")
                )

        elif ptype == "llama_cpp":
            # Health check first
            health = _requests.get(f"{base}/health", timeout=1, allow_redirects=False)
            if health.status_code == 200:
                result["connected"] = True
                try:
                    models_resp = _requests.get(
                        f"{base}/models", timeout=1, allow_redirects=False
                    )
                    if models_resp.status_code == 200:
                        data = models_resp.json()
                        model_list = (
                            data.get("data", data) if isinstance(data, dict) else data
                        )
                        if isinstance(model_list, list):
                            result["models"] = [
                                m.get("id", "") for m in model_list if m.get("id")
                            ]
                except Exception:
                    pass

        elif ptype == "openai_compatible":
            headers: Dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            resp = _requests.get(
                f"{base}/models", headers=headers, timeout=1, allow_redirects=False
            )
            if resp.status_code == 200:
                result["connected"] = True
                try:
                    data = resp.json()
                    model_list = data.get("data", []) if isinstance(data, dict) else []
                    if isinstance(model_list, list):
                        result["models"] = [
                            m.get("id", "") for m in model_list if m.get("id")
                        ]
                except Exception:
                    pass

    except Exception:
        pass

    return result


@router.get("/", response_model=ProviderListResponse)
def list_providers(config_service: ConfigServiceDep) -> ProviderListResponse:
    """List all configured providers with live connectivity status."""
    config = config_service.load()
    providers: List[ProviderResponse] = []

    for p in config.providers:
        probe = _probe_provider(p.type, p.base_url, p.api_key)
        providers.append(
            ProviderResponse(
                id=p.id,
                type=p.type,
                base_url=p.base_url,
                api_key=_mask_api_key(p.api_key),
                timeout=p.timeout,
                models=p.models,
                status="connected" if probe["connected"] else "unreachable",
                model_count=len(probe["models"]),
            )
        )

    return ProviderListResponse(providers=providers)


@router.post("/", response_model=ProviderResponse, status_code=201)
def add_provider(
    body: ProviderCreateRequest, config_service: ConfigServiceDep
) -> ProviderResponse:
    """Add a new provider."""
    try:
        _validate_provider_url(body.base_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    provider = ProviderConfig(
        id=body.id,
        type=body.type,
        base_url=body.base_url,
        api_key=body.api_key or "",
        timeout=body.timeout or 300,
        models=body.models or [],
    )

    try:
        config_service.add_provider(provider)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    _reset_provider_registry()

    probe = _probe_provider(provider.type, provider.base_url, provider.api_key)
    return ProviderResponse(
        id=provider.id,
        type=provider.type,
        base_url=provider.base_url,
        api_key=_mask_api_key(provider.api_key),
        timeout=provider.timeout,
        models=provider.models,
        status="connected" if probe["connected"] else "unreachable",
        model_count=len(probe["models"]),
    )


@router.patch("/{provider_id}", response_model=ProviderResponse)
def update_provider(
    provider_id: str,
    body: ProviderUpdateRequest,
    config_service: ConfigServiceDep,
) -> ProviderResponse:
    """Update an existing provider."""
    updates: Dict[str, Any] = {}
    if body.base_url is not None:
        try:
            _validate_provider_url(body.base_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        updates["base_url"] = body.base_url
    if body.api_key is not None and body.api_key != "***":
        updates["api_key"] = body.api_key
    if body.timeout is not None:
        updates["timeout"] = body.timeout
    if body.models is not None:
        updates["models"] = body.models

    try:
        config = config_service.update_provider(provider_id, **updates)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    _reset_provider_registry()

    # Find the updated provider
    provider = None
    for p in config.providers:
        if p.id == provider_id:
            provider = p
            break

    if provider is None:
        raise HTTPException(
            status_code=404, detail=f"Provider '{provider_id}' not found"
        )

    probe = _probe_provider(provider.type, provider.base_url, provider.api_key)
    return ProviderResponse(
        id=provider.id,
        type=provider.type,
        base_url=provider.base_url,
        api_key=_mask_api_key(provider.api_key),
        timeout=provider.timeout,
        models=provider.models,
        status="connected" if probe["connected"] else "unreachable",
        model_count=len(probe["models"]),
    )


@router.delete("/{provider_id}", status_code=200)
def remove_provider(provider_id: str, config_service: ConfigServiceDep) -> dict:
    """Remove a provider. Refuses if it's the last one."""
    # Check if provider exists before attempting removal
    config = config_service.load()
    if not any(p.id == provider_id for p in config.providers):
        raise HTTPException(
            status_code=404, detail=f"Provider '{provider_id}' not found"
        )

    try:
        config_service.remove_provider(provider_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _reset_provider_registry()
    return {"status": "removed", "id": provider_id}


@router.post("/test", response_model=ProviderTestResponse)
def test_provider(body: ProviderTestRequest) -> ProviderTestResponse:
    """Test connectivity to an arbitrary provider URL without saving."""
    try:
        _validate_provider_url(body.base_url)
    except ValueError as e:
        return ProviderTestResponse(
            success=False,
            message=str(e),
        )

    probe = _probe_provider(body.type, body.base_url, body.api_key or "")

    if probe["connected"]:
        model_count = len(probe["models"])
        return ProviderTestResponse(
            success=True,
            message=(
                f"Connected successfully. Found {model_count} "
                f"model{'s' if model_count != 1 else ''}."
            ),
            models=probe["models"],
        )
    else:
        return ProviderTestResponse(
            success=False,
            message=f"Could not connect to {body.base_url}",
        )


@router.get("/discover", response_model=DiscoverResponse)
def discover_servers(config_service: ConfigServiceDep) -> DiscoverResponse:
    """Auto-discover local LLM servers not yet configured."""
    config = config_service.load()
    configured_urls = {p.base_url.rstrip("/") for p in config.providers}

    candidates = [
        ("ollama", "http://localhost:11434", "ollama"),
        ("llama_cpp", "http://localhost:8080", "llama-cpp"),
    ]

    servers: List[DiscoveredServer] = []
    for ptype, url, suggested_id in candidates:
        if url in configured_urls:
            continue
        probe = _probe_provider(ptype, url)
        if probe["connected"]:
            servers.append(
                DiscoveredServer(
                    type=ptype,
                    base_url=url,
                    suggested_id=suggested_id,
                    model_count=len(probe["models"]),
                    models=probe["models"],
                )
            )

    return DiscoverResponse(servers=servers)


def _reset_provider_registry() -> None:
    """Reset the ProviderRegistry singleton so it reloads from config."""
    try:
        from tensortruth.core.providers import ProviderRegistry

        ProviderRegistry.reset()
    except Exception as e:
        logger.warning("Failed to reset ProviderRegistry: %s", e)
