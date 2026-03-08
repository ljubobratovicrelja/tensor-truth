"""llama.cpp router mode API interaction utilities."""

import logging
import re
from typing import Any, Dict, List
from urllib.parse import quote as url_quote

import requests

logger = logging.getLogger(__name__)


def get_available_models(base_url: str, timeout: int = 2) -> List[Dict[str, Any]]:
    """Get all models from a llama.cpp server in router mode.

    Queries ``GET /models`` and returns the model list.

    Returns:
        List of dicts with keys: id, status, in_cache, path.
        Empty list if the server is unreachable.
    """
    try:
        url = f"{base_url.rstrip('/')}/models"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return []
        data = resp.json()
        models = []
        for m in data.get("data", []):
            status_obj = m.get("status", {})
            if isinstance(status_obj, dict):
                status_val = status_obj.get("value", "unloaded")
            else:
                status_val = "unloaded"
            models.append(
                {
                    "id": m.get("id", ""),
                    "status": status_val,
                    "in_cache": m.get("in_cache", False),
                    "path": m.get("path", ""),
                }
            )
        return models
    except Exception as e:
        logger.warning("Failed to query llama.cpp models at %s: %s", base_url, e)
        return []


def get_loaded_models(base_url: str, timeout: int = 2) -> List[Dict[str, Any]]:
    """Get models that are currently loaded or loading."""
    all_models = get_available_models(base_url, timeout=timeout)
    return [m for m in all_models if m.get("status") in ("loaded", "loading")]


def load_model(base_url: str, model_id: str, timeout: int = 60) -> bool:
    """Load a model into VRAM via ``POST /models/load``."""
    try:
        url = f"{base_url.rstrip('/')}/models/load"
        resp = requests.post(url, json={"model": model_id}, timeout=timeout)
        return resp.status_code == 200
    except Exception as e:
        logger.error("Failed to load model %s: %s", model_id, e)
        return False


def unload_model(base_url: str, model_id: str, timeout: int = 10) -> bool:
    """Unload a model from VRAM via ``POST /models/unload``."""
    try:
        url = f"{base_url.rstrip('/')}/models/unload"
        resp = requests.post(url, json={"model": model_id}, timeout=timeout)
        return resp.status_code == 200
    except Exception as e:
        logger.error("Failed to unload model %s: %s", model_id, e)
        return False


def check_health(base_url: str, timeout: int = 2) -> bool:
    """Check if the llama.cpp server is healthy via ``GET /health``."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def check_capabilities(
    base_url: str, model_id: str = "", timeout: int = 5
) -> List[str]:
    """Detect model capabilities via ``GET /props``.

    In router mode, uses ``/props?model=<id>`` to query per-model
    properties.  Checks ``chat_template_tool_use`` first, then falls
    back to inspecting the ``chat_template`` Jinja source for tool
    patterns (``tool_call``, ``tools``).

    Args:
        base_url: llama.cpp server URL.
        model_id: Model ID for per-model queries in router mode.
        timeout: Request timeout (may trigger model auto-load).

    Returns:
        List of capability strings, e.g. ``["tools", "thinking"]``.
    """
    caps: List[str] = []
    try:
        base = base_url.rstrip("/")
        # Try per-model props first (router mode), fall back to global
        if model_id:
            url = f"{base}/props?model={url_quote(model_id)}"
        else:
            url = f"{base}/props"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return []  # Assume no capabilities when unavailable
        data = resp.json()
        # Router global response — no per-model info
        if data.get("role") == "router":
            return []  # Assume no capabilities when unavailable
        # Check dedicated tool template field
        if data.get("chat_template_tool_use"):
            caps.append("tools")
        else:
            # Inspect the Jinja chat template for tool patterns
            tmpl = data.get("chat_template", "")
            tmpl_lower = tmpl.lower()
            if "tool_call" in tmpl_lower or "tools" in tmpl_lower:
                caps.append("tools")
        # Thinking support heuristic
        tmpl = data.get("chat_template", "")
        tmpl_lower = tmpl.lower()
        if "think" in tmpl_lower or data.get("reasoning_format"):
            caps.append("thinking")
    except Exception as e:
        logger.debug("Failed to query llama.cpp /props at %s: %s", base_url, e)
        return []  # Assume no capabilities on failure
    return caps


def format_display_name(model_id: str) -> str:
    """Create a human-friendly display name from a llama.cpp model ID.

    Examples:
        "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M" -> "gemma-3-4b-it Q4_K_M"
        "Qwen3-8B-Q4_K_M.gguf" -> "Qwen3-8B-Q4_K_M"
        "models/Qwen3-8B-Q4_K_M.gguf" -> "Qwen3-8B-Q4_K_M"
    """
    name = model_id

    # Strip repo prefix (e.g. "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M")
    if ":" in name:
        repo_part, quant = name.rsplit(":", 1)
        # Take the part after the last /
        base = repo_part.rsplit("/", 1)[-1]
        # Remove -GGUF suffix
        base = re.sub(r"-GGUF$", "", base, flags=re.IGNORECASE)
        return f"{base} {quant}"

    # Strip directory prefixes
    name = name.rsplit("/", 1)[-1]
    # Strip .gguf extension
    name = re.sub(r"\.gguf$", "", name, flags=re.IGNORECASE)

    return name
