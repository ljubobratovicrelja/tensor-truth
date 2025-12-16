"""Ollama API interaction utilities."""

import logging
import os

import requests

logger = logging.getLogger(__name__)

# API base constant for backward compatibility
OLLAMA_API_BASE = "http://localhost:11434/api"


def get_ollama_url() -> str:
    """
    Get Ollama base URL with precedence:
    1. Environment variable (OLLAMA_HOST)
    2. Config file
    3. Default (http://localhost:11434)
    """
    # 1. Check Environment Variable (highest priority)
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        # Handle cases where OLLAMA_HOST might be just "0.0.0.0:11434"
        if not env_host.startswith("http"):
            return f"http://{env_host}".rstrip("/")
        return env_host.rstrip("/")

    # 2. Check Config File
    try:
        from tensortruth.app_utils.config import load_config

        config = load_config()
        return config.ollama.base_url.rstrip("/")
    except Exception:
        pass

    # 3. Return Default
    return "http://localhost:11434"


def get_api_base():
    """Get the base API endpoint for raw requests."""
    return f"{get_ollama_url()}/api"


def get_running_models():
    """
    Equivalent to `ollama ps`. Returns list of active models with VRAM usage.
    """
    try:
        response = requests.get(f"{get_api_base()}/ps", timeout=2)
        if response.status_code == 200:
            data = response.json()
            # simplify data for UI
            active = []
            for m in data.get("models", []):
                active.append(
                    {
                        "name": m["name"],
                        "size_vram": f"{m.get('size_vram', 0) / 1024**3:.1f} GB",
                        "expires": m.get("expires_at", "Unknown"),
                    }
                )
            return active
    except Exception:
        return []  # Server likely down
    return []


def get_available_models():
    """
    Get list of available Ollama models.
    Returns sorted list of model names, or default fallback if unavailable.
    """
    try:
        response = requests.get(f"{get_api_base()}/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data["models"]]
            return sorted(models)
    except Exception:
        pass
    return ["deepseek-r1:8b"]  # Default fallback


def get_running_models_detailed():
    """
    Get detailed running model information (raw API response).
    Returns list of model dictionaries with full details.
    """
    try:
        response = requests.get(f"{get_api_base()}/ps", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
    except Exception:
        pass
    return []


def stop_model(model_name):
    """
    Forces a model to unload immediately by setting keep_alive to 0.
    """
    try:
        # We send a dummy request with keep_alive=0 to trigger unload
        payload = {"model": model_name, "keep_alive": 0}
        # We use /api/chat as the generic endpoint
        requests.post(f"{get_api_base()}/chat", json=payload, timeout=2)
        return True
    except Exception as e:
        logger.error(f"Failed to stop {model_name}: {e}")
        return False
