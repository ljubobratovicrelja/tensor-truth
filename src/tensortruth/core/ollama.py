"""Ollama API interaction utilities."""

import logging

import requests

from tensortruth.app_utils.config import get_ollama_url

logger = logging.getLogger(__name__)


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
