"""Ollama API interaction utilities."""

import logging
import os
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

# API base constant for backward compatibility
OLLAMA_API_BASE = "http://localhost:11434/api"


def get_ollama_url() -> str:
    """Get Ollama base URL with precedence.

    Priority:
    1. Environment variable (OLLAMA_HOST)
    2. Config file
    3. Default (http://localhost:11434)

    Returns:
        Ollama base URL string
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


def get_api_base() -> str:
    """Get the base API endpoint for raw requests.

    Returns:
        API base URL string
    """
    return f"{get_ollama_url()}/api"


def get_running_models() -> List[Dict[str, Any]]:
    """Get list of active models with VRAM usage.

    Equivalent to `ollama ps` command.

    Returns:
        List of running model dictionaries
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


def get_available_models() -> List[str]:
    """
    Get list of available Ollama models.
    Returns sorted list of model names, or empty list if Ollama is unavailable.
    """
    try:
        response = requests.get(f"{get_api_base()}/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data["models"]]
            return sorted(models)
    except Exception:
        pass

    return []


def get_running_models_detailed() -> List[Dict[str, Any]]:
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


def stop_model(model_name: str) -> bool:
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


def check_thinking_support(model_name: str) -> bool:
    """Check if a model supports thinking/reasoning tokens.

    This queries the Ollama API for the model's capabilities and checks
    if "thinking" is in the capabilities list.

    Args:
        model_name: The name of the model to check

    Returns:
        True if the model supports thinking tokens, False otherwise
    """
    try:
        response = requests.post(
            f"{get_api_base()}/show", json={"model": model_name}, timeout=2
        )
        if response.status_code == 200:
            data = response.json()
            capabilities = data.get("capabilities", [])
            return "thinking" in capabilities
    except Exception as e:
        logger.warning(f"Failed to check thinking support for {model_name}: {e}")

    return False


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed model information from Ollama.

    Queries the Ollama /api/show endpoint to get model metadata including
    context window size, parameter count, and capabilities.

    Args:
        model_name: The name of the model to query

    Returns:
        Dictionary with model info:
        - context_length: int (default 4096 if not found)
        - parameter_size: str or None (e.g., "8B")
        - capabilities: list of strings
        - family: str or None
    """
    result: Dict[str, Any] = {
        "context_length": 4096,  # Default fallback
        "parameter_size": None,
        "capabilities": [],
        "family": None,
    }

    try:
        response = requests.post(
            f"{get_api_base()}/show", json={"model": model_name}, timeout=2
        )
        if response.status_code == 200:
            data = response.json()

            # Extract model info from response
            model_info = data.get("model_info", {})

            # Context length - try multiple possible keys
            for key in model_info:
                if "context_length" in key.lower():
                    result["context_length"] = model_info[key]
                    break

            # Get details
            details = data.get("details", {})
            result["parameter_size"] = details.get("parameter_size")
            result["family"] = details.get("family")

            # Capabilities
            result["capabilities"] = data.get("capabilities", [])

    except Exception as e:
        logger.warning(f"Failed to get model info for {model_name}: {e}")

    return result


def pull_model(model_name: str, callback=None) -> bool:
    """Pull a model from Ollama repository.

    Args:
        model_name: Name of the model to pull (e.g., "llama3:8b")
        callback: Optional callback function for progress updates
                    Callback signature: callback(status: str, progress: float, message: str)

    Returns:
        True if pull was successful, False otherwise
    """
    ollama_url = get_ollama_url()
    pull_url = f"{ollama_url}/api/pull"

    try:
        response = requests.post(
            pull_url,
            json={"name": model_name},
            stream=True,
            timeout=300,  # 5 minute timeout for large model downloads
        )
        response.raise_for_status()

        # Process streaming response
        for line in response.iter_lines():
            if line:  # filter out keep-alive new lines
                try:
                    data = line.decode("utf-8")
                    if data.startswith("{"):  # JSON line
                        import json

                        status_data = json.loads(data)

                        if callback:
                            # Extract progress information
                            status = status_data.get("status", "unknown")
                            total = status_data.get("total", 0)
                            completed = status_data.get("completed", 0)

                            if total > 0 and completed > 0:
                                progress = completed / total
                            else:
                                progress = 0.0

                            callback(status, progress, data)
                except Exception as e:
                    logger.debug(f"Error processing pull status: {e}")

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to pull model {model_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error pulling model {model_name}: {e}")
        return False


def ensure_required_models_available() -> List[str]:
    """Check and pull required models if they're not available.

    Returns:
        List of models that were successfully pulled
    """
    try:
        from tensortruth.app_utils.config import load_config

        # Get required models from config
        config = load_config()
        required_models = [
            config.models.default_rag_model,
            config.models.default_agent_reasoning_model,
        ]

        # Remove duplicates while preserving order
        required_models = list(dict.fromkeys(required_models))

        # Get available models
        available_models = get_available_models()

        # Find missing models
        missing_models = [
            model for model in required_models if model not in available_models
        ]

        pulled_models = []

        if missing_models:
            logger.info(
                (
                    f"Found {len(missing_models)} required models that "
                    f"need to be pulled: {missing_models}"
                )
            )

            for model in missing_models:
                logger.info(f"Pulling model: {model}")
                if pull_model(model):
                    pulled_models.append(model)
                    logger.info(f"Successfully pulled: {model}")
                else:
                    logger.warning(f"Failed to pull: {model}")
        else:
            logger.info("All required models are already available")

        return pulled_models

    except Exception as e:
        logger.error(f"Error ensuring required models: {e}")
        return []
