"""Ollama API interaction utilities."""

import logging
import os
import uuid
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import requests

if TYPE_CHECKING:
    from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)

# Module-level singletons for LLM instances.
# Keyed by (model, base_url, context_window) so they are replaced when
# session configuration changes.  All instances always set num_ctx in
# additional_kwargs to prevent Ollama from reloading the model when the
# context size differs between calls.
_orchestrator_llm_instance: Optional[Any] = None
_orchestrator_llm_key: Optional[Tuple[str, str, int]] = None

_tool_llm_instance: Optional[Any] = None
_tool_llm_key: Optional[Tuple[str, str, int]] = None


_DEFAULT_OLLAMA_URL = "http://localhost:11434"


def get_ollama_url() -> str:
    """Get Ollama base URL with precedence.

    Priority:
    1. Config file — if explicitly set to a non-default value (app-specific intent)
    2. Environment variable (OLLAMA_HOST — generic system-wide fallback)
    3. Config file default / hardcoded default (http://localhost:11434)

    Returns:
        Ollama base URL string
    """
    # 1. Check Config File (highest priority when explicitly configured)
    config_url = None
    try:
        from tensortruth.app_utils.config import load_config

        config = load_config()
        config_url = config.ollama.base_url.rstrip("/")
    except Exception:
        pass

    if config_url and config_url != _DEFAULT_OLLAMA_URL:
        return config_url

    # 2. Check Environment Variable (generic Ollama fallback)
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        if not env_host.startswith("http"):
            return f"http://{env_host}".rstrip("/")
        return env_host.rstrip("/")

    # 3. Return config default (if loaded) or hardcoded default
    return config_url or _DEFAULT_OLLAMA_URL


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


@lru_cache(maxsize=32)
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


@lru_cache(maxsize=32)
def check_tool_call_support(model_name: str) -> bool:
    """Check if a model supports native tool-calling.

    This queries the Ollama API for the model's capabilities and checks
    if "tools" is in the capabilities list.

    Args:
        model_name: The name of the model to check

    Returns:
        True if the model supports tool-calling, False otherwise
    """
    try:
        response = requests.post(
            f"{get_api_base()}/show", json={"model": model_name}, timeout=2
        )
        if response.status_code == 200:
            data = response.json()
            capabilities = data.get("capabilities", [])
            return "tools" in capabilities
    except Exception as e:
        logger.warning(f"Failed to check tool call support for {model_name}: {e}")

    return False


def _patch_parallel_tool_calls(llm: Any) -> None:
    """Patch an Ollama LLM instance for proper parallel tool call support.

    The upstream LlamaIndex Ollama adapter sets ``tool_id=tool_name`` in
    ``get_tool_calls_from_response``, making multiple calls to the same
    tool indistinguishable.  This patches the method on the instance to
    generate unique UUIDs for each tool call instead.
    """
    import types

    from llama_index.core.base.llms.types import ToolCallBlock
    from llama_index.core.llms.llm import ToolSelection

    def _get_tool_calls_from_response(
        self: Any,
        response: Any,
        error_on_no_tool_call: bool = True,
    ) -> list:
        tool_calls = [
            block
            for block in response.message.blocks
            if isinstance(block, ToolCallBlock)
        ]
        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)}."
                )
            return []

        tool_selections = []
        for tc in tool_calls:
            tool_selections.append(
                ToolSelection(
                    tool_id=tc.tool_call_id or f"call_{uuid.uuid4().hex[:12]}",
                    tool_name=tc.tool_name,
                    tool_kwargs=cast(dict, tc.tool_kwargs),
                )
            )
        return tool_selections

    # Pydantic v2 blocks __setattr__ for unknown fields, so we must
    # bypass it with object.__setattr__ to patch the instance method.
    object.__setattr__(
        llm,
        "get_tool_calls_from_response",
        types.MethodType(_get_tool_calls_from_response, llm),
    )


def get_orchestrator_llm(
    model: str,
    base_url: str,
    context_window: int = 16384,
) -> "Ollama":
    """Get or create a cached Ollama LLM instance for orchestrator use.

    Returns an Ollama instance with thinking disabled, intended for fast
    tool-calling decisions in the orchestrator agent.  Also suitable for
    query condensation and other low-temperature auxiliary tasks.

    The instance is cached as a module-level singleton keyed by
    ``(model, base_url, context_window)``.  ``num_ctx`` is always set in
    ``additional_kwargs`` so that Ollama never sees a context-size change
    between requests (which would trigger a costly model reload).

    Args:
        model: Ollama model name (e.g., "qwen3:32b").
        base_url: Ollama server base URL.
        context_window: Context window size for the model.

    Returns:
        Cached Ollama LLM instance with thinking=False.
    """
    global _orchestrator_llm_instance, _orchestrator_llm_key

    key = (model, base_url, context_window)

    if _orchestrator_llm_instance is not None and _orchestrator_llm_key == key:
        return _orchestrator_llm_instance

    from llama_index.llms.ollama import Ollama

    logger.info(
        "Creating orchestrator LLM singleton: model=%s, base_url=%s, "
        "context_window=%d",
        model,
        base_url,
        context_window,
    )

    _orchestrator_llm_instance = Ollama(
        model=model,
        base_url=base_url,
        temperature=0.2,
        context_window=context_window,
        thinking=False,
        request_timeout=120.0,
        additional_kwargs={"num_ctx": context_window, "num_predict": -1},
    )
    _patch_parallel_tool_calls(_orchestrator_llm_instance)
    _orchestrator_llm_key = key

    return _orchestrator_llm_instance


def get_tool_llm(
    model: str,
    base_url: str,
    context_window: int = 16384,
) -> "Ollama":
    """Get or create a cached thinking-enabled Ollama LLM for tool/synthesis use.

    Returns an Ollama instance with thinking enabled (when the model supports
    it), intended for synthesis, RAG generation, and other content-producing
    calls.  Shares the same ``num_ctx`` as the orchestrator LLM so Ollama
    keeps the model loaded without reloading.

    Args:
        model: Ollama model name.
        base_url: Ollama server base URL.
        context_window: Context window size for the model.

    Returns:
        Cached Ollama LLM instance with thinking enabled (if supported).
    """
    global _tool_llm_instance, _tool_llm_key

    key = (model, base_url, context_window)

    if _tool_llm_instance is not None and _tool_llm_key == key:
        return _tool_llm_instance

    from llama_index.llms.ollama import Ollama

    thinking_supported = check_thinking_support(model)

    logger.info(
        "Creating tool LLM singleton: model=%s, base_url=%s, "
        "context_window=%d, thinking=%s",
        model,
        base_url,
        context_window,
        thinking_supported,
    )

    _tool_llm_instance = Ollama(
        model=model,
        base_url=base_url,
        temperature=0.7,
        context_window=context_window,
        thinking=thinking_supported,
        request_timeout=120.0,
        additional_kwargs={"num_ctx": context_window, "num_predict": -1},
    )
    _tool_llm_key = key

    return _tool_llm_instance


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
            config.llm.default_model,
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
