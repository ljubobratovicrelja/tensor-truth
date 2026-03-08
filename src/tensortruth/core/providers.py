"""Multi-provider LLM abstraction layer.

Manages configured LLM providers (Ollama, OpenAI-compatible, llama.cpp)
and provides a unified factory for creating LlamaIndex LLM instances.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelReference:
    """Fully qualified model reference across providers."""

    provider_id: str  # "ollama", "vllm-server"
    model_name: str  # "qwen3:32b"
    display_name: str  # "qwen3:32b" or custom display name
    provider_type: str  # "ollama" | "openai_compatible" | "llama_cpp"
    base_url: str
    api_key: str = field(default="", repr=False)
    capabilities: List[str] = field(default_factory=list)
    context_window: int = 4096
    timeout: int = 300


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------


class ProviderRegistry:
    """Manages all configured providers and their models."""

    _instance: Optional["ProviderRegistry"] = None

    def __init__(self) -> None:
        from tensortruth.app_utils.config_schema import ProviderConfig

        self._providers: List[ProviderConfig] = []
        # Cache for dynamically detected capabilities (keyed by model_name)
        self._caps_cache: Dict[str, List[str]] = {}

    @classmethod
    def get_instance(cls) -> "ProviderRegistry":
        """Get or create the singleton registry."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_from_config()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing or config reload)."""
        cls._instance = None

    def _load_from_config(self) -> None:
        """Load providers from the application config.

        Environment variable references (``${VAR}``) in provider
        ``api_key`` fields are expanded here at runtime so that the
        config object retains the raw ``${VAR}`` reference and
        ``save_config()`` never writes secrets to disk.
        """
        try:
            from tensortruth.app_utils.config import _expand_env_vars, load_config

            config = load_config()
            self._providers = list(config.providers)
            # Expand env vars in api_key for runtime use only
            for provider in self._providers:
                if provider.api_key:
                    provider.api_key = _expand_env_vars(provider.api_key)
        except Exception:
            from tensortruth.app_utils.config_schema import ProviderConfig

            self._providers = [
                ProviderConfig(
                    id="ollama",
                    type="ollama",
                    base_url="http://localhost:11434",
                )
            ]

    def reload(self) -> None:
        """Reload providers from config."""
        self._caps_cache = {}
        self._load_from_config()

    @property
    def providers(self):
        return self._providers

    def get_provider(self, provider_id: str):
        """Get a provider by ID."""
        for p in self._providers:
            if p.id == provider_id:
                return p
        return None

    def get_default_ollama_provider(self):
        """Get the first Ollama-type provider."""
        for p in self._providers:
            if p.type == "ollama":
                return p
        return None

    def get_all_models(self) -> List[ModelReference]:
        """Aggregate models from all providers.

        Ollama providers: queries ``/api/tags`` + ``/api/show`` for capabilities.
        openai_compatible providers: returns static model list from config.
        llama_cpp providers: queries ``GET /models`` for dynamic discovery.
        """
        all_models: List[ModelReference] = []

        for provider in self._providers:
            if provider.type == "ollama":
                all_models.extend(self._get_ollama_models(provider))
            elif provider.type == "openai_compatible":
                all_models.extend(self._get_static_models(provider))
            elif provider.type == "llama_cpp":
                all_models.extend(self._get_llama_cpp_models(provider))

        return all_models

    def _get_ollama_models(self, provider) -> List[ModelReference]:
        """Query Ollama API for available models."""
        import requests

        models: List[ModelReference] = []
        try:
            base = provider.base_url.rstrip("/")
            resp = requests.get(f"{base}/api/tags", timeout=2)
            if resp.status_code != 200:
                return models
            data = resp.json()

            for m in data.get("models", []):
                name = m.get("name", "")
                if not name:
                    continue
                info = _get_ollama_model_info(base, name)
                models.append(
                    ModelReference(
                        provider_id=provider.id,
                        model_name=name,
                        display_name=name,
                        provider_type="ollama",
                        base_url=base,
                        capabilities=info.get("capabilities", []),
                        context_window=info.get("context_length", 4096),
                        timeout=provider.timeout,
                    )
                )
        except Exception as e:
            logger.warning("Failed to query Ollama models for %s: %s", provider.id, e)

        return models

    def _get_static_models(self, provider) -> List[ModelReference]:
        """Return statically-configured models for an openai_compatible provider."""
        models: List[ModelReference] = []
        for m in provider.models:
            name = m.get("name", "")
            if not name:
                continue
            models.append(
                ModelReference(
                    provider_id=provider.id,
                    model_name=name,
                    display_name=m.get("display_name") or name,
                    provider_type="openai_compatible",
                    base_url=provider.base_url.rstrip("/"),
                    api_key=provider.api_key,
                    capabilities=m.get("capabilities", []),
                    context_window=m.get("context_window", 4096),
                    timeout=provider.timeout,
                )
            )
        return models

    def _get_llama_cpp_models(self, provider) -> List[ModelReference]:
        """Query llama.cpp router mode for available models."""
        from tensortruth.core.llama_cpp import format_display_name, get_available_models

        models: List[ModelReference] = []
        try:
            base = provider.base_url.rstrip("/")
            server_models = get_available_models(base)

            # Build lookup from static config for capability enrichment
            static_lookup: Dict[str, Dict[str, Any]] = {}
            for m in provider.models:
                name = m.get("name", "")
                if name:
                    static_lookup[name] = m

            for sm in server_models:
                model_id = sm.get("id", "")
                if not model_id:
                    continue
                static = static_lookup.get(model_id, {})
                models.append(
                    ModelReference(
                        provider_id=provider.id,
                        model_name=model_id,
                        display_name=static.get("display_name")
                        or format_display_name(model_id),
                        provider_type="llama_cpp",
                        base_url=base,
                        api_key=provider.api_key,
                        capabilities=static.get("capabilities", []),
                        context_window=static.get("context_window", 4096),
                        timeout=provider.timeout,
                    )
                )
        except Exception as e:
            logger.warning(
                "Failed to query llama.cpp models for %s: %s", provider.id, e
            )

        return models

    def resolve_model(
        self, model_name: str, provider_id: Optional[str] = None
    ) -> ModelReference:
        """Resolve a model name to a full ModelReference.

        If ``provider_id`` is given, looks only in that provider.
        Otherwise searches Ollama providers first, then others (backward compat).

        If the model cannot be found via API / static config, a best-effort
        ModelReference is constructed from the provider config so that LLM
        creation can still proceed.
        """
        if provider_id:
            provider = self.get_provider(provider_id)
            if provider is None:
                # Unknown provider — fall back to first Ollama
                logger.warning(
                    "Provider '%s' not found, falling back to default", provider_id
                )
                provider = self.get_default_ollama_provider()
                if provider is None:
                    return self._fallback_model_ref(model_name, provider_id)
            return self._resolve_for_provider(model_name, provider)

        # Search non-Ollama providers first (they have definitive model lists),
        # then fall back to Ollama (which accepts any model name).
        for provider in self._providers:
            if provider.type == "llama_cpp":
                try:
                    from tensortruth.core.llama_cpp import get_available_models

                    base = provider.base_url.rstrip("/")
                    for sm in get_available_models(base):
                        if sm.get("id") == model_name:
                            return self._resolve_for_provider(model_name, provider)
                except Exception:
                    pass
            elif provider.type == "openai_compatible":
                for m in provider.models:
                    if m.get("name") == model_name:
                        return self._resolve_for_provider(model_name, provider)

        # Fall back to Ollama (accepts any model name without validation)
        for provider in self._providers:
            if provider.type == "ollama":
                return self._resolve_for_provider(model_name, provider)

        return self._fallback_model_ref(model_name, "ollama")

    def _build_llama_cpp_ref(
        self,
        provider,
        model_name: str,
        base: str,
        static: Dict[str, Any],
        caps_key: str,
    ) -> ModelReference:
        """Build a ModelReference for a llama.cpp model.

        Handles capability detection/caching and static config enrichment.
        """
        from tensortruth.core.llama_cpp import check_capabilities, format_display_name

        caps = static.get("capabilities", [])
        if not caps:
            cached = self._caps_cache.get(caps_key)
            if cached is not None:
                caps = cached
            else:
                caps = check_capabilities(base, model_name)
                self._caps_cache[caps_key] = caps
        return ModelReference(
            provider_id=provider.id,
            model_name=model_name,
            display_name=static.get("display_name") or format_display_name(model_name),
            provider_type="llama_cpp",
            base_url=base,
            api_key=provider.api_key,
            capabilities=caps,
            context_window=static.get("context_window", 4096),
            timeout=provider.timeout,
        )

    def _resolve_for_provider(self, model_name: str, provider) -> ModelReference:
        """Resolve model for a specific provider."""
        if provider.type == "ollama":
            base = provider.base_url.rstrip("/")
            info = _get_ollama_model_info(base, model_name)
            return ModelReference(
                provider_id=provider.id,
                model_name=model_name,
                display_name=model_name,
                provider_type="ollama",
                base_url=base,
                capabilities=info.get("capabilities", []),
                context_window=info.get("context_length", 4096),
                timeout=provider.timeout,
            )
        elif provider.type == "llama_cpp":
            base = provider.base_url.rstrip("/")
            caps_key = f"{provider.id}:{model_name}"

            # Look up static config once
            static: Dict[str, Any] = {}
            for m in provider.models:
                if m.get("name") == model_name:
                    static = m
                    break

            return self._build_llama_cpp_ref(
                provider, model_name, base, static, caps_key
            )
        else:
            # openai_compatible — look in static model list
            for m in provider.models:
                if m.get("name") == model_name:
                    return ModelReference(
                        provider_id=provider.id,
                        model_name=model_name,
                        display_name=m.get("display_name") or model_name,
                        provider_type="openai_compatible",
                        base_url=provider.base_url.rstrip("/"),
                        api_key=provider.api_key,
                        capabilities=m.get("capabilities", []),
                        context_window=m.get("context_window", 4096),
                        timeout=provider.timeout,
                    )
            # Model not in static list — still allow it
            return ModelReference(
                provider_id=provider.id,
                model_name=model_name,
                display_name=model_name,
                provider_type="openai_compatible",
                base_url=provider.base_url.rstrip("/"),
                api_key=provider.api_key,
                timeout=provider.timeout,
            )

    def _fallback_model_ref(self, model_name: str, provider_id: str) -> ModelReference:
        """Construct a best-effort ModelReference for unknown setups."""
        return ModelReference(
            provider_id=provider_id,
            model_name=model_name,
            display_name=model_name,
            provider_type="ollama",
            base_url="http://localhost:11434",
        )

    def check_tool_support(self, model_ref: ModelReference) -> bool:
        """Check if a model supports tool calling."""
        if model_ref.provider_type == "ollama":
            from tensortruth.core.ollama import check_tool_call_support

            return check_tool_call_support(model_ref.model_name)
        if model_ref.capabilities:
            return "tools" in model_ref.capabilities
        # No static capabilities — try dynamic detection for llama_cpp
        if model_ref.provider_type == "llama_cpp":
            return self._check_llama_cpp_capability(model_ref, "tools")
        return False

    def check_thinking_support(self, model_ref: ModelReference) -> bool:
        """Check if a model supports thinking tokens."""
        if model_ref.provider_type == "ollama":
            from tensortruth.core.ollama import check_thinking_support

            return check_thinking_support(model_ref.model_name)
        if model_ref.capabilities:
            return "thinking" in model_ref.capabilities
        # No static capabilities — try dynamic detection for llama_cpp
        if model_ref.provider_type == "llama_cpp":
            return self._check_llama_cpp_capability(model_ref, "thinking")
        return False

    def _check_llama_cpp_capability(self, model_ref: ModelReference, cap: str) -> bool:
        """Dynamic capability detection for llama.cpp via /props.

        Uses ``/props?model=<id>`` to query per-model properties in
        router mode, inspecting the chat template for tool patterns.
        Results are cached so subsequent ``resolve_model`` calls for
        the same model inherit the detected capabilities.
        """
        key = f"{model_ref.provider_id}:{model_ref.model_name}"
        if key in self._caps_cache:
            caps = self._caps_cache[key]
        else:
            from tensortruth.core.llama_cpp import check_capabilities

            caps = check_capabilities(model_ref.base_url, model_ref.model_name)
            self._caps_cache[key] = caps
        # Propagate to model ref so LLM creation sees them
        if caps:
            model_ref.capabilities = caps
        return cap in caps


# ---------------------------------------------------------------------------
# Ollama model info cache (shared helper)
# ---------------------------------------------------------------------------

# TTL-based cache for Ollama model info.  Keyed by ``(base_url, model_name)``
# with values of ``(timestamp, result_dict)``.  Entries expire after
# ``_MODEL_INFO_TTL`` seconds so that transient failures are retried.
_model_info_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
_MODEL_INFO_TTL: float = 300.0  # 5 minutes


def _get_ollama_model_info(base_url: str, model_name: str) -> Dict[str, Any]:
    """Query ``/api/show`` for a single Ollama model.

    Results are cached with a 5-minute TTL so transient failures are retried.
    Delegates to ``ollama.get_model_info()`` and returns the subset needed
    by the provider layer (``context_length``, ``capabilities``).
    """
    cache_key = (base_url, model_name)
    now = time.monotonic()

    cached = _model_info_cache.get(cache_key)
    if cached is not None:
        ts, cached_result = cached
        if now - ts < _MODEL_INFO_TTL:
            return cached_result

    import requests as _req

    result: Dict[str, Any] = {
        "context_length": 4096,
        "capabilities": [],
    }
    try:
        resp = _req.post(
            f"{base_url.rstrip('/')}/api/show",
            json={"model": model_name},
            timeout=2,
        )
        if resp.status_code == 200:
            data = resp.json()
            model_info = data.get("model_info", {})
            for key in model_info:
                if "context_length" in key.lower():
                    result["context_length"] = model_info[key]
                    break
            result["capabilities"] = data.get("capabilities", [])
    except Exception:
        pass
    _model_info_cache[cache_key] = (now, result)
    return result


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------


def create_llm(model_ref: ModelReference, **kwargs) -> LLM:
    """Create a LlamaIndex LLM instance for any provider type.

    Keyword arguments are merged on top of provider defaults.  Common kwargs:
    ``temperature``, ``context_window``, ``thinking``, ``request_timeout``,
    ``system_prompt``, ``additional_kwargs``.
    """
    if model_ref.provider_type == "ollama":
        return _create_ollama_llm(model_ref, **kwargs)
    elif model_ref.provider_type == "openai_compatible":
        return _create_openai_like_llm(model_ref, **kwargs)
    elif model_ref.provider_type == "llama_cpp":
        return _create_llama_cpp_llm(model_ref, **kwargs)
    else:
        raise ValueError(f"Unknown provider type: {model_ref.provider_type}")


def _create_ollama_llm(model_ref: ModelReference, **kwargs) -> LLM:
    from llama_index.llms.ollama import Ollama

    from tensortruth.core.ollama import _patch_parallel_tool_calls

    ctx = kwargs.pop("context_window", model_ref.context_window)
    additional = kwargs.pop("additional_kwargs", {})
    additional.setdefault("num_ctx", ctx)
    additional.setdefault("num_predict", -1)

    llm = Ollama(
        model=model_ref.model_name,
        base_url=model_ref.base_url,
        context_window=ctx,
        request_timeout=kwargs.pop("request_timeout", float(model_ref.timeout)),
        additional_kwargs=additional,
        **kwargs,
    )
    _patch_parallel_tool_calls(llm)
    return llm


def _create_openai_like_llm(model_ref: ModelReference, **kwargs) -> LLM:
    from llama_index.llms.openai_like import OpenAILike

    ctx = kwargs.pop("context_window", model_ref.context_window)
    # OpenAILike does not use additional_kwargs; translate to max_tokens
    additional = kwargs.pop("additional_kwargs", {})
    max_tokens = kwargs.pop("max_tokens", additional.get("num_predict", None))
    # Strip Ollama-specific kwargs
    kwargs.pop("thinking", None)

    llm_kwargs: Dict[str, Any] = {
        "model": model_ref.model_name,
        "api_base": model_ref.base_url,
        "api_key": model_ref.api_key or "no-key",
        "context_window": ctx,
        "timeout": float(model_ref.timeout),
        "is_chat_model": True,
    }
    if max_tokens and max_tokens > 0:
        llm_kwargs["max_tokens"] = max_tokens

    # Pass through remaining kwargs
    llm_kwargs.update(kwargs)

    return OpenAILike(**llm_kwargs)


def _create_llama_cpp_llm(model_ref: ModelReference, **kwargs) -> LLM:
    """Create LLM for a llama.cpp provider.

    Translates thinking preferences into llama.cpp-specific request params
    (``reasoning_format``, ``think_budget``), enables ``is_function_calling_model``
    when tool support is detected, and delegates to OpenAILike.
    """
    thinking = kwargs.pop("thinking", None)
    additional = kwargs.pop("additional_kwargs", {})

    if thinking is not None:
        if thinking is False:
            additional["reasoning_format"] = "none"
        elif isinstance(thinking, str) and thinking in ("low", "medium", "high"):
            additional["reasoning_format"] = "deepseek"
            budget_map = {"low": 1024, "medium": 4096, "high": -1}
            additional["think_budget"] = budget_map[thinking]
        elif thinking:
            additional["reasoning_format"] = "deepseek"

    # Enable function calling if model supports tools
    if "tools" in model_ref.capabilities:
        kwargs.setdefault("is_function_calling_model", True)

    kwargs["additional_kwargs"] = additional
    return _create_openai_like_llm(model_ref, **kwargs)


# ---------------------------------------------------------------------------
# Cached singleton LLM getters
# ---------------------------------------------------------------------------

_orchestrator_llm_instance: Optional[LLM] = None
_orchestrator_llm_key: Optional[Tuple] = None

_tool_llm_instance: Optional[LLM] = None
_tool_llm_key: Optional[Tuple] = None


def get_orchestrator_llm(model_ref: ModelReference, context_window: int) -> LLM:
    """Get or create a cached LLM for orchestrator use (thinking disabled)."""
    global _orchestrator_llm_instance, _orchestrator_llm_key

    key = (
        model_ref.provider_id,
        model_ref.model_name,
        model_ref.base_url,
        context_window,
    )

    if _orchestrator_llm_instance is not None and _orchestrator_llm_key == key:
        return _orchestrator_llm_instance

    logger.info(
        "Creating orchestrator LLM: provider=%s, model=%s, ctx=%d",
        model_ref.provider_id,
        model_ref.model_name,
        context_window,
    )

    _orchestrator_llm_instance = create_llm(
        model_ref,
        temperature=0.2,
        context_window=context_window,
        thinking=False,
        request_timeout=120.0,
    )
    _orchestrator_llm_key = key
    return _orchestrator_llm_instance


def get_tool_llm(
    model_ref: ModelReference,
    context_window: int,
    thinking: Optional[Union[bool, str]] = None,
) -> LLM:
    """Get or create a cached LLM for tool/synthesis use (thinking enabled when supported)."""
    global _tool_llm_instance, _tool_llm_key

    resolved_thinking = (
        thinking
        if thinking is not None
        else (ProviderRegistry.get_instance().check_thinking_support(model_ref))
    )

    key = (
        model_ref.provider_id,
        model_ref.model_name,
        model_ref.base_url,
        context_window,
        resolved_thinking,
    )

    if _tool_llm_instance is not None and _tool_llm_key == key:
        return _tool_llm_instance

    logger.info(
        "Creating tool LLM: provider=%s, model=%s, ctx=%d, thinking=%s",
        model_ref.provider_id,
        model_ref.model_name,
        context_window,
        resolved_thinking,
    )

    extra_kwargs: Dict[str, Any] = {
        "temperature": 0.7,
        "context_window": context_window,
        "request_timeout": 120.0,
    }
    # Pass thinking to Ollama (native) and llama_cpp (translated to reasoning_format)
    if model_ref.provider_type in ("ollama", "llama_cpp"):
        extra_kwargs["thinking"] = resolved_thinking

    _tool_llm_instance = create_llm(model_ref, **extra_kwargs)
    _tool_llm_key = key
    return _tool_llm_instance


# ---------------------------------------------------------------------------
# Thinking resolution (provider-aware)
# ---------------------------------------------------------------------------


def resolve_thinking(
    model_ref: ModelReference,
    user_preference: Optional[Union[bool, Literal["low", "medium", "high"]]] = None,
) -> Union[bool, Literal["low", "medium", "high"]]:
    """Resolve the effective thinking setting for a model (provider-aware).

    For Ollama models, delegates to the existing Ollama-specific logic.
    For openai_compatible models, uses capabilities from config.
    """
    if model_ref.provider_type == "ollama":
        from tensortruth.core.ollama import resolve_thinking as ollama_resolve

        return ollama_resolve(model_ref.model_name, user_preference)

    # openai_compatible / llama_cpp: check config capabilities
    supports = "thinking" in model_ref.capabilities

    if user_preference is None:
        return supports
    if user_preference is False:
        return False
    if not supports:
        return False
    return user_preference


# ---------------------------------------------------------------------------
# Convenience helper for resolving models from session/engine params
# ---------------------------------------------------------------------------


def resolve_model_from_params(
    params: Dict[str, Any], model_name: str
) -> "ModelReference":
    """Resolve a ModelReference from session/engine params.

    Extracts ``provider_id`` from *params* and delegates to
    ``ProviderRegistry.resolve_model``.  This avoids repeating the
    three-line lookup pattern across the codebase.
    """
    provider_id = params.get("provider_id")
    registry = ProviderRegistry.get_instance()
    return registry.resolve_model(model_name, provider_id)
