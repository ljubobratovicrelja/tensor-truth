"""
Unit tests for tensortruth.core.providers module.
"""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.app_utils.config_schema import ProviderConfig
from tensortruth.core.providers import (
    ModelReference,
    ProviderRegistry,
    create_llm,
    get_orchestrator_llm,
    resolve_thinking,
)


# ============================================================================
# ModelReference tests
# ============================================================================


@pytest.mark.unit
class TestModelReference:
    """Tests for the ModelReference dataclass."""

    def test_basic_creation(self):
        ref = ModelReference(
            provider_id="ollama",
            model_name="qwen3:32b",
            display_name="qwen3:32b",
            provider_type="ollama",
            base_url="http://localhost:11434",
        )
        assert ref.provider_id == "ollama"
        assert ref.model_name == "qwen3:32b"
        assert ref.provider_type == "ollama"
        assert ref.api_key == ""
        assert ref.capabilities == []
        assert ref.context_window == 4096
        assert ref.timeout == 300

    def test_openai_compatible(self):
        ref = ModelReference(
            provider_id="vllm",
            model_name="meta-llama/Llama-3.1-70B",
            display_name="Llama 3.1 70B",
            provider_type="openai_compatible",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            capabilities=["tools"],
            context_window=131072,
        )
        assert ref.provider_type == "openai_compatible"
        assert ref.api_key == "test-key"
        assert "tools" in ref.capabilities
        assert ref.context_window == 131072


# ============================================================================
# ProviderRegistry tests
# ============================================================================


@pytest.mark.unit
class TestProviderRegistry:
    """Tests for the ProviderRegistry singleton."""

    def setup_method(self):
        ProviderRegistry.reset()

    def teardown_method(self):
        ProviderRegistry.reset()

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    def test_singleton(self, mock_load):
        """Singleton returns same instance."""
        r1 = ProviderRegistry.get_instance()
        r2 = ProviderRegistry.get_instance()
        assert r1 is r2
        # _load_from_config called only once
        mock_load.assert_called_once()

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    def test_reset(self, mock_load):
        """Reset clears singleton."""
        r1 = ProviderRegistry.get_instance()
        ProviderRegistry.reset()
        r2 = ProviderRegistry.get_instance()
        assert r1 is not r2

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    def test_get_provider(self, mock_load):
        registry = ProviderRegistry.get_instance()
        registry._providers = [
            ProviderConfig(id="ollama", type="ollama"),
            ProviderConfig(id="vllm", type="openai_compatible", base_url="http://localhost:8000/v1"),
        ]
        assert registry.get_provider("ollama").id == "ollama"
        assert registry.get_provider("vllm").type == "openai_compatible"
        assert registry.get_provider("nonexistent") is None

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    def test_get_default_ollama_provider(self, mock_load):
        registry = ProviderRegistry.get_instance()
        registry._providers = [
            ProviderConfig(id="vllm", type="openai_compatible"),
            ProviderConfig(id="my-ollama", type="ollama", base_url="http://gpu:11434"),
        ]
        ollama = registry.get_default_ollama_provider()
        assert ollama is not None
        assert ollama.id == "my-ollama"

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    def test_get_default_ollama_provider_none(self, mock_load):
        registry = ProviderRegistry.get_instance()
        registry._providers = [
            ProviderConfig(id="vllm", type="openai_compatible"),
        ]
        assert registry.get_default_ollama_provider() is None

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    def test_get_static_models(self, mock_load):
        registry = ProviderRegistry.get_instance()
        provider = ProviderConfig(
            id="vllm",
            type="openai_compatible",
            base_url="http://localhost:8000/v1",
            models=[
                {"name": "llama-70b", "display_name": "Llama 70B", "capabilities": ["tools"]},
                {"name": "mistral-7b"},
            ],
        )
        models = registry._get_static_models(provider)
        assert len(models) == 2
        assert models[0].model_name == "llama-70b"
        assert models[0].display_name == "Llama 70B"
        assert models[0].capabilities == ["tools"]
        assert models[1].model_name == "mistral-7b"
        assert models[1].display_name == "mistral-7b"

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    def test_resolve_model_with_provider_id(self, mock_load):
        registry = ProviderRegistry.get_instance()
        registry._providers = [
            ProviderConfig(
                id="vllm",
                type="openai_compatible",
                base_url="http://localhost:8000/v1",
                models=[{"name": "llama-70b", "capabilities": ["tools"]}],
            ),
        ]
        ref = registry.resolve_model("llama-70b", provider_id="vllm")
        assert ref.provider_id == "vllm"
        assert ref.model_name == "llama-70b"
        assert ref.provider_type == "openai_compatible"

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    def test_resolve_model_fallback(self, mock_load):
        """Resolving with no providers returns a fallback."""
        registry = ProviderRegistry.get_instance()
        registry._providers = []
        ref = registry.resolve_model("some-model")
        assert ref.model_name == "some-model"
        assert ref.provider_type == "ollama"

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    def test_check_tool_support_openai_compatible(self, mock_load):
        registry = ProviderRegistry.get_instance()
        ref_with_tools = ModelReference(
            provider_id="vllm",
            model_name="llama-70b",
            display_name="llama-70b",
            provider_type="openai_compatible",
            base_url="http://localhost:8000/v1",
            capabilities=["tools"],
        )
        ref_without_tools = ModelReference(
            provider_id="vllm",
            model_name="mistral-7b",
            display_name="mistral-7b",
            provider_type="openai_compatible",
            base_url="http://localhost:8000/v1",
        )
        assert registry.check_tool_support(ref_with_tools) is True
        assert registry.check_tool_support(ref_without_tools) is False

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    def test_check_thinking_support_openai_compatible(self, mock_load):
        registry = ProviderRegistry.get_instance()
        ref = ModelReference(
            provider_id="openrouter",
            model_name="claude-sonnet",
            display_name="Claude Sonnet",
            provider_type="openai_compatible",
            base_url="https://openrouter.ai/api/v1",
            capabilities=["tools", "thinking"],
        )
        assert registry.check_thinking_support(ref) is True


# ============================================================================
# create_llm factory tests
# ============================================================================


@pytest.mark.unit
class TestCreateLlm:
    """Tests for the create_llm factory function."""

    @patch("tensortruth.core.providers._create_ollama_llm")
    def test_dispatches_to_ollama(self, mock_create):
        mock_create.return_value = MagicMock()
        ref = ModelReference(
            provider_id="ollama",
            model_name="qwen3:32b",
            display_name="qwen3:32b",
            provider_type="ollama",
            base_url="http://localhost:11434",
        )
        result = create_llm(ref, temperature=0.5)
        mock_create.assert_called_once_with(ref, temperature=0.5)
        assert result == mock_create.return_value

    @patch("tensortruth.core.providers._create_openai_like_llm")
    def test_dispatches_to_openai_compatible(self, mock_create):
        mock_create.return_value = MagicMock()
        ref = ModelReference(
            provider_id="vllm",
            model_name="llama-70b",
            display_name="llama-70b",
            provider_type="openai_compatible",
            base_url="http://localhost:8000/v1",
        )
        result = create_llm(ref, temperature=0.3)
        mock_create.assert_called_once_with(ref, temperature=0.3)
        assert result == mock_create.return_value

    @patch("tensortruth.core.providers._create_llama_cpp_llm")
    def test_dispatches_to_llama_cpp(self, mock_create):
        mock_create.return_value = MagicMock()
        ref = ModelReference(
            provider_id="llama-cpp",
            model_name="model.gguf",
            display_name="model",
            provider_type="llama_cpp",
            base_url="http://localhost:8080",
        )
        result = create_llm(ref, temperature=0.5)
        mock_create.assert_called_once_with(ref, temperature=0.5)
        assert result == mock_create.return_value

    def test_unknown_provider_raises(self):
        ref = ModelReference(
            provider_id="unknown",
            model_name="model",
            display_name="model",
            provider_type="unknown_type",
            base_url="http://localhost",
        )
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_llm(ref)

    @patch("llama_index.llms.ollama.Ollama")
    @patch("tensortruth.core.ollama._patch_parallel_tool_calls")
    def test_create_ollama_llm_sets_num_ctx(self, mock_patch, mock_ollama_cls):
        """Ollama LLM gets num_ctx in additional_kwargs."""
        from tensortruth.core.providers import _create_ollama_llm

        mock_llm = MagicMock()
        mock_ollama_cls.return_value = mock_llm

        ref = ModelReference(
            provider_id="ollama",
            model_name="test",
            display_name="test",
            provider_type="ollama",
            base_url="http://localhost:11434",
        )
        _create_ollama_llm(ref, context_window=32768)

        call_kwargs = mock_ollama_cls.call_args
        additional = call_kwargs.kwargs.get("additional_kwargs", {})
        assert additional.get("num_ctx") == 32768
        mock_patch.assert_called_once_with(mock_llm)

    def test_create_openai_like_llm(self):
        from unittest.mock import patch as _patch

        mock_openai_cls = MagicMock()
        mock_llm = MagicMock()
        mock_openai_cls.return_value = mock_llm

        # The import happens inside the function, so we mock the module
        mock_module = MagicMock()
        mock_module.OpenAILike = mock_openai_cls

        with _patch.dict("sys.modules", {"llama_index.llms.openai_like": mock_module}):
            from tensortruth.core.providers import _create_openai_like_llm

            ref = ModelReference(
                provider_id="vllm",
                model_name="llama-70b",
                display_name="llama-70b",
                provider_type="openai_compatible",
                base_url="http://localhost:8000/v1",
                api_key="test-key",
            )
            result = _create_openai_like_llm(ref, temperature=0.5)
            assert result == mock_llm

            call_kwargs = mock_openai_cls.call_args.kwargs
            assert call_kwargs["model"] == "llama-70b"
            assert call_kwargs["api_base"] == "http://localhost:8000/v1"
            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["is_chat_model"] is True


# ============================================================================
# Singleton LLM getter tests
# ============================================================================


@pytest.mark.unit
class TestSingletonLLMs:
    """Tests for get_orchestrator_llm and get_tool_llm caching."""

    def setup_method(self):
        import tensortruth.core.providers as mod

        mod._orchestrator_llm_instance = None
        mod._orchestrator_llm_key = None
        mod._tool_llm_instance = None
        mod._tool_llm_key = None
        ProviderRegistry.reset()

    def teardown_method(self):
        import tensortruth.core.providers as mod

        mod._orchestrator_llm_instance = None
        mod._orchestrator_llm_key = None
        mod._tool_llm_instance = None
        mod._tool_llm_key = None
        ProviderRegistry.reset()

    @patch("tensortruth.core.providers.create_llm")
    def test_orchestrator_llm_caches(self, mock_create):
        mock_llm = MagicMock()
        mock_create.return_value = mock_llm

        ref = ModelReference(
            provider_id="ollama",
            model_name="qwen3:32b",
            display_name="qwen3:32b",
            provider_type="ollama",
            base_url="http://localhost:11434",
        )

        llm1 = get_orchestrator_llm(ref, 16384)
        llm2 = get_orchestrator_llm(ref, 16384)

        assert llm1 is llm2
        assert mock_create.call_count == 1

    @patch("tensortruth.core.providers.create_llm")
    def test_orchestrator_llm_recreates_on_different_params(self, mock_create):
        mock_create.return_value = MagicMock()

        ref = ModelReference(
            provider_id="ollama",
            model_name="qwen3:32b",
            display_name="qwen3:32b",
            provider_type="ollama",
            base_url="http://localhost:11434",
        )

        get_orchestrator_llm(ref, 16384)
        get_orchestrator_llm(ref, 32768)  # Different context window

        assert mock_create.call_count == 2


# ============================================================================
# resolve_thinking tests
# ============================================================================


@pytest.mark.unit
class TestResolveThinking:
    """Tests for provider-aware thinking resolution."""

    def test_openai_compatible_with_thinking_capability(self):
        ref = ModelReference(
            provider_id="openrouter",
            model_name="claude",
            display_name="Claude",
            provider_type="openai_compatible",
            base_url="https://openrouter.ai/api/v1",
            capabilities=["thinking"],
        )
        # Auto (None) -> True (model supports it)
        assert resolve_thinking(ref, None) is True
        # Explicit True -> True
        assert resolve_thinking(ref, True) is True
        # Explicit False -> False
        assert resolve_thinking(ref, False) is False

    def test_openai_compatible_without_thinking_capability(self):
        ref = ModelReference(
            provider_id="vllm",
            model_name="llama",
            display_name="llama",
            provider_type="openai_compatible",
            base_url="http://localhost:8000/v1",
        )
        # Auto (None) -> False (model doesn't support)
        assert resolve_thinking(ref, None) is False
        # Explicit True but unsupported -> False
        assert resolve_thinking(ref, True) is False

    def test_llama_cpp_with_thinking_capability(self):
        ref = ModelReference(
            provider_id="llama-cpp",
            model_name="qwen3:Q4_K_M",
            display_name="Qwen3 Q4_K_M",
            provider_type="llama_cpp",
            base_url="http://localhost:8080",
            capabilities=["thinking"],
        )
        assert resolve_thinking(ref, None) is True
        assert resolve_thinking(ref, True) is True
        assert resolve_thinking(ref, False) is False
        assert resolve_thinking(ref, "high") == "high"

    def test_llama_cpp_without_thinking_capability(self):
        ref = ModelReference(
            provider_id="llama-cpp",
            model_name="mistral",
            display_name="Mistral",
            provider_type="llama_cpp",
            base_url="http://localhost:8080",
        )
        assert resolve_thinking(ref, None) is False
        assert resolve_thinking(ref, True) is False

    @patch("tensortruth.core.ollama.resolve_thinking")
    def test_ollama_delegates(self, mock_resolve):
        mock_resolve.return_value = True
        ref = ModelReference(
            provider_id="ollama",
            model_name="qwen3:32b",
            display_name="qwen3:32b",
            provider_type="ollama",
            base_url="http://localhost:11434",
        )
        result = resolve_thinking(ref, None)
        mock_resolve.assert_called_once_with("qwen3:32b", None)
        assert result is True


# ============================================================================
# _create_llama_cpp_llm thinking translation tests
# ============================================================================


@pytest.mark.unit
class TestCreateLlamaCppLlm:
    """Tests for _create_llama_cpp_llm thinking parameter translation."""

    @patch("tensortruth.core.providers._create_openai_like_llm")
    def test_thinking_false_sets_reasoning_format_none(self, mock_openai):
        from tensortruth.core.providers import _create_llama_cpp_llm

        mock_openai.return_value = MagicMock()
        ref = ModelReference(
            provider_id="llama-cpp",
            model_name="model",
            display_name="model",
            provider_type="llama_cpp",
            base_url="http://localhost:8080",
        )
        _create_llama_cpp_llm(ref, thinking=False)
        call_kwargs = mock_openai.call_args.kwargs
        additional = call_kwargs.get("additional_kwargs", {})
        assert additional["reasoning_format"] == "none"

    @patch("tensortruth.core.providers._create_openai_like_llm")
    def test_thinking_true_sets_deepseek(self, mock_openai):
        from tensortruth.core.providers import _create_llama_cpp_llm

        mock_openai.return_value = MagicMock()
        ref = ModelReference(
            provider_id="llama-cpp",
            model_name="model",
            display_name="model",
            provider_type="llama_cpp",
            base_url="http://localhost:8080",
        )
        _create_llama_cpp_llm(ref, thinking=True)
        call_kwargs = mock_openai.call_args.kwargs
        additional = call_kwargs.get("additional_kwargs", {})
        assert additional["reasoning_format"] == "deepseek"
        assert "think_budget" not in additional

    @patch("tensortruth.core.providers._create_openai_like_llm")
    def test_thinking_level_sets_budget(self, mock_openai):
        from tensortruth.core.providers import _create_llama_cpp_llm

        mock_openai.return_value = MagicMock()
        ref = ModelReference(
            provider_id="llama-cpp",
            model_name="model",
            display_name="model",
            provider_type="llama_cpp",
            base_url="http://localhost:8080",
        )
        _create_llama_cpp_llm(ref, thinking="medium")
        call_kwargs = mock_openai.call_args.kwargs
        additional = call_kwargs.get("additional_kwargs", {})
        assert additional["reasoning_format"] == "deepseek"
        assert additional["think_budget"] == 4096

    @patch("tensortruth.core.providers._create_openai_like_llm")
    def test_thinking_high_budget_unlimited(self, mock_openai):
        from tensortruth.core.providers import _create_llama_cpp_llm

        mock_openai.return_value = MagicMock()
        ref = ModelReference(
            provider_id="llama-cpp",
            model_name="model",
            display_name="model",
            provider_type="llama_cpp",
            base_url="http://localhost:8080",
        )
        _create_llama_cpp_llm(ref, thinking="high")
        call_kwargs = mock_openai.call_args.kwargs
        additional = call_kwargs.get("additional_kwargs", {})
        assert additional["think_budget"] == -1

    @patch("tensortruth.core.providers._create_openai_like_llm")
    def test_no_thinking_kwarg(self, mock_openai):
        from tensortruth.core.providers import _create_llama_cpp_llm

        mock_openai.return_value = MagicMock()
        ref = ModelReference(
            provider_id="llama-cpp",
            model_name="model",
            display_name="model",
            provider_type="llama_cpp",
            base_url="http://localhost:8080",
        )
        _create_llama_cpp_llm(ref, temperature=0.5)
        call_kwargs = mock_openai.call_args.kwargs
        additional = call_kwargs.get("additional_kwargs", {})
        assert "reasoning_format" not in additional


# ============================================================================
# _get_llama_cpp_models tests
# ============================================================================


@pytest.mark.unit
class TestGetLlamaCppModels:
    """Tests for ProviderRegistry._get_llama_cpp_models."""

    def setup_method(self):
        ProviderRegistry.reset()

    def teardown_method(self):
        ProviderRegistry.reset()

    @patch("tensortruth.core.providers.ProviderRegistry._load_from_config")
    @patch("tensortruth.core.llama_cpp.get_available_models")
    def test_get_llama_cpp_models(self, mock_get_models, mock_load):
        mock_get_models.return_value = [
            {
                "id": "ggml-org/Qwen3-4B-GGUF:Q4_K_M",
                "status": "loaded", "in_cache": True, "path": "",
            },
            {
                "id": "Mistral-7B.gguf",
                "status": "unloaded", "in_cache": False, "path": "",
            },
        ]

        registry = ProviderRegistry.get_instance()
        provider = ProviderConfig(
            id="llama-cpp",
            type="llama_cpp",
            base_url="http://localhost:8080",
            models=[
                {"name": "ggml-org/Qwen3-4B-GGUF:Q4_K_M", "capabilities": ["tools", "thinking"]},
            ],
        )
        models = registry._get_llama_cpp_models(provider)
        assert len(models) == 2
        assert models[0].model_name == "ggml-org/Qwen3-4B-GGUF:Q4_K_M"
        assert models[0].provider_type == "llama_cpp"
        assert models[0].capabilities == ["tools", "thinking"]
        # Second model has no static config — empty capabilities
        assert models[1].capabilities == []
        assert models[1].display_name == "Mistral-7B"
