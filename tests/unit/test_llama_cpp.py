"""Unit tests for tensortruth.core.llama_cpp module."""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.core.llama_cpp import (
    check_health,
    format_display_name,
    get_available_models,
    get_loaded_models,
    load_model,
    unload_model,
)


@pytest.mark.unit
class TestGetAvailableModels:
    """Tests for get_available_models."""

    @patch("tensortruth.core.llama_cpp.requests.get")
    def test_returns_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": "ggml-org/Qwen3-4B-GGUF:Q4_K_M",
                    "status": {"value": "loaded"},
                    "in_cache": True,
                    "path": "/models/qwen3.gguf",
                },
                {
                    "id": "Mistral-7B.gguf",
                    "status": {"value": "unloaded"},
                    "in_cache": False,
                    "path": "/models/mistral.gguf",
                },
            ]
        }
        mock_get.return_value = mock_resp

        models = get_available_models("http://localhost:8080")
        assert len(models) == 2
        assert models[0]["id"] == "ggml-org/Qwen3-4B-GGUF:Q4_K_M"
        assert models[0]["status"] == "loaded"
        assert models[0]["in_cache"] is True
        assert models[1]["status"] == "unloaded"

    @patch("tensortruth.core.llama_cpp.requests.get")
    def test_server_down(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        models = get_available_models("http://localhost:8080")
        assert models == []

    @patch("tensortruth.core.llama_cpp.requests.get")
    def test_non_200(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp
        models = get_available_models("http://localhost:8080")
        assert models == []


@pytest.mark.unit
class TestGetLoadedModels:
    """Tests for get_loaded_models."""

    @patch("tensortruth.core.llama_cpp.get_available_models")
    def test_filters_loaded(self, mock_get):
        mock_get.return_value = [
            {"id": "a", "status": "loaded"},
            {"id": "b", "status": "unloaded"},
            {"id": "c", "status": "loading"},
        ]
        loaded = get_loaded_models("http://localhost:8080")
        assert len(loaded) == 2
        assert loaded[0]["id"] == "a"
        assert loaded[1]["id"] == "c"


@pytest.mark.unit
class TestLoadModel:
    """Tests for load_model."""

    @patch("tensortruth.core.llama_cpp.requests.post")
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        assert load_model("http://localhost:8080", "test-model") is True
        mock_post.assert_called_once()

    @patch("tensortruth.core.llama_cpp.requests.post")
    def test_failure(self, mock_post):
        mock_post.side_effect = ConnectionError("refused")
        assert load_model("http://localhost:8080", "test-model") is False


@pytest.mark.unit
class TestUnloadModel:
    """Tests for unload_model."""

    @patch("tensortruth.core.llama_cpp.requests.post")
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        assert unload_model("http://localhost:8080", "test-model") is True

    @patch("tensortruth.core.llama_cpp.requests.post")
    def test_failure(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp
        assert unload_model("http://localhost:8080", "test-model") is False


@pytest.mark.unit
class TestCheckHealth:
    """Tests for check_health."""

    @patch("tensortruth.core.llama_cpp.requests.get")
    def test_healthy(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        assert check_health("http://localhost:8080") is True

    @patch("tensortruth.core.llama_cpp.requests.get")
    def test_down(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        assert check_health("http://localhost:8080") is False


@pytest.mark.unit
class TestFormatDisplayName:
    """Tests for format_display_name."""

    def test_repo_with_quant(self):
        assert format_display_name("ggml-org/gemma-3-4b-it-GGUF:Q4_K_M") == "gemma-3-4b-it Q4_K_M"

    def test_gguf_extension(self):
        assert format_display_name("Qwen3-8B-Q4_K_M.gguf") == "Qwen3-8B-Q4_K_M"

    def test_with_path_prefix(self):
        assert format_display_name("models/Qwen3-8B-Q4_K_M.gguf") == "Qwen3-8B-Q4_K_M"

    def test_plain_name(self):
        assert format_display_name("my-model") == "my-model"

    def test_repo_without_gguf_suffix(self):
        assert format_display_name("user/model-name:Q8_0") == "model-name Q8_0"
