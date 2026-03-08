"""Tests for model load/unload status features."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestOllamaLoadModel:
    """Tests for ollama.load_model()."""

    @patch("tensortruth.core.ollama.requests.post")
    @patch(
        "tensortruth.core.ollama.get_api_base",
        return_value="http://localhost:11434/api",
    )
    def test_load_model_success(self, mock_base, mock_post):
        from tensortruth.core.ollama import load_model

        mock_post.return_value = MagicMock(status_code=200)
        result = load_model("llama3:8b", num_ctx=8192)
        assert result is True
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b",
                "keep_alive": "5m",
                "prompt": "",
                "options": {"num_ctx": 8192},
            },
            timeout=120,
        )

    @patch("tensortruth.core.ollama.requests.post")
    @patch(
        "tensortruth.core.ollama.get_api_base",
        return_value="http://localhost:11434/api",
    )
    def test_load_model_failure(self, mock_base, mock_post):
        from tensortruth.core.ollama import load_model

        mock_post.return_value = MagicMock(status_code=500)
        result = load_model("nonexistent:latest", num_ctx=8192)
        assert result is False

    @patch("tensortruth.core.ollama.requests.post")
    @patch(
        "tensortruth.core.ollama.get_api_base",
        return_value="http://localhost:11434/api",
    )
    def test_load_model_connection_error(self, mock_base, mock_post):
        from tensortruth.core.ollama import load_model

        mock_post.side_effect = Exception("Connection refused")
        result = load_model("llama3:8b", num_ctx=8192)
        assert result is False

    @patch("tensortruth.core.ollama.requests.post")
    @patch(
        "tensortruth.core.ollama.get_api_base",
        return_value="http://localhost:11434/api",
    )
    def test_load_model_custom_keep_alive(self, mock_base, mock_post):
        from tensortruth.core.ollama import load_model

        mock_post.return_value = MagicMock(status_code=200)
        load_model("llama3:8b", keep_alive="10m", num_ctx=16384)
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b",
                "keep_alive": "10m",
                "prompt": "",
                "options": {"num_ctx": 16384},
            },
            timeout=120,
        )

    @patch("tensortruth.core.ollama.requests.post")
    @patch(
        "tensortruth.core.ollama.get_api_base",
        return_value="http://localhost:11434/api",
    )
    def test_load_model_reads_config_default(self, mock_base, mock_post):
        """When num_ctx is None, load_model reads from config."""
        from tensortruth.core.ollama import load_model

        mock_post.return_value = MagicMock(status_code=200)

        # Mock load_config to return a config with default_context_window=4096
        mock_config = MagicMock()
        mock_config.llm.default_context_window = 4096
        with patch(
            "tensortruth.app_utils.config.load_config",
            return_value=mock_config,
        ):
            load_model("llama3:8b")  # num_ctx=None triggers config read

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b",
                "keep_alive": "5m",
                "prompt": "",
                "options": {"num_ctx": 4096},
            },
            timeout=120,
        )

    @patch("tensortruth.core.ollama.requests.post")
    @patch(
        "tensortruth.core.ollama.get_api_base",
        return_value="http://localhost:11434/api",
    )
    def test_load_model_fallback_when_config_unavailable(self, mock_base, mock_post):
        """When config is unavailable, falls back to 8192."""
        from tensortruth.core.ollama import load_model

        mock_post.return_value = MagicMock(status_code=200)

        with patch(
            "tensortruth.app_utils.config.load_config",
            side_effect=Exception("no config"),
        ):
            load_model("llama3:8b")

        call_args = mock_post.call_args
        assert call_args[1]["json"]["options"]["num_ctx"] == 8192


@pytest.mark.unit
class TestOllamaLoadUnloadEndpoints:
    """Tests for /system/ollama/load and /system/ollama/unload endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the system router."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tensortruth.api.routes.system import router

        app = FastAPI()
        app.include_router(router, prefix="/system")
        return TestClient(app)

    @patch("tensortruth.core.ollama.load_model", return_value=True)
    def test_load_endpoint_success(self, mock_load, client):
        resp = client.post("/system/ollama/load", json={"model": "llama3:8b"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "llama3:8b" in data["message"]
        mock_load.assert_called_once_with("llama3:8b")

    @patch("tensortruth.core.ollama.load_model", return_value=False)
    def test_load_endpoint_failure(self, mock_load, client):
        resp = client.post("/system/ollama/load", json={"model": "bad-model"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    @patch("tensortruth.core.ollama.stop_model", return_value=True)
    def test_unload_endpoint_success(self, mock_stop, client):
        resp = client.post("/system/ollama/unload", json={"model": "llama3:8b"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        mock_stop.assert_called_once_with("llama3:8b")

    @patch("tensortruth.core.ollama.stop_model", return_value=False)
    def test_unload_endpoint_failure(self, mock_stop, client):
        resp = client.post("/system/ollama/unload", json={"model": "bad-model"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False


@pytest.mark.unit
class TestModelInfoStatus:
    """Tests for status field in ModelInfo."""

    def test_model_info_status_field(self):
        from tensortruth.api.routes.modules import ModelInfo

        # Default is None
        m = ModelInfo(name="test")
        assert m.status is None

        # Can set to loaded/unloaded
        m_loaded = ModelInfo(name="test", status="loaded")
        assert m_loaded.status == "loaded"

        m_unloaded = ModelInfo(name="test", status="unloaded")
        assert m_unloaded.status == "unloaded"
