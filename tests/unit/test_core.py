"""
Unit tests for tensortruth.core modules (ollama, system).
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from tensortruth.core import get_max_memory_gb, get_running_models, stop_model

# ============================================================================
# Tests for Ollama Module
# ============================================================================


@pytest.mark.unit
class TestOllamaModule:
    """Tests for core.ollama module."""

    @patch("tensortruth.core.ollama.requests.get")
    def test_get_running_models_success(self, mock_get):
        """Test successful retrieval of running models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "deepseek-r1:8b",
                    "size_vram": 5500000000,  # ~5.1 GB
                    "expires_at": "2023-12-13T11:00:00Z",
                }
            ]
        }
        mock_get.return_value = mock_response

        models = get_running_models()

        assert len(models) == 1
        assert models[0]["name"] == "deepseek-r1:8b"
        assert "5.1 GB" in models[0]["size_vram"]

    @patch("tensortruth.core.ollama.requests.get")
    def test_get_running_models_connection_error(self, mock_get):
        """Test handling of connection errors."""
        mock_get.side_effect = Exception("Connection refused")

        models = get_running_models()

        assert models == []

    @patch("tensortruth.core.ollama.requests.get")
    def test_get_running_models_empty(self, mock_get):
        """Test when no models are running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        models = get_running_models()

        assert models == []

    @patch("tensortruth.core.ollama.requests.post")
    def test_stop_model_success(self, mock_post):
        """Test successful model stop."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = stop_model("deepseek-r1:8b")

        assert result is True
        mock_post.assert_called_once()

        # Verify payload
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "deepseek-r1:8b"
        assert call_args[1]["json"]["keep_alive"] == 0

    @patch("tensortruth.core.ollama.requests.post")
    def test_stop_model_failure(self, mock_post):
        """Test model stop failure."""
        mock_post.side_effect = Exception("API error")

        result = stop_model("deepseek-r1:8b")

        assert result is False


# ============================================================================
# Tests for System Module
# ============================================================================


@pytest.mark.unit
class TestSystemModule:
    """Tests for core.system module."""

    def test_get_max_memory_gb_with_cuda(self, monkeypatch):
        """Test memory detection with CUDA."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def mock_mem_get_info():
            # 8 GiB free, 24 GiB total
            return (8 * 1024**3, 24 * 1024**3)

        monkeypatch.setattr(torch.cuda, "mem_get_info", mock_mem_get_info)

        memory = get_max_memory_gb()

        assert memory == pytest.approx(24.0, rel=0.01)

    def test_get_max_memory_gb_with_mps(self, monkeypatch):
        """Test memory detection with MPS (Apple Silicon)."""
        # Mock CUDA as unavailable
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Mock MPS availability
        if not hasattr(torch.backends, "mps"):
            mock_mps = MagicMock()
            mock_mps.is_available.return_value = True
            monkeypatch.setattr(torch.backends, "mps", mock_mps)
        else:
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        memory = get_max_memory_gb()

        # Should return system RAM (reasonable value check)
        assert memory > 0
        assert memory < 1024  # Less than 1TB

    def test_get_max_memory_gb_cpu_only(self, monkeypatch):
        """Test memory detection with CPU only."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Mock MPS as unavailable
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        memory = get_max_memory_gb()

        # Should return system RAM
        assert memory > 0
        assert memory < 1024

    def test_get_max_memory_gb_fallback(self, monkeypatch):
        """Test fallback when all detection methods fail."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        # Make psutil unavailable
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("psutil not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        memory = get_max_memory_gb()

        # Should return fallback value
        assert memory == 16.0
