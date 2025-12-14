"""
Unit tests for tensortruth.app_utils modules.
"""

from unittest.mock import MagicMock, patch

import pytest

# Import modules directly to avoid streamlit dependency
from tensortruth.app_utils import presets, title_generation
from tensortruth.app_utils.helpers import get_ollama_models, get_system_devices
from tensortruth.app_utils.vram import estimate_vram_usage

# ============================================================================
# Tests for Helpers Module
# ============================================================================


@pytest.mark.unit
class TestHelpers:
    """Tests for app_utils.helpers module."""

    def test_get_system_devices_with_cuda(self, monkeypatch):
        """Test device detection with CUDA available."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        devices = get_system_devices()

        assert "cuda" in devices
        assert "cpu" in devices

    def test_get_system_devices_cpu_only(self, monkeypatch):
        """Test device detection with CPU only."""
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        devices = get_system_devices()

        assert "cpu" in devices
        assert "cuda" not in devices

    @patch("tensortruth.app_utils.helpers.requests.get")
    def test_get_ollama_models_success(self, mock_get):
        """Test successful Ollama model fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "deepseek-r1:8b"},
                {"name": "llama2:7b"},
            ]
        }
        mock_get.return_value = mock_response

        models = get_ollama_models()

        assert len(models) == 2
        assert "deepseek-r1:8b" in models
        assert "llama2:7b" in models

    @patch("tensortruth.app_utils.helpers.requests.get")
    def test_get_ollama_models_failure(self, mock_get):
        """Test Ollama model fetch when service is down."""
        mock_get.side_effect = Exception("Connection refused")

        models = get_ollama_models()

        # Should return default fallback
        assert "deepseek-r1:8b" in models


# ============================================================================
# Tests for VRAM Module
# ============================================================================


@pytest.mark.unit
class TestVRAM:
    """Tests for app_utils.vram module."""

    def test_estimate_vram_usage_cuda_rag_gpu_llm(self):
        """Test VRAM estimation with CUDA RAG and GPU LLM."""
        predicted, stats, cost = estimate_vram_usage(
            model_name="deepseek-r1:8b",
            num_indices=2,
            context_window=4096,
            rag_device="cuda",
            llm_device="gpu",
        )

        # RAG overhead (1.8) + index overhead (2*0.15) + LLM (5.5) + KV cache (~0.8)
        assert cost > 7.0  # Should be around 8.28
        assert cost < 10.0

    def test_estimate_vram_usage_cpu_rag_cpu_llm(self):
        """Test VRAM estimation with CPU RAG and CPU LLM."""
        predicted, stats, cost = estimate_vram_usage(
            model_name="deepseek-r1:8b",
            num_indices=2,
            context_window=4096,
            rag_device="cpu",
            llm_device="cpu",
        )

        # Only index overhead matters, LLM and RAG are in RAM
        assert cost < 1.0  # Should be just index overhead (0.3)

    def test_estimate_vram_usage_large_model(self):
        """Test VRAM estimation with large model."""
        predicted, stats, cost = estimate_vram_usage(
            model_name="deepseek-r1:70b",
            num_indices=1,
            context_window=8192,
            rag_device="cuda",
            llm_device="gpu",
        )

        # Should be significantly higher for 70b model
        assert cost > 40.0  # 70b models use ~40GB


# ============================================================================
# Tests for Title Generation Module
# ============================================================================


@pytest.mark.unit
class TestTitleGeneration:
    """Tests for app_utils.title_generation module."""

    @patch("tensortruth.app_utils.title_generation.requests.get")
    @patch("tensortruth.app_utils.title_generation.requests.post")
    def test_generate_smart_title_success(self, mock_post, mock_get):
        """Test successful title generation."""
        # Mock model availability check
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": [{"name": "qwen2.5:0.5b"}]}
        mock_get.return_value = mock_get_response

        # Mock title generation response
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {"response": "PyTorch Basics"}
        mock_post.return_value = mock_post_response

        title = title_generation.generate_smart_title(
            "What are the basics of PyTorch?", "deepseek-r1:8b"
        )

        assert title == "PyTorch Basics"

    @patch("tensortruth.app_utils.title_generation.requests.post")
    @patch("tensortruth.app_utils.title_generation.requests.get")
    def test_generate_smart_title_model_unavailable(self, mock_get, mock_post):
        """Test title generation when model is unavailable."""
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # Make model pull fail to simulate unavailability
        mock_post_response = MagicMock()
        mock_post_response.status_code = 500
        mock_post.return_value = mock_post_response

        text = (
            "This is a very long query that should be truncated for the fallback title"
        )
        title = title_generation.generate_smart_title(text)

        assert len(title) <= 32
        assert ".." in title


# ============================================================================
# Tests for Presets Module
# ============================================================================


@pytest.mark.unit
class TestPresets:
    """Tests for app_utils.presets module."""

    def test_save_and_load_preset(self, tmp_path):
        """Test saving and loading presets."""
        presets_file = tmp_path / "test_presets.json"

        config = {
            "modules": ["pytorch"],
            "model": "deepseek-r1:8b",
            "temperature": 0.3,
        }

        # Save preset
        presets.save_preset("test_preset", config, str(presets_file))

        # Load presets
        loaded = presets.load_presets(str(presets_file))

        assert "test_preset" in loaded
        assert loaded["test_preset"]["model"] == "deepseek-r1:8b"

    def test_delete_preset(self, tmp_path):
        """Test deleting a preset."""
        presets_file = tmp_path / "test_presets.json"

        config = {"model": "deepseek-r1:8b"}
        presets.save_preset("to_delete", config, str(presets_file))

        # Verify it exists
        loaded = presets.load_presets(str(presets_file))
        assert "to_delete" in loaded

        # Delete it
        presets.delete_preset("to_delete", str(presets_file))

        # Verify it's gone
        loaded = presets.load_presets(str(presets_file))
        assert "to_delete" not in loaded


# ============================================================================
# Tests for Session Module
# ============================================================================
# Note: Session module tests require streamlit and are better suited for
# integration tests. Skipped in unit tests to avoid streamlit dependency.
