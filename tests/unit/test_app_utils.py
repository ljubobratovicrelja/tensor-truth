"""
Unit tests for tensortruth.app_utils modules.
"""

from unittest.mock import AsyncMock, MagicMock, patch

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

    @patch("tensortruth.core.ollama.get_available_models")
    def test_get_ollama_models_success(self, mock_get_available):
        """Test successful Ollama model fetch."""
        mock_get_available.return_value = ["deepseek-r1:8b", "llama2:7b"]

        models = get_ollama_models()

        assert len(models) == 2
        assert "deepseek-r1:8b" in models
        assert "llama2:7b" in models

    @patch("tensortruth.core.ollama.get_available_models")
    def test_get_ollama_models_failure(self, mock_get_available):
        """Test Ollama model fetch when service is down."""
        # Clear any Streamlit cache that might exist on the function
        if hasattr(get_ollama_models, "clear"):
            get_ollama_models.clear()

        # Mock get_available_models to return empty list (as it does when Ollama is down)
        mock_get_available.return_value = []

        models = get_ollama_models()

        # Should return empty list when Ollama is unavailable
        assert models == []


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

    @patch("tensortruth.app_utils.title_generation.aiohttp.ClientSession")
    def test_generate_smart_title_success(self, mock_session_class):
        """Test successful title generation."""
        # Mock model availability check (GET request)
        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(
            return_value={"models": [{"name": "qwen2.5:0.5b"}]}
        )

        # Mock title generation response (POST request)
        mock_post_response = AsyncMock()
        mock_post_response.status = 200
        mock_post_response.json = AsyncMock(return_value={"response": "PyTorch Basics"})

        # Create mock session that handles both GET and POST
        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_get_response
        mock_session.post.return_value.__aenter__.return_value = mock_post_response
        mock_session_class.return_value.__aenter__.return_value = mock_session

        title = title_generation.generate_smart_title(
            "What are the basics of PyTorch?", "deepseek-r1:8b"
        )

        assert title == "PyTorch Basics"

    @patch("tensortruth.app_utils.title_generation.aiohttp.ClientSession")
    def test_generate_smart_title_model_unavailable(self, mock_session_class):
        """Test title generation when model is unavailable."""
        # Mock model availability check - model not found
        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={"models": []})

        # Make model pull fail to simulate unavailability
        mock_post_response = AsyncMock()
        mock_post_response.status = 500

        # Create mock session that handles both GET and POST
        mock_session = MagicMock()
        mock_session.get.return_value.__aenter__.return_value = mock_get_response
        mock_session.post.return_value.__aenter__.return_value = mock_post_response
        mock_session_class.return_value.__aenter__.return_value = mock_session

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


@pytest.mark.unit
class TestSessionModule:
    """Tests for app_utils.session module."""

    def test_save_sessions_filters_empty_sessions(self, tmp_path, monkeypatch):
        """Test that save_sessions filters out empty sessions but keeps current session."""
        import json

        import streamlit as st

        from tensortruth.app_utils.session import save_sessions

        # Mock streamlit session state as an object with chat_data attribute
        class MockSessionState:
            def __init__(self):
                self.chat_data = {
                    "current_id": None,
                    "sessions": {
                        "empty1": {"title": "Empty Session 1", "messages": []},
                        "with_content": {
                            "title": "Session with content",
                            "messages": [{"role": "user", "content": "Hello"}],
                        },
                        "empty2": {"title": "Empty Session 2", "messages": []},
                    },
                }

        # Mock st.session_state
        monkeypatch.setattr(st, "session_state", MockSessionState())

        sessions_file = tmp_path / "test_sessions.json"

        # Call save_sessions
        save_sessions(str(sessions_file))

        # Verify the file was created and contains filtered sessions
        assert sessions_file.exists()

        with open(sessions_file, "r") as f:
            saved_data = json.load(f)

        # Should only have the session with content
        assert len(saved_data["sessions"]) == 1
        assert "with_content" in saved_data["sessions"]
        assert "empty1" not in saved_data["sessions"]
        assert "empty2" not in saved_data["sessions"]

    def test_save_sessions_preserves_current_session(self, tmp_path, monkeypatch):
        """Test that save_sessions preserves current session even if empty."""
        import json

        import streamlit as st

        from tensortruth.app_utils.session import save_sessions

        # Mock streamlit session state with current session
        class MockSessionState:
            def __init__(self):
                self.chat_data = {
                    "current_id": "empty1",  # This is the current session
                    "sessions": {
                        "empty1": {"title": "Current Empty Session", "messages": []},
                        "with_content": {
                            "title": "Session with content",
                            "messages": [{"role": "user", "content": "Hello"}],
                        },
                        "empty2": {"title": "Empty Session 2", "messages": []},
                    },
                }

        # Mock st.session_state
        monkeypatch.setattr(st, "session_state", MockSessionState())

        sessions_file = tmp_path / "test_sessions.json"

        # Call save_sessions
        save_sessions(str(sessions_file))

        # Verify the file was created and contains current session + session with content
        assert sessions_file.exists()

        with open(sessions_file, "r") as f:
            saved_data = json.load(f)

        # Should have current session and session with content
        assert len(saved_data["sessions"]) == 2
        assert "empty1" in saved_data["sessions"]  # Current session preserved
        assert (
            "with_content" in saved_data["sessions"]
        )  # Session with content preserved
        assert (
            "empty2" not in saved_data["sessions"]
        )  # Other empty session filtered out

    def test_save_sessions_all_empty_except_current(self, tmp_path, monkeypatch):
        """Test edge case where only current session exists and it's empty."""
        import json

        import streamlit as st

        from tensortruth.app_utils.session import save_sessions

        # Mock streamlit session state with only current empty session
        class MockSessionState:
            def __init__(self):
                self.chat_data = {
                    "current_id": "only_session",
                    "sessions": {
                        "only_session": {"title": "Only Empty Session", "messages": []}
                    },
                }

        # Mock st.session_state
        monkeypatch.setattr(st, "session_state", MockSessionState())

        sessions_file = tmp_path / "test_sessions.json"

        # Call save_sessions
        save_sessions(str(sessions_file))

        # Verify the file was created and contains the current session
        assert sessions_file.exists()

        with open(sessions_file, "r") as f:
            saved_data = json.load(f)

        # Should still have the current session even though it's empty
        assert len(saved_data["sessions"]) == 1
        assert "only_session" in saved_data["sessions"]


# Note: Additional session tests that require full streamlit context are
# better suited for integration tests.
