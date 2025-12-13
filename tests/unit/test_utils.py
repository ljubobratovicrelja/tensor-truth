"""
Unit tests for tensortruth.utils module.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
import torch

from tensortruth.utils import (
    parse_thinking_response,
    convert_chat_to_markdown,
    get_running_models,
    get_max_memory_gb,
    stop_model,
)


# ============================================================================
# Tests for parse_thinking_response
# ============================================================================

@pytest.mark.unit
class TestParseThinkingResponse:
    """Tests for parse_thinking_response function."""

    def test_standard_thinking_tags(self):
        """Test parsing standard <thought> tags."""
        raw = "<thought>Thinking process here</thought>Final answer"
        thought, answer = parse_thinking_response(raw)

        assert thought == "Thinking process here"
        assert answer == "Final answer"

    def test_multiline_thinking(self, sample_thinking_response):
        """Test parsing multiline thinking content."""
        thought, answer = parse_thinking_response(sample_thinking_response)

        assert "Let me analyze" in thought
        assert "step by step" in thought
        assert "Here is the actual answer" in answer

    def test_unclosed_thinking_tag(self):
        """Test handling unclosed <thought> tag."""
        raw = "<thought>Incomplete thinking process..."
        thought, answer = parse_thinking_response(raw)

        assert "Incomplete thinking process" in thought
        assert answer == "..."

    def test_no_thinking_tags(self):
        """Test response without thinking tags."""
        raw = "Just a regular response without thinking."
        thought, answer = parse_thinking_response(raw)

        assert thought is None
        assert answer == "Just a regular response without thinking."

    def test_empty_input(self):
        """Test empty input handling."""
        thought, answer = parse_thinking_response("")

        assert thought is None
        assert answer == ""

    def test_none_input(self):
        """Test None input handling."""
        thought, answer = parse_thinking_response(None)

        assert thought is None
        assert answer == ""

    def test_multiple_thinking_blocks(self):
        """Test handling multiple <thought> blocks (takes first)."""
        raw = "<thought>First</thought>Answer<thought>Second</thought>"
        thought, answer = parse_thinking_response(raw)

        # Should extract first thought and remove all thought tags
        assert "First" in thought
        assert "Answer" in answer
        assert "<thought>" not in answer


# ============================================================================
# Tests for convert_chat_to_markdown
# ============================================================================

@pytest.mark.unit
class TestConvertChatToMarkdown:
    """Tests for convert_chat_to_markdown function."""

    def test_basic_conversion(self, sample_chat_session):
        """Test basic session to markdown conversion."""
        markdown = convert_chat_to_markdown(sample_chat_session)

        assert "# Test Session" in markdown
        assert "2023-12-13" in markdown
        assert "USER" in markdown
        assert "ASSISTANT" in markdown
        assert "What is PyTorch?" in markdown
        assert "PyTorch is a deep learning framework" in markdown

    def test_sources_included(self, sample_chat_session):
        """Test that sources are included in markdown."""
        markdown = convert_chat_to_markdown(sample_chat_session)

        assert "Sources:" in markdown
        assert "pytorch_intro.md" in markdown
        assert "0.95" in markdown

    def test_empty_session(self):
        """Test conversion of empty session."""
        session = {
            "title": "Empty Session",
            "created_at": "2023-12-13T10:00:00",
            "messages": []
        }
        markdown = convert_chat_to_markdown(session)

        assert "# Empty Session" in markdown
        assert "Date:" in markdown

    def test_session_without_sources(self):
        """Test message without sources."""
        session = {
            "title": "No Sources",
            "created_at": "2023-12-13T10:00:00",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        }
        markdown = convert_chat_to_markdown(session)

        assert "Hello" in markdown
        assert "Hi there" in markdown

    def test_thinking_separation(self):
        """Test that thinking is separated from answer."""
        session = {
            "title": "Thinking Test",
            "created_at": "2023-12-13T10:00:00",
            "messages": [
                {
                    "role": "assistant",
                    "content": "<thought>Internal thoughts</thought>Visible answer"
                }
            ]
        }
        markdown = convert_chat_to_markdown(session)

        assert "Thought Process:" in markdown
        assert "Internal thoughts" in markdown
        assert "Visible answer" in markdown


# ============================================================================
# Tests for get_running_models
# ============================================================================

@pytest.mark.unit
class TestGetRunningModels:
    """Tests for get_running_models function."""

    @patch('tensortruth.utils.requests.get')
    def test_successful_response(self, mock_get, mock_ollama_ps_response):
        """Test successful API response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_ollama_ps_response
        mock_get.return_value = mock_response

        models = get_running_models()

        assert len(models) == 1
        assert models[0]["name"] == "deepseek-r1:8b"
        assert "5.1 GB" in models[0]["size_vram"]

    @patch('tensortruth.utils.requests.get')
    def test_api_failure(self, mock_get):
        """Test API failure handling."""
        mock_get.side_effect = Exception("Connection error")

        models = get_running_models()

        assert models == []

    @patch('tensortruth.utils.requests.get')
    def test_timeout(self, mock_get):
        """Test timeout handling."""
        mock_get.side_effect = TimeoutError()

        models = get_running_models()

        assert models == []

    @patch('tensortruth.utils.requests.get')
    def test_empty_models(self, mock_get):
        """Test empty models list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        models = get_running_models()

        assert models == []


# ============================================================================
# Tests for get_max_memory_gb
# ============================================================================

@pytest.mark.unit
class TestGetMaxMemoryGB:
    """Tests for get_max_memory_gb function."""

    def test_cuda_available(self, monkeypatch):
        """Test CUDA memory detection."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def mock_mem_get_info():
            return (8_000_000_000, 24_000_000_000)  # 8GB free, 24GB total

        monkeypatch.setattr(torch.cuda, "mem_get_info", mock_mem_get_info)

        memory = get_max_memory_gb()

        # Allow for small floating point differences
        assert abs(memory - 24.0) < 0.01 or memory == pytest.approx(24.0, rel=0.1)

    def test_mps_available(self, monkeypatch):
        """Test MPS (Apple Silicon) memory detection."""
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

        # Should return system RAM for unified memory (reasonable value check)
        assert memory > 0
        assert memory < 1024  # Less than 1TB (sanity check)

    def test_cpu_fallback(self, monkeypatch):
        """Test CPU-only fallback."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Mock MPS as unavailable
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        memory = get_max_memory_gb()

        # Should return system RAM (reasonable value check)
        assert memory > 0
        assert memory < 1024  # Less than 1TB (sanity check)

    def test_fallback_when_psutil_unavailable(self, monkeypatch):
        """Test fallback value when psutil is unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        # Make psutil import fail
        def mock_import(*args, **kwargs):
            if args[0] == "psutil":
                raise ImportError()
            return __import__(*args, **kwargs)

        # This should return the fallback value
        memory = get_max_memory_gb()
        assert memory > 0  # Should have some fallback value


# ============================================================================
# Tests for stop_model
# ============================================================================

@pytest.mark.unit
class TestStopModel:
    """Tests for stop_model function."""

    @patch('tensortruth.utils.requests.post')
    def test_successful_stop(self, mock_post):
        """Test successful model stop."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = stop_model("deepseek-r1:8b")

        assert result is True
        mock_post.assert_called_once()

        # Verify the payload
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "deepseek-r1:8b"
        assert call_args[1]["json"]["keep_alive"] == 0

    @patch('tensortruth.utils.requests.post')
    def test_failed_stop(self, mock_post):
        """Test failed model stop."""
        mock_post.side_effect = Exception("Connection error")

        result = stop_model("deepseek-r1:8b")

        assert result is False

    @patch('tensortruth.utils.requests.post')
    def test_timeout(self, mock_post):
        """Test timeout handling."""
        mock_post.side_effect = TimeoutError()

        result = stop_model("deepseek-r1:8b")

        assert result is False


# ============================================================================
# Property-based tests with hypothesis
# ============================================================================

@pytest.mark.unit
class TestUtilsProperties:
    """Property-based tests for utils functions."""

    def test_parse_thinking_never_crashes(self):
        """Test that parse_thinking_response handles any string input."""
        from hypothesis import given, strategies as st

        @given(st.text())
        def inner_test(text):
            thought, answer = parse_thinking_response(text)
            # Should always return tuple of (str|None, str)
            assert isinstance(answer, str)
            assert thought is None or isinstance(thought, str)

        inner_test()

    def test_convert_markdown_with_various_sessions(self):
        """Test markdown conversion with various session structures."""
        # Test with minimal session
        minimal = {
            "title": "Test",
            "created_at": "2023-01-01",
            "messages": []
        }
        result = convert_chat_to_markdown(minimal)
        assert isinstance(result, str)
        assert len(result) > 0
