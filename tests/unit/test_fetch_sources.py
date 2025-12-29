"""
Unit tests for fetch_sources.py validation and helper functions.

Tests the core validation logic, sanitization, and ArXiv ID handling
that is critical for the interactive source addition feature.
"""

from unittest.mock import patch

import pytest

from tensortruth.fetch_sources import (
    sanitize_config_key,
    validate_arxiv_id,
    validate_url,
)


@pytest.mark.unit
class TestValidateUrl:
    """Tests for URL validation function."""

    def test_valid_https_url(self):
        """Test validation of a standard HTTPS URL."""
        assert validate_url("https://pytorch.org/docs/stable/") is True

    @patch("requests.head")
    def test_valid_http_url(self, mock_head):
        """Test validation of HTTP URL."""
        mock_head.return_value.status_code = 200
        assert validate_url("http://example.com/docs") is True

    @patch("requests.head")
    def test_url_with_port(self, mock_head):
        """Test URL with explicit port number."""
        mock_head.return_value.status_code = 200
        assert validate_url("https://localhost:8080/docs") is True

    def test_url_with_path(self):
        """Test URL with complex path."""
        assert validate_url("https://docs.python.org/3/library/index.html") is True

    def test_invalid_url_missing_protocol(self):
        """Test that URL without protocol is rejected."""
        assert validate_url("example.com/docs") is False

    def test_invalid_url_malformed(self):
        """Test that malformed URL is rejected."""
        assert validate_url("ht!tp://not a url") is False

    def test_empty_url(self):
        """Test that empty string is rejected."""
        assert validate_url("") is False

    @patch("requests.head")
    def test_url_accessibility_check_success(self, mock_head):
        """Test that accessible URLs pass HTTP check."""
        mock_head.return_value.status_code = 200
        assert validate_url("https://example.com") is True

    @patch("requests.head")
    def test_url_accessibility_check_404(self, mock_head):
        """Test that 404 URLs are rejected."""
        mock_head.return_value.status_code = 404
        assert validate_url("https://example.com/nonexistent") is False

    @patch("requests.head")
    @patch("requests.get")
    def test_fallback_to_get_on_head_failure(self, mock_get, mock_head):
        """Test that GET is tried when HEAD fails."""
        mock_head.side_effect = Exception("HEAD not allowed")
        mock_get.return_value.status_code = 200
        assert validate_url("https://example.com") is True

    @patch("requests.head")
    @patch("requests.get")
    def test_network_error_returns_false(self, mock_get, mock_head):
        """Test that network errors result in rejection."""
        mock_head.side_effect = Exception("Connection timeout")
        mock_get.side_effect = Exception("Connection timeout")
        assert validate_url("https://unreachable.example.com") is False


@pytest.mark.unit
class TestSanitizeConfigKey:
    """Tests for config key sanitization."""

    def test_lowercase_conversion(self):
        """Test that uppercase letters are converted to lowercase."""
        assert sanitize_config_key("PyTorch") == "pytorch"

    def test_space_to_underscore(self):
        """Test that spaces are replaced with underscores."""
        assert sanitize_config_key("Deep Learning") == "deep_learning"

    def test_special_characters_removed(self):
        """Test that special characters are replaced with underscores."""
        assert sanitize_config_key("C++") == "c"
        assert sanitize_config_key("Rust@2024") == "rust_2024"

    def test_multiple_underscores_collapsed(self):
        """Test that consecutive underscores are collapsed to one."""
        assert sanitize_config_key("foo___bar") == "foo_bar"

    def test_leading_trailing_underscores_removed(self):
        """Test that leading/trailing underscores are stripped."""
        assert sanitize_config_key("_test_") == "test"

    def test_hyphen_preserved(self):
        """Test that hyphens are preserved in the key."""
        assert sanitize_config_key("deep-learning") == "deep-learning"

    def test_complex_sanitization(self):
        """Test complex sanitization with multiple transformations."""
        assert sanitize_config_key("PyTorch 2.0 (Beta)") == "pytorch_2_0_beta"

    def test_alphanumeric_only(self):
        """Test that alphanumeric strings pass through unchanged."""
        assert sanitize_config_key("pytorch20") == "pytorch20"


@pytest.mark.unit
class TestValidateArxivId:
    """Tests for ArXiv ID validation."""

    def test_new_format_valid(self):
        """Test validation of new ArXiv ID format (YYMM.NNNNN)."""
        assert validate_arxiv_id("1706.03762") == "1706.03762"
        assert validate_arxiv_id("2010.11929") == "2010.11929"

    def test_new_format_five_digits(self):
        """Test new format with 5-digit paper number."""
        assert validate_arxiv_id("2301.12345") == "2301.12345"

    def test_new_format_four_digits(self):
        """Test new format with 4-digit paper number."""
        assert validate_arxiv_id("1512.0338") == "1512.0338"

    def test_old_format_valid(self):
        """Test validation of old ArXiv ID format (arch-ive/NNNNNNN)."""
        assert validate_arxiv_id("cs/0501001") == "cs/0501001"
        assert validate_arxiv_id("math/0309136") == "math/0309136"

    def test_extract_from_abs_url(self):
        """Test extraction from arxiv.org/abs/ URL."""
        url = "https://arxiv.org/abs/1706.03762"
        assert validate_arxiv_id(url) == "1706.03762"

    def test_extract_from_pdf_url(self):
        """Test extraction from arxiv.org/pdf/ URL."""
        url = "https://arxiv.org/pdf/2010.11929.pdf"
        assert validate_arxiv_id(url) == "2010.11929"

    def test_extract_old_format_from_url(self):
        """Test extraction of old format ID from URL."""
        url = "https://arxiv.org/abs/cs/0501001"
        assert validate_arxiv_id(url) == "cs/0501001"

    def test_invalid_format_returns_none(self):
        """Test that invalid formats return None."""
        assert validate_arxiv_id("not-an-arxiv-id") is None
        assert validate_arxiv_id("123.456") is None  # Wrong length
        assert validate_arxiv_id("abcd.1234") is None  # Non-numeric year

    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        assert validate_arxiv_id("") is None

    def test_whitespace_stripped(self):
        """Test that leading/trailing whitespace is handled."""
        assert validate_arxiv_id("  1706.03762  ") == "1706.03762"

    def test_invalid_url_returns_none(self):
        """Test that non-arxiv URLs return None."""
        assert validate_arxiv_id("https://example.com/paper.pdf") is None
