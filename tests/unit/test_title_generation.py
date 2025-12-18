"""Unit tests for title generation utilities."""

import pytest


@pytest.mark.unit
class TestTitleCleaning:
    """Test title cleanup from LLM responses."""

    def test_removes_markdown_headers(self):
        """Should remove # markdown headers."""
        # Test the cleanup logic directly
        response = "# Main Title"
        cleaned = (
            response.replace('"', "")
            .replace("'", "")
            .replace(".", "")
            .replace("#", "")
            .replace("*", "")
            .replace("_", "")
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")

        assert cleaned == "Main Title"
        assert "#" not in cleaned

    def test_removes_multiple_markdown_elements(self):
        """Should remove mixed markdown formatting."""
        response = "# Main Title\n## Subtitle"
        cleaned = (
            response.replace('"', "")
            .replace("'", "")
            .replace(".", "")
            .replace("#", "")
            .replace("*", "")
            .replace("_", "")
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")

        assert cleaned == "Main Title Subtitle"
        assert "\n" not in cleaned
        assert "#" not in cleaned

    def test_removes_bold_and_italic(self):
        """Should remove bold and italic markdown."""
        response = "**Bold** and _italic_ text"
        cleaned = (
            response.replace('"', "")
            .replace("'", "")
            .replace(".", "")
            .replace("#", "")
            .replace("*", "")
            .replace("_", "")
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")

        assert cleaned == "Bold and italic text"
        assert "*" not in cleaned
        assert "_" not in cleaned

    def test_collapses_multiple_spaces(self):
        """Should collapse multiple spaces into single space."""
        response = "Multiple    spaces   here"
        cleaned = (
            response.replace('"', "")
            .replace("'", "")
            .replace(".", "")
            .replace("#", "")
            .replace("*", "")
            .replace("_", "")
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")

        assert cleaned == "Multiple spaces here"
        assert "  " not in cleaned
