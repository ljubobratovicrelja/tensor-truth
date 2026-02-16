"""Unit tests for title generation utilities."""

import pytest

from tensortruth.app_utils.title_generation import (
    _clean_title,
    _make_fallback_title,
    _strip_markdown,
)


@pytest.mark.unit
class TestStripMarkdown:
    """Test _strip_markdown helper."""

    def test_removes_heading_prefixes(self):
        assert _strip_markdown("# Hello World") == "Hello World"
        assert _strip_markdown("## Subtitle Here") == "Subtitle Here"
        assert _strip_markdown("### Deep Heading") == "Deep Heading"

    def test_removes_bold_and_italic(self):
        assert _strip_markdown("**Bold** and _italic_") == "Bold and italic"
        assert _strip_markdown("***both***") == "both"

    def test_removes_inline_code(self):
        assert _strip_markdown("Use `print()` here") == "Use print() here"

    def test_removes_code_fences(self):
        text = "Before\n```python\nprint('hi')\n```\nAfter"
        result = _strip_markdown(text)
        assert "```" not in result
        assert "After" in result

    def test_removes_link_syntax(self):
        assert _strip_markdown("[click here](https://example.com)") == "click here"

    def test_removes_blockquotes(self):
        assert _strip_markdown("> Some quoted text") == "Some quoted text"

    def test_removes_list_markers(self):
        assert _strip_markdown("- Item one") == "Item one"
        assert _strip_markdown("* Item two") == "Item two"
        assert _strip_markdown("1. First item") == "First item"

    def test_collapses_whitespace(self):
        assert _strip_markdown("Multiple    spaces   here") == "Multiple spaces here"

    def test_removes_quotes(self):
        assert _strip_markdown('"Quoted"') == "Quoted"
        assert _strip_markdown("'Single'") == "Single"

    def test_newlines_become_spaces(self):
        assert _strip_markdown("Line one\nLine two") == "Line one Line two"

    def test_truncates_input_to_200_chars(self):
        long_text = "a" * 300
        result = _strip_markdown(long_text)
        assert len(result) <= 200

    def test_mixed_markdown(self):
        text = (
            "## What Is **Backpropagation**?\n\nBackpropagation is a _key_ algorithm."
        )
        result = _strip_markdown(text)
        assert "#" not in result
        assert "**" not in result
        assert "_" not in result
        assert "Backpropagation" in result


@pytest.mark.unit
class TestCleanTitle:
    """Test _clean_title helper (strip + word-limit, returns None if empty)."""

    def test_returns_cleaned_title(self):
        assert _clean_title("# Hello World") == "Hello World"

    def test_enforces_word_limit(self):
        result = _clean_title("One Two Three Four Five Six Seven")
        assert result == "One Two Three Four Five"

    def test_custom_word_limit(self):
        result = _clean_title("One Two Three Four", max_words=2)
        assert result == "One Two"

    def test_returns_none_for_empty(self):
        assert _clean_title("") is None
        assert _clean_title("   ") is None
        assert _clean_title("```\n```") is None


@pytest.mark.unit
class TestMakeFallbackTitle:
    """Test _make_fallback_title helper."""

    def test_short_text_unchanged(self):
        assert _make_fallback_title("Hello World") == "Hello World"

    def test_long_text_truncated_on_word_boundary(self):
        long = "What Is Backpropagation and Why Does It Matter So Much In Modern Deep Learning"
        title = _make_fallback_title(long)
        assert len(title) <= 62  # max_len(60) + len("..")
        assert title.endswith("..")
        assert not title.endswith(" ..")  # no trailing space before ..

    def test_heading_prefixed_text(self):
        title = _make_fallback_title(
            "## What Is Backpropagation? Backpropagation is the engine behind modern deep learning systems"
        )
        assert not title.startswith("#")
        assert title.endswith("..")

    def test_empty_after_stripping_returns_new_chat(self):
        assert _make_fallback_title("```\n```") == "New Chat"
        assert _make_fallback_title("") == "New Chat"
        assert _make_fallback_title("   ") == "New Chat"

    def test_custom_max_len(self):
        title = _make_fallback_title("This is a somewhat longer title text", max_len=15)
        assert len(title) <= 17  # max_len + len("..")
        assert title.endswith("..")

    def test_strips_markdown_before_truncating(self):
        title = _make_fallback_title(
            "# Backpropagation – The Engine of Neural Networks and How It Powers Modern AI Systems"
        )
        assert "#" not in title
        assert title.endswith("..")

    def test_exact_boundary_no_suffix(self):
        """Text exactly at max_len should not get .. suffix."""
        text = "a" * 60
        assert _make_fallback_title(text) == text


@pytest.mark.unit
class TestGenerateSmartTitleAsync:
    """Test the public generate_smart_title_async function."""

    @pytest.mark.asyncio
    async def test_no_model_returns_fallback(self):
        from tensortruth.app_utils.title_generation import generate_smart_title_async

        result = await generate_smart_title_async(
            "## Hello World of AI", model_name=None
        )
        assert "#" not in result
        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_no_model_empty_text_returns_new_chat(self):
        from tensortruth.app_utils.title_generation import generate_smart_title_async

        result = await generate_smart_title_async("", model_name=None)
        assert result == "New Chat"
