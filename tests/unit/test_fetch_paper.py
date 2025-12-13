"""
Unit tests for tensortruth.fetch_paper module.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from tensortruth.fetch_paper import (
    book_already_processed,
    clean_filename,
    detect_category_type,
    paper_already_processed,
    post_process_math,
)

# ============================================================================
# Tests for clean_filename
# ============================================================================


@pytest.mark.unit
class TestCleanFilename:
    """Tests for clean_filename function."""

    def test_basic_sanitization(self):
        """Test basic filename sanitization."""
        result = clean_filename("Test Paper: A Study")
        assert result == "Test_Paper__A_Study"

    def test_special_characters(self):
        """Test removal of special characters."""
        result = clean_filename("Paper@#$%^&*()Name")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert all(c.isalnum() or c == "_" for c in result)

    def test_truncation(self):
        """Test that long titles are truncated."""
        long_title = "A" * 100
        result = clean_filename(long_title)
        assert len(result) == 50

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        result = clean_filename("Café résumé naïve")
        assert isinstance(result, str)
        assert len(result) <= 50

    def test_empty_string(self):
        """Test empty string handling."""
        result = clean_filename("")
        assert isinstance(result, str)

    def test_whitespace_only(self):
        """Test whitespace-only input."""
        result = clean_filename("   \t\n  ")
        assert isinstance(result, str)


# ============================================================================
# Tests for paper_already_processed
# ============================================================================


@pytest.mark.unit
class TestPaperAlreadyProcessed:
    """Tests for paper_already_processed function."""

    def test_paper_not_processed_no_directory(self, temp_dir):
        """Test when output directory doesn't exist."""
        result = paper_already_processed(
            "test_category", "1234.56789", root_dir=str(temp_dir)
        )
        assert result is False

    def test_paper_not_processed_no_pdf(self, temp_dir):
        """Test when PDF doesn't exist."""
        category_dir = temp_dir / "test_category"
        category_dir.mkdir()

        result = paper_already_processed(
            "test_category", "1234.56789", root_dir=str(temp_dir)
        )
        assert result is False

    def test_paper_processed(self, temp_dir):
        """Test when paper is already processed."""
        category_dir = temp_dir / "test_category"
        category_dir.mkdir()

        # Create PDF
        pdf_path = category_dir / "1234.56789.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        # Create MD with arxiv_id in header
        md_path = category_dir / "test_paper.md"
        md_path.write_text(f"# ArXiv ID: 1234.56789\n\nContent...")

        result = paper_already_processed(
            "test_category", "1234.56789", root_dir=str(temp_dir)
        )
        assert result is True

    def test_paper_pdf_exists_but_no_matching_md(self, temp_dir):
        """Test when PDF exists but no matching MD file."""
        category_dir = temp_dir / "test_category"
        category_dir.mkdir()

        # Create PDF
        pdf_path = category_dir / "1234.56789.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        # Create MD without arxiv_id
        md_path = category_dir / "other_paper.md"
        md_path.write_text("# Some other paper\n\nContent...")

        result = paper_already_processed(
            "test_category", "1234.56789", root_dir=str(temp_dir)
        )
        assert result is False


# ============================================================================
# Tests for book_already_processed
# ============================================================================


@pytest.mark.unit
class TestBookAlreadyProcessed:
    """Tests for book_already_processed function."""

    def test_book_not_processed(self, temp_dir):
        """Test when book is not processed."""
        result = book_already_processed(
            "test_category", "Test Book", root_dir=str(temp_dir)
        )
        assert result is False

    def test_book_processed(self, temp_dir):
        """Test when book is already processed."""
        category_dir = temp_dir / "test_category"
        category_dir.mkdir()

        # Create markdown file for book
        safe_title = "Test_Book"
        md_path = category_dir / f"{safe_title}.md"
        md_path.write_text("# Test Book\n\nContent...")

        result = book_already_processed(
            "test_category", "Test Book", root_dir=str(temp_dir)
        )
        assert result is True

    def test_book_with_chapters(self, temp_dir):
        """Test when book has multiple chapter files."""
        category_dir = temp_dir / "test_category"
        category_dir.mkdir()

        safe_title = "Test_Book"
        # Create chapter files
        for i in range(3):
            md_path = category_dir / f"{safe_title}_{i:02d}_chapter.md"
            md_path.write_text(f"# Chapter {i}\n\nContent...")

        result = book_already_processed(
            "test_category", "Test Book", root_dir=str(temp_dir)
        )
        assert result is True


# ============================================================================
# Tests for post_process_math
# ============================================================================


@pytest.mark.unit
class TestPostProcessMath:
    """Tests for post_process_math function."""

    def test_unicode_symbol_conversion(self):
        """Test conversion of unicode math symbols."""
        input_text = "The sum is ∑ and product is ∏"
        result = post_process_math(input_text)

        assert r"\sum" in result
        assert r"\prod" in result

    def test_greek_letters(self):
        """Test conversion of greek letters."""
        input_text = "Variables: α, β, γ, δ"
        result = post_process_math(input_text)

        assert r"\alpha" in result
        assert r"\beta" in result
        assert r"\gamma" in result
        assert r"\delta" in result

    def test_math_operators(self):
        """Test conversion of math operators."""
        input_text = "a × b ÷ c ≤ d ≥ e"
        result = post_process_math(input_text)

        assert r"\times" in result
        assert r"\div" in result
        assert r"\leq" in result
        assert r"\geq" in result

    def test_preserves_existing_latex(self):
        """Test that existing $...$ blocks are preserved."""
        input_text = "$E = mc^2$ and some text with α"
        result = post_process_math(input_text)

        # Should preserve existing latex
        assert "$E = mc^2$" in result
        # Should still convert unicode outside latex
        assert r"\alpha" in result or "α" in result

    def test_empty_input(self):
        """Test empty input handling."""
        result = post_process_math("")
        assert result == ""

    def test_none_input(self):
        """Test None input handling."""
        result = post_process_math(None)
        assert result is None

    def test_text_without_math(self):
        """Test text without math symbols."""
        input_text = "Regular text without math"
        result = post_process_math(input_text)

        assert result == input_text


# ============================================================================
# Tests for detect_category_type
# ============================================================================


@pytest.mark.unit
class TestDetectCategoryType:
    """Tests for detect_category_type function."""

    def test_papers_category(self):
        """Test detection of papers category."""
        category_data = {
            "description": "Research papers",
            "items": [
                {
                    "title": "Test Paper",
                    "arxiv_id": "1234.56789",
                    "url": "https://arxiv.org/abs/1234.56789",
                }
            ],
        }

        result = detect_category_type(category_data)
        assert result == "papers"

    def test_books_category(self):
        """Test detection of books category."""
        category_data = {
            "description": "Textbooks",
            "items": [
                {
                    "title": "Test Book",
                    "source": "https://example.com/book.pdf",
                    "split_method": "none",
                }
            ],
        }

        result = detect_category_type(category_data)
        assert result == "books"

    def test_empty_category(self):
        """Test empty category defaults to papers."""
        category_data = {"description": "Empty category", "items": []}

        result = detect_category_type(category_data)
        assert result == "papers"

    def test_no_items_key(self):
        """Test category without items key."""
        category_data = {"description": "No items"}

        result = detect_category_type(category_data)
        assert result == "papers"


# ============================================================================
# Property-based tests
# ============================================================================


@pytest.mark.unit
class TestFetchPaperProperties:
    """Property-based tests for fetch_paper functions."""

    def test_clean_filename_never_crashes(self):
        """Test that clean_filename handles any string input."""
        from hypothesis import given
        from hypothesis import strategies as st

        @given(st.text(max_size=1000))
        def inner_test(title):
            result = clean_filename(title)
            assert isinstance(result, str)
            assert len(result) <= 50

        inner_test()
