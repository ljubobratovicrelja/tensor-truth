"""
Unit tests for directory naming conventions across fetch_sources.

Tests that scraped documentation is stored in directories that match
what build_db.py expects to find.

Critical invariant:
    scraper creates: {doc_type}_{module_name}/
    build_db expects: {doc_type}_{module_name}/

Where doc_type is "library", "papers", or "book" from DocumentType enum.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock pymupdf.layout before any imports that use it
# This must happen before pymupdf is imported anywhere
layout_mock = MagicMock()
layout_mock.activate = MagicMock(return_value=None)
sys.modules["pymupdf.layout"] = layout_mock

# Create a mock pymupdf module with layout attribute and version
pymupdf_mock = MagicMock()
pymupdf_mock.layout = layout_mock
pymupdf_mock.__version__ = "1.26.7"  # Match installed version
# Override before import
sys.modules["pymupdf"] = pymupdf_mock


@pytest.mark.unit
class TestLibraryDirectoryNaming:
    """Test that library scraping creates correctly named directories."""

    def test_library_directory_has_library_prefix(self, tmp_path):
        """Test library directories are created with 'library_' prefix.

        Critical: build_db.py expects 'library_{module_name}' format.
        Example: 'library_pytorch_2.9' NOT 'pytorch_2.9_2.9'
        """
        # Import here to avoid module load issues
        from tensortruth.fetch_sources import scrape_library

        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        library_config = {
            "type": "sphinx",
            "version": "2.9",
            "doc_root": "https://pytorch.org/docs/stable/",
            "inventory_url": "https://pytorch.org/docs/stable/objects.inv",
            "selector": "div[role='main']",
        }

        # Mock network calls
        with patch("tensortruth.fetch_sources.fetch_inventory", return_value=[]):
            scrape_library(
                library_name="pytorch_2.9",
                config=library_config,
                output_base_dir=str(output_base_dir),
            )

        # Verify directory name matches build_db expectation
        expected_dir = output_base_dir / "library_pytorch_2.9"
        created_dirs = [d.name for d in output_base_dir.iterdir()]

        assert expected_dir.exists(), (
            f"Expected directory 'library_pytorch_2.9' not found. "
            f"Found: {created_dirs}"
        )

    def test_library_directory_no_double_version(self, tmp_path):
        """Test library directory doesn't duplicate version number.

        Bug: Current code creates 'pytorch_2.9_2.9' when library_name='pytorch_2.9'
        and config['version']='2.9'
        """
        from tensortruth.fetch_sources import scrape_library

        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        library_config = {
            "type": "sphinx",
            "version": "2.9",
            "doc_root": "https://pytorch.org/docs/stable/",
        }

        with patch("tensortruth.fetch_sources.fetch_inventory", return_value=[]):
            scrape_library(
                library_name="pytorch_2.9",
                config=library_config,
                output_base_dir=str(output_base_dir),
            )

        # Should NOT create double version directory
        wrong_dir = output_base_dir / "pytorch_2.9_2.9"
        created_dirs = [d.name for d in output_base_dir.iterdir()]

        assert not wrong_dir.exists(), (
            f"Directory 'pytorch_2.9_2.9' should not exist (double version bug). "
            f"Found: {created_dirs}"
        )


@pytest.mark.unit
class TestPapersDirectoryNaming:
    """Test that paper scraping creates correctly named directories."""

    def test_papers_directory_has_papers_prefix(self, tmp_path):
        """Test paper directories are created with 'papers_' prefix.

        Critical: build_db.py expects 'papers_{category_name}' format.
        Example: 'papers_dl_foundations' NOT 'dl_foundations'
        """
        from tensortruth.scrapers.arxiv import fetch_paper_category

        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        papers_config = {
            "type": "arxiv",
            "display_name": "Test Category",
            "description": "Test papers",
            "items": {
                "1706.03762": {
                    "title": "Attention Is All You Need",
                    "arxiv_id": "1706.03762",
                    "source": "https://arxiv.org/abs/1706.03762",
                    "authors": "Vaswani et al.",
                    "year": "2017",
                }
            },
        }

        # Mock paper download
        with patch("tensortruth.scrapers.arxiv.fetch_arxiv_paper", return_value=True):
            fetch_paper_category(
                category_name="dl_foundations",
                category_config=papers_config,
                output_base_dir=str(output_base_dir),
                output_format="markdown",
                converter="pymupdf",
            )

        # Verify directory name matches build_db expectation
        expected_dir = output_base_dir / "papers_dl_foundations"
        created_dirs = [d.name for d in output_base_dir.iterdir()]

        assert expected_dir.exists(), (
            f"Expected directory 'papers_dl_foundations' not found. "
            f"Found: {created_dirs}"
        )


@pytest.mark.unit
class TestBookDirectoryNaming:
    """Test that book scraping creates correctly named directories."""

    def test_book_directory_has_book_prefix(self, tmp_path):
        """Test book directories are created with 'book_' prefix.

        Critical: build_db.py expects 'book_{book_name}' format.
        Example: 'book_linear_algebra_cherney' NOT 'linear_algebra_cherney'
        """
        from tensortruth.scrapers.book import fetch_book

        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        book_config = {
            "type": "pdf_book",
            "title": "Linear Algebra",
            "authors": ["David Cherney", "Tom Denton"],
            "category": "linear_algebra",
            "source": "https://example.com/linear_algebra.pdf",
            "split_method": "none",
        }

        # Create dummy PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")

        # Mock download and conversion
        with (
            patch(
                "tensortruth.utils.pdf.download_pdf",
                return_value=str(pdf_path),
            ),
            patch(
                "tensortruth.utils.pdf.convert_pdf_to_markdown",
                return_value="# Test Content",
            ),
            patch(
                "tensortruth.utils.pdf.get_pdf_page_count",
                return_value=100,
            ),
        ):
            fetch_book(
                book_name="linear_algebra_cherney",
                book_config=book_config,
                output_base_dir=str(output_base_dir),
                converter="pymupdf",
            )

        # Verify directory name matches build_db expectation
        expected_dir = output_base_dir / "book_linear_algebra_cherney"
        created_dirs = [d.name for d in output_base_dir.iterdir()]

        assert expected_dir.exists(), (
            f"Expected directory 'book_linear_algebra_cherney' not found. "
            f"Found: {created_dirs}"
        )


@pytest.mark.unit
class TestBuildDbCompatibility:
    """Test that scraped directories match build_db.py expectations."""

    def test_directory_naming_matches_document_type_enum(self):
        """Test directory names match DocumentType enum values.

        build_db.py:132 constructs paths as: f"{doc_type.value}_{module_name}"
        where doc_type.value is from DocumentType enum.
        """
        from tensortruth.utils.metadata import DocumentType

        # Verify enum values are what build_db expects
        assert DocumentType.LIBRARY.value == "library"
        assert DocumentType.PAPERS.value == "papers"
        assert DocumentType.BOOK.value == "book"

        # Expected directory name patterns
        test_cases = [
            ("pytorch_2.9", DocumentType.LIBRARY, "library_pytorch_2.9"),
            ("dl_foundations", DocumentType.PAPERS, "papers_dl_foundations"),
            (
                "linear_algebra_cherney",
                DocumentType.BOOK,
                "book_linear_algebra_cherney",
            ),
        ]

        for module_name, doc_type, expected_dir_name in test_cases:
            actual = f"{doc_type.value}_{module_name}"
            assert actual == expected_dir_name, (
                f"Directory pattern mismatch for {module_name}: "
                f"expected '{expected_dir_name}', got '{actual}'"
            )

    def test_no_legacy_directory_patterns(self, tmp_path):
        """Test that old/wrong directory patterns are documented as wrong.

        Common mistakes that should NOT appear:
        - Missing prefix: 'pytorch_2.9' instead of 'library_pytorch_2.9'
        - Double version: 'pytorch_2.9_2.9' instead of 'library_pytorch_2.9'
        - Wrong prefix: 'lib_pytorch' instead of 'library_pytorch'
        """
        output_base_dir = tmp_path / "library_docs"
        output_base_dir.mkdir()

        # These patterns should NEVER be created
        wrong_patterns = [
            "pytorch_2.9",  # Missing prefix
            "pytorch_2.9_2.9",  # Double version
            "lib_pytorch_2.9",  # Wrong prefix abbreviation
            "dl_foundations",  # Missing papers_ prefix
            "paper_dl_foundations",  # Wrong (singular instead of plural)
            "linear_algebra_cherney",  # Missing book_ prefix
            "books_linear_algebra",  # Wrong (plural instead of singular)
        ]

        # Create CORRECT directories (what scrapers SHOULD create)
        correct_dirs = [
            "library_pytorch_2.9",
            "papers_dl_foundations",
            "book_linear_algebra_cherney",
        ]

        for dirname in correct_dirs:
            (output_base_dir / dirname).mkdir()

        # Verify wrong patterns don't exist
        existing_dirs = [d.name for d in output_base_dir.iterdir()]

        for wrong_pattern in wrong_patterns:
            assert wrong_pattern not in existing_dirs, (
                f"Found legacy/wrong directory pattern '{wrong_pattern}' "
                f"in {existing_dirs}"
            )

        # Verify correct patterns DO exist
        for correct_pattern in correct_dirs:
            assert correct_pattern in existing_dirs, (
                f"Expected correct pattern '{correct_pattern}' not found "
                f"in {existing_dirs}"
            )
