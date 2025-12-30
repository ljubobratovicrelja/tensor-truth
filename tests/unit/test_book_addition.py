"""
Tests for interactive book addition feature.

Tests cover PDF metadata extraction, interactive prompts, validation,
and configuration management for adding books to sources.json.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.fetch_sources import (
    add_book_interactive,
    download_pdf_with_headers,
    extract_pdf_metadata,
    generate_book_name,
)


@pytest.mark.unit
class TestExtractPdfMetadata:
    """Tests for extract_pdf_metadata function (to be implemented)."""

    def test_extract_title_and_authors_from_pdf(self, create_test_pdf):
        """Test extraction of title and authors from PDF metadata."""

        # Mock PyMuPDF metadata extraction
        with patch("fitz.open") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.metadata = {
                "title": "Introduction to Machine Learning",
                "author": "Author One; Author Two",
            }
            mock_fitz.return_value.__enter__.return_value = mock_doc

            metadata = extract_pdf_metadata("test_book.pdf")
            assert metadata["title"] == "Introduction to Machine Learning"
            assert "Author One" in metadata["authors"]
            assert "Author Two" in metadata["authors"]

    def test_handle_missing_metadata(self, create_test_pdf):
        """Test handling of PDFs with missing metadata."""

        with patch("fitz.open") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.metadata = {}  # No metadata
            mock_fitz.return_value.__enter__.return_value = mock_doc

            metadata = extract_pdf_metadata("no_metadata.pdf")
            assert metadata["title"] is None
            assert metadata["authors"] == []

    def test_parse_multiple_author_formats(self):
        """Test parsing various author separator formats."""

        # Test semicolon separated
        with patch("fitz.open") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.metadata = {"author": "Author One; Author Two; Author Three"}
            mock_fitz.return_value.__enter__.return_value = mock_doc
            metadata = extract_pdf_metadata("test.pdf")
            assert len(metadata["authors"]) == 3

        # Test comma separated
        with patch("fitz.open") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.metadata = {"author": "Author One, Author Two, Author Three"}
            mock_fitz.return_value.__enter__.return_value = mock_doc
            metadata = extract_pdf_metadata("test.pdf")
            assert len(metadata["authors"]) == 3

        # Test "and" separated
        with patch("fitz.open") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.metadata = {"author": "Author One and Author Two and Author Three"}
            mock_fitz.return_value.__enter__.return_value = mock_doc
            metadata = extract_pdf_metadata("test.pdf")
            assert len(metadata["authors"]) == 3


@pytest.mark.unit
class TestGenerateBookName:
    """Tests for generate_book_name function (to be implemented)."""

    def test_generate_from_title_and_author(self):
        """Test generating book name from title and first author."""

        name = generate_book_name(
            "Introduction to Machine Learning", ["Smith", "Jones"]
        )
        assert name == "introduction_to_machine_learning_smith"

    def test_sanitize_special_characters(self):
        """Test that special characters are sanitized."""

        name = generate_book_name("C++ Programming (2nd Edition)", ["Stroustrup"])
        assert name == "c_programming_2nd_edition_stroustrup"

    def test_handle_long_titles(self):
        """Test truncation of very long titles."""

        name = generate_book_name(
            "A Very Long Title That Should Be Truncated To Reasonable Length",
            ["Author"],
        )
        # Should truncate but keep readable (max 60 chars)
        assert len(name) <= 60

    def test_handle_no_authors(self):
        """Test handling books with no author information."""

        name = generate_book_name("Anonymous Textbook", [])
        assert name == "anonymous_textbook"


@pytest.mark.unit
class TestDownloadPdfWithHeaders:
    """Tests for download_pdf_with_headers function (to be implemented)."""

    @patch("requests.get")
    def test_download_pdf_with_user_agent(self, mock_get, tmp_path):
        """Test PDF download with proper headers."""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"%PDF-1.4\nTest PDF content"
        mock_get.return_value = mock_response

        result = download_pdf_with_headers(
            "https://example.com/book.pdf", str(tmp_path / "book.pdf")
        )

        # Verify headers were sent
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "User-Agent" in call_args[1]["headers"]
        assert result is not None

    @patch("requests.get")
    def test_handle_download_failure(self, mock_get, tmp_path):
        """Test handling of download failures."""

        mock_get.side_effect = Exception("Network error")

        download_path = download_pdf_with_headers(
            "https://example.com/book.pdf", str(tmp_path / "book.pdf")
        )
        assert download_path is None

    @patch("requests.get")
    def test_handle_404_response(self, mock_get, tmp_path):
        """Test handling of 404 responses."""

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status = MagicMock(
            side_effect=Exception("404 Not Found")
        )
        mock_get.return_value = mock_response

        download_path = download_pdf_with_headers(
            "https://example.com/nonexistent.pdf", str(tmp_path / "book.pdf")
        )
        assert download_path is None


@pytest.mark.integration
class TestAddBookInteractive:
    """Tests for add_book_interactive function (to be implemented)."""

    @pytest.fixture
    def sources_config(self, tmp_path):
        """Create temporary sources config file."""
        config_file = tmp_path / "sources.json"
        initial = {"libraries": {}, "papers": {}, "books": {}}
        config_file.write_text(json.dumps(initial, indent=2))
        return str(config_file)

    def test_add_book_with_auto_metadata(self, tmp_path, sources_config):
        """Test adding book with automatic metadata extraction."""

        args = MagicMock()
        args.url = "https://example.com/book.pdf"
        args.category = None  # Will prompt

        # Mock PDF download and metadata extraction
        with patch(
            "tensortruth.fetch_sources.download_pdf_with_headers"
        ) as mock_download:
            with patch(
                "tensortruth.fetch_sources.extract_pdf_metadata"
            ) as mock_extract:
                mock_download.return_value = str(tmp_path / "book.pdf")
                mock_extract.return_value = {
                    "title": "Machine Learning Basics",
                    "authors": ["Smith", "Jones"],
                }

                # Mock user inputs: accept title, accept authors,
                # accept key, category, split choice, confirm
                with patch(
                    "builtins.input",
                    side_effect=[
                        "y",  # Accept auto-detected title
                        "y",  # Accept auto-detected authors
                        "y",  # Accept generated config key
                        "ml_basics",  # Category
                        "1",  # Split method (toc)
                        "y",  # Confirm
                        "n",  # Don't fetch now
                    ],
                ):
                    result = add_book_interactive(sources_config, str(tmp_path), args)

                    assert result == 0  # Success

                    # Verify book was added
                    config = json.loads(open(sources_config).read())
                    # Note: generate_book_name uses first author's last name ("Smith")
                    assert "machine_learning_basics_smith" in config["books"]
                    book = config["books"]["machine_learning_basics_smith"]
                    assert book["title"] == "Machine Learning Basics"
                    assert book["authors"] == ["Smith", "Jones"]
                    assert book["split_method"] == "toc"
                    # Schema validation: books use "source" not "url"
                    assert "source" in book
                    assert "url" not in book
                    assert book["source"] == "https://example.com/book.pdf"

    def test_manual_metadata_entry(self, tmp_path, sources_config):
        """Test manual metadata entry when auto-extraction fails."""

        args = MagicMock()
        args.url = "https://example.com/book.pdf"
        args.category = None

        with patch(
            "tensortruth.fetch_sources.download_pdf_with_headers"
        ) as mock_download:
            with patch(
                "tensortruth.fetch_sources.extract_pdf_metadata"
            ) as mock_extract:
                mock_download.return_value = str(tmp_path / "book.pdf")
                mock_extract.return_value = {
                    "title": None,
                    "authors": [],
                }

                # Mock inputs: manual title, manual authors,
                # accept key, category, split method, confirm
                with patch(
                    "builtins.input",
                    side_effect=[
                        "Manual Book Title",  # Title (prompted since None)
                        "Author One, Author Two",  # Authors (prompted since empty)
                        "y",  # Accept generated config key
                        "category",  # Category
                        "2",  # Split method (none)
                        "y",  # Confirm
                        "n",  # Don't fetch now
                    ],
                ):
                    result = add_book_interactive(sources_config, str(tmp_path), args)

                    assert result == 0

                    config = json.loads(open(sources_config).read())
                    # Note: generate_book_name uses first author's last name
                    # For "Author One", last name is "One"
                    book_key = "manual_book_title_one"
                    assert book_key in config["books"]
                    book = config["books"][book_key]
                    assert book["title"] == "Manual Book Title"
                    assert book["authors"] == [
                        "Author One",
                        "Author Two",
                    ]
                    assert book["split_method"] == "none"
                    # Schema validation: books use "source" not "url"
                    assert "source" in book
                    assert "url" not in book

    def test_skip_url_prompt_with_cli_arg(self, tmp_path, sources_config):
        """Test that --url CLI arg skips URL prompt."""

        args = MagicMock()
        args.url = "https://example.com/book.pdf"
        args.category = None

        with patch(
            "tensortruth.fetch_sources.download_pdf_with_headers"
        ) as mock_download:
            with patch(
                "tensortruth.fetch_sources.extract_pdf_metadata"
            ) as mock_extract:
                with patch("tensortruth.fetch_sources.prompt_for_url") as mock_prompt:
                    mock_download.return_value = str(tmp_path / "book.pdf")
                    mock_extract.return_value = {
                        "title": "Test Book",
                        "authors": ["Author"],
                    }

                    with patch(
                        "builtins.input",
                        side_effect=[
                            "y",  # Accept title
                            "y",  # Accept authors
                            "y",  # Accept key
                            "category",  # Category
                            "2",  # Split method (none)
                            "y",  # Confirm
                            "n",  # Don't fetch now
                        ],
                    ):
                        result = add_book_interactive(
                            sources_config, str(tmp_path), args
                        )

                        assert result == 0
                        # Should not have called prompt_for_url since URL provided
                        mock_prompt.assert_not_called()

    def test_split_method_toc(self, tmp_path, sources_config):
        """Test book with TOC-based splitting."""

        args = MagicMock()
        args.url = "https://example.com/book.pdf"
        args.category = None

        with patch(
            "tensortruth.fetch_sources.download_pdf_with_headers"
        ) as mock_download:
            with patch(
                "tensortruth.fetch_sources.extract_pdf_metadata"
            ) as mock_extract:
                mock_download.return_value = str(tmp_path / "book.pdf")
                mock_extract.return_value = {
                    "title": "Test Book",
                    "authors": ["Author"],
                }

                with patch(
                    "builtins.input",
                    side_effect=[
                        "y",  # Accept title
                        "y",  # Accept authors
                        "y",  # Accept key
                        "category",  # Category
                        "1",  # TOC split
                        "y",  # Confirm
                        "n",  # Don't fetch now
                    ],
                ):
                    result = add_book_interactive(sources_config, str(tmp_path), args)

                    assert result == 0

                    config = json.loads(open(sources_config).read())
                    book_entry = config["books"]["test_book_author"]
                    assert book_entry["split_method"] == "toc"

    def test_split_method_none(self, tmp_path, sources_config):
        """Test book with no splitting."""

        args = MagicMock()
        args.url = "https://example.com/book.pdf"
        args.category = None

        with patch(
            "tensortruth.fetch_sources.download_pdf_with_headers"
        ) as mock_download:
            with patch(
                "tensortruth.fetch_sources.extract_pdf_metadata"
            ) as mock_extract:
                mock_download.return_value = str(tmp_path / "book.pdf")
                mock_extract.return_value = {
                    "title": "Test Book",
                    "authors": ["Author"],
                }

                with patch(
                    "builtins.input",
                    side_effect=[
                        "y",  # Accept title
                        "y",  # Accept authors
                        "y",  # Accept key
                        "category",  # Category
                        "2",  # No split
                        "y",  # Confirm
                        "n",  # Don't fetch now
                    ],
                ):
                    result = add_book_interactive(sources_config, str(tmp_path), args)

                    assert result == 0

                    config = json.loads(open(sources_config).read())
                    book_entry = config["books"]["test_book_author"]
                    assert book_entry["split_method"] == "none"

    def test_split_method_manual(self, tmp_path, sources_config):
        """Test that manual split method is blocked in interactive
        mode and requires re-selection."""

        args = MagicMock()
        args.url = "https://example.com/book.pdf"
        args.category = None

        with patch(
            "tensortruth.fetch_sources.download_pdf_with_headers"
        ) as mock_download:
            with patch(
                "tensortruth.fetch_sources.extract_pdf_metadata"
            ) as mock_extract:
                mock_download.return_value = str(tmp_path / "book.pdf")
                mock_extract.return_value = {
                    "title": "Test Book",
                    "authors": ["Author"],
                }

                # User selects manual (3), gets blocked, then selects toc (1)
                with patch(
                    "builtins.input",
                    side_effect=[
                        "y",  # Accept title
                        "y",  # Accept authors
                        "y",  # Accept key
                        "category",  # Category
                        "3",  # Manual split (blocked)
                        "1",  # Re-select: TOC split
                        "y",  # Confirm
                        "n",  # Don't fetch now
                    ],
                ):
                    result = add_book_interactive(sources_config, str(tmp_path), args)

                    assert result == 0

                    config = json.loads(open(sources_config).read())
                    book_entry = config["books"]["test_book_author"]
                    # Should have toc, not manual, since manual was blocked
                    assert book_entry["split_method"] == "toc"

    def test_invalid_url_rejected(self, tmp_path, sources_config):
        """Test that invalid PDF URLs are rejected."""

        args = MagicMock()
        args.url = None
        args.category = None

        with patch("tensortruth.fetch_sources.validate_url") as mock_validate:
            mock_validate.return_value = False

            # User enters invalid URL, then cancels
            with patch(
                "builtins.input",
                side_effect=[
                    "not-a-url",  # Invalid URL
                    "",  # Cancel (empty input)
                ],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    add_book_interactive(sources_config, str(tmp_path), args)

                # prompt_for_url raises SystemExit(1) on cancel
                assert exc_info.value.code == 1

    def test_duplicate_book_rejected(self, tmp_path, sources_config):
        """Test that duplicate book names are rejected."""

        # Add existing book
        config = json.loads(open(sources_config).read())
        config["books"]["existing_book"] = {
            "type": "pdf_book",
            "title": "Existing",
        }
        open(sources_config, "w").write(json.dumps(config, indent=2))

        args = MagicMock()
        args.url = "https://example.com/book.pdf"
        args.category = None

        with patch(
            "tensortruth.fetch_sources.download_pdf_with_headers"
        ) as mock_download:
            with patch(
                "tensortruth.fetch_sources.extract_pdf_metadata"
            ) as mock_extract:
                mock_download.return_value = str(tmp_path / "book.pdf")
                mock_extract.return_value = {
                    "title": "Existing Book",  # Will generate "existing_book" key
                    "authors": ["Author"],
                }

                # User accepts metadata, but duplicate detected, chooses not to overwrite
                with patch(
                    "builtins.input",
                    side_effect=[
                        "y",  # Accept title
                        "y",  # Accept authors
                        "existing_book",  # Use this as config key (duplicate!)
                        "n",  # Don't overwrite
                    ],
                ):
                    result = add_book_interactive(sources_config, str(tmp_path), args)

                    # Should return error code 1
                    assert result == 1

                    # Config should not be modified
                    config = json.loads(open(sources_config).read())
                    assert config["books"]["existing_book"]["title"] == "Existing"

    def test_user_cancels_at_confirmation(self, tmp_path, sources_config):
        """Test that user can cancel at confirmation step."""

        args = MagicMock()
        args.url = "https://example.com/book.pdf"
        args.category = None

        with patch(
            "tensortruth.fetch_sources.download_pdf_with_headers"
        ) as mock_download:
            with patch(
                "tensortruth.fetch_sources.extract_pdf_metadata"
            ) as mock_extract:
                mock_download.return_value = str(tmp_path / "book.pdf")
                mock_extract.return_value = {
                    "title": "Test Book",
                    "authors": ["Author"],
                }

                # Cancel at confirmation
                with patch(
                    "builtins.input",
                    side_effect=[
                        "y",  # Accept title
                        "y",  # Accept authors
                        "y",  # Accept key
                        "category",  # Category
                        "2",  # Split method (none)
                        "n",  # Cancel at confirmation
                    ],
                ):
                    result = add_book_interactive(sources_config, str(tmp_path), args)
                    assert result == 1

                    # Config should not be modified
                    config = json.loads(open(sources_config).read())
                    assert len(config["books"]) == 0
