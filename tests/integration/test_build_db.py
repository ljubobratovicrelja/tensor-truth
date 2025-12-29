"""
Integration tests for build_db.py functionality.

Tests the vector database building process, module validation,
and metadata extraction workflows.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.build_db import build_module, extract_metadata
from tensortruth.build_db import main as build_main
from tensortruth.utils.metadata import DocumentType


@pytest.mark.integration
@pytest.mark.slow
class TestBuildModule:
    """Tests for build_module function."""

    @pytest.fixture
    def mock_sources_config(self):
        """Mock sources configuration."""
        return {
            "libraries": {
                "test_lib": {
                    "type": "sphinx",
                    "version": "1.0",
                    "display_name": "Test Library",
                    "doc_root": "https://example.com/docs/",
                }
            },
            "papers": {
                "test_papers": {
                    "type": "arxiv",
                    "display_name": "Test Papers",
                    "description": "Test category",
                    "items": {
                        "1706.03762": {
                            "title": "Test Paper",
                            "authors": "Author",
                            "year": "2017",
                        }
                    },
                }
            },
            "books": {
                "test_book": {
                    "type": "pdf_book",
                    "title": "Test Book",
                    "authors": ["Author"],
                    "category": "test_category",
                }
            },
        }

    @pytest.fixture
    def library_docs_with_content(self, tmp_path):
        """Create library docs directory with sample markdown files."""
        docs_dir = tmp_path / "library_docs"
        docs_dir.mkdir()

        # Create library module
        lib_dir = docs_dir / "library_test_lib"
        lib_dir.mkdir()
        (lib_dir / "intro.md").write_text("# Introduction\n\nTest content.")
        (lib_dir / "api.md").write_text("# API Reference\n\nAPI docs.")

        # Create papers module
        papers_dir = docs_dir / "papers_test_papers"
        papers_dir.mkdir()
        (papers_dir / "1706.03762.md").write_text("# Test Paper\n\nPaper content.")

        # Create book module (note: singular "book" from DocumentType.BOOK.value)
        book_dir = docs_dir / "book_test_book"
        book_dir.mkdir()
        (book_dir / "chapter1.md").write_text("# Chapter 1\n\nBook content.")

        return str(docs_dir)

    @patch("tensortruth.build_db.get_embed_model")
    @patch("tensortruth.build_db.VectorStoreIndex")
    def test_build_library_module(
        self,
        mock_index,
        mock_embed,
        tmp_path,
        library_docs_with_content,
        mock_sources_config,
    ):
        """Test building a library module."""

        indexes_dir = str(tmp_path / "indexes")

        # Mock embedding model
        mock_embed.return_value = MagicMock()

        build_module(
            "test_lib",
            library_docs_with_content,
            indexes_dir,
            mock_sources_config,
        )

        # Verify index was created
        assert Path(indexes_dir, "library_test_lib").exists()

        # Verify VectorStoreIndex was called
        mock_index.assert_called_once()

    @patch("tensortruth.build_db.get_embed_model")
    @patch("tensortruth.build_db.VectorStoreIndex")
    def test_build_papers_module(
        self,
        mock_index,
        mock_embed,
        tmp_path,
        library_docs_with_content,
        mock_sources_config,
    ):
        """Test building a papers module."""

        indexes_dir = str(tmp_path / "indexes")
        mock_embed.return_value = MagicMock()

        build_module(
            "test_papers",
            library_docs_with_content,
            indexes_dir,
            mock_sources_config,
        )

        # Verify papers index was created
        assert Path(indexes_dir, "papers_test_papers").exists()

    @patch("tensortruth.build_db.get_embed_model")
    @patch("tensortruth.build_db.VectorStoreIndex")
    def test_build_book_module(
        self,
        mock_index,
        mock_embed,
        tmp_path,
        library_docs_with_content,
        mock_sources_config,
    ):
        """Test building a book module."""

        indexes_dir = str(tmp_path / "indexes")
        mock_embed.return_value = MagicMock()

        build_module(
            "test_book",
            library_docs_with_content,
            indexes_dir,
            mock_sources_config,
        )

        # Verify book index was created (singular "book")
        assert Path(indexes_dir, "book_test_book").exists()

    def test_missing_source_directory(self, tmp_path, mock_sources_config, caplog):
        """Test that missing source directory is handled gracefully."""

        library_docs_dir = str(tmp_path / "nonexistent_docs")
        indexes_dir = str(tmp_path / "indexes")

        build_module(
            "test_lib",
            library_docs_dir,
            indexes_dir,
            mock_sources_config,
        )

        # Should log error, not raise
        assert "missing" in caplog.text.lower()

    @patch("tensortruth.build_db.get_embed_model")
    @patch("tensortruth.build_db.VectorStoreIndex")
    def test_rebuild_replaces_old_index(
        self,
        mock_index,
        mock_embed,
        tmp_path,
        library_docs_with_content,
        mock_sources_config,
    ):
        """Test that rebuilding removes old index first."""

        indexes_dir = str(tmp_path / "indexes")
        index_path = Path(indexes_dir, "library_test_lib")

        # Create existing index
        index_path.mkdir(parents=True)
        old_file = index_path / "old_data.txt"
        old_file.write_text("old index data")

        mock_embed.return_value = MagicMock()

        build_module(
            "test_lib",
            library_docs_with_content,
            indexes_dir,
            mock_sources_config,
        )

        # Old file should be gone
        assert not old_file.exists()

    def test_empty_directory_skipped(self, tmp_path, mock_sources_config, caplog):
        """Test that empty directories are skipped."""

        docs_dir = tmp_path / "library_docs"
        docs_dir.mkdir()

        # Create empty module directory
        empty_dir = docs_dir / "library_test_lib"
        empty_dir.mkdir()

        indexes_dir = str(tmp_path / "indexes")

        build_module(
            "test_lib",
            str(docs_dir),
            indexes_dir,
            mock_sources_config,
        )

        # Should skip without creating index
        assert "no files" in caplog.text.lower() or "skipping" in caplog.text.lower()


@pytest.mark.integration
class TestExtractMetadata:
    """Tests for metadata extraction function."""

    @pytest.fixture
    def mock_document(self):
        """Create mock document."""
        doc = MagicMock()
        doc.metadata = {"file_path": "/path/to/doc.md"}
        return doc

    @pytest.fixture
    def sources_config(self):
        """Sample sources config."""
        return {
            "libraries": {
                "pytorch": {
                    "display_name": "PyTorch",
                    "version": "2.0",
                }
            },
            "papers": {
                "dl_foundations": {
                    "display_name": "DL Foundations",
                    "items": {
                        "1706.03762": {
                            "title": "Attention Is All You Need",
                            "authors": "Vaswani et al.",
                        }
                    },
                }
            },
            "books": {
                "linear_algebra": {
                    "title": "Linear Algebra",
                    "authors": ["Author"],
                }
            },
        }

    @patch("tensortruth.build_db.extract_library_module_metadata")
    def test_library_metadata_extraction(
        self, mock_extract, mock_document, sources_config
    ):
        """Test metadata extraction for library documents."""

        mock_extract.return_value = {
            "title": "PyTorch Intro",
            "display_name": "PyTorch",
            "library_display_name": "PyTorch",
        }

        documents = [mock_document]

        extract_metadata(
            "pytorch",
            DocumentType.LIBRARY,
            sources_config,
            documents,
        )

        # Verify extraction was called
        mock_extract.assert_called_once()

        # Verify metadata was added to document
        assert "title" in mock_document.metadata

    @patch("tensortruth.build_db.extract_arxiv_metadata_from_config")
    def test_papers_metadata_extraction(
        self, mock_extract, mock_document, sources_config
    ):
        """Test metadata extraction for paper documents."""

        mock_extract.return_value = {
            "title": "Attention Is All You Need",
            "formatted_authors": "Vaswani et al.",
        }

        documents = [mock_document]

        extract_metadata(
            "dl_foundations",
            DocumentType.PAPERS,
            sources_config,
            documents,
        )

        mock_extract.assert_called_once()

    @patch("tensortruth.build_db.extract_book_chapter_metadata")
    @patch("tensortruth.build_db.get_book_metadata_from_config")
    def test_book_metadata_extraction(
        self, mock_get_book, mock_extract, mock_document, sources_config
    ):
        """Test metadata extraction for book documents."""

        mock_get_book.return_value = {
            "title": "Linear Algebra",
            "authors": ["Author"],
        }

        mock_extract.return_value = {
            "title": "Chapter 1",
            "book_display_name": "Linear Algebra",
        }

        documents = [mock_document]

        extract_metadata(
            "linear_algebra",
            DocumentType.BOOK,
            sources_config,
            documents,
        )

        mock_get_book.assert_called_once()
        mock_extract.assert_called_once()


@pytest.mark.integration
class TestBuildMainCLI:
    """Tests for build_db main CLI."""

    @pytest.fixture
    def setup_build_env(self, tmp_path):
        """Setup build environment with config and docs."""
        # Create sources config
        config_file = tmp_path / "sources.json"
        config = {
            "libraries": {"test_lib": {"type": "sphinx"}},
            "papers": {"test_papers": {"type": "arxiv", "items": {}}},
            "books": {},
        }
        config_file.write_text(json.dumps(config, indent=2))

        # Create library docs
        docs_dir = tmp_path / "library_docs"
        docs_dir.mkdir()

        # Create library module directory
        lib_dir = docs_dir / "library_test_lib"
        lib_dir.mkdir()
        (lib_dir / "test.md").write_text("# Test\n\nContent.")

        # Create papers module directory (needed for --all and --papers flags)
        papers_dir = docs_dir / "papers_test_papers"
        papers_dir.mkdir()
        (papers_dir / "test_paper.md").write_text("# Test Paper\n\nPaper content.")

        indexes_dir = tmp_path / "indexes"
        indexes_dir.mkdir()

        return {
            "config": str(config_file),
            "docs": str(docs_dir),
            "indexes": str(indexes_dir),
        }

    @patch("tensortruth.build_db.get_embed_model")
    @patch("tensortruth.build_db.VectorStoreIndex")
    def test_build_all_flag(self, mock_index, mock_embed, setup_build_env):
        """Test --all flag builds all modules."""

        mock_embed.return_value = MagicMock()

        with patch(
            "sys.argv",
            [
                "tensor-truth-build",
                "--all",
                "--library-docs-dir",
                setup_build_env["docs"],
                "--sources-config",
                setup_build_env["config"],
                "--indexes-dir",
                setup_build_env["indexes"],
            ],
        ):
            result = build_main()

        # Should succeed
        assert result == 0

    @patch("tensortruth.build_db.get_embed_model")
    @patch("tensortruth.build_db.VectorStoreIndex")
    def test_build_specific_modules(self, mock_index, mock_embed, setup_build_env):
        """Test building specific modules."""

        mock_embed.return_value = MagicMock()

        with patch(
            "sys.argv",
            [
                "tensor-truth-build",
                "--modules",
                "test_lib",
                "--library-docs-dir",
                setup_build_env["docs"],
                "--sources-config",
                setup_build_env["config"],
                "--indexes-dir",
                setup_build_env["indexes"],
            ],
        ):
            result = build_main()

        assert result == 0

    def test_no_modules_specified_error(self, setup_build_env):
        """Test that error is raised when no modules specified."""

        with patch(
            "sys.argv",
            [
                "tensor-truth-build",
                "--library-docs-dir",
                setup_build_env["docs"],
                "--sources-config",
                setup_build_env["config"],
            ],
        ):
            result = build_main()

        # Should return error
        assert result == 1

    def test_libraries_flag(self, setup_build_env):
        """Test --libraries flag builds only library modules."""

        with patch(
            "sys.argv",
            [
                "tensor-truth-build",
                "--libraries",
                "--library-docs-dir",
                setup_build_env["docs"],
                "--sources-config",
                setup_build_env["config"],
                "--indexes-dir",
                setup_build_env["indexes"],
            ],
        ):
            with patch("tensortruth.build_db.build_module") as mock_build:
                build_main()

        # Should call build for library modules only
        mock_build.assert_called()

    def test_papers_flag(self, setup_build_env):
        """Test --papers flag builds only paper modules."""

        with patch(
            "sys.argv",
            [
                "tensor-truth-build",
                "--papers",
                "--library-docs-dir",
                setup_build_env["docs"],
                "--sources-config",
                setup_build_env["config"],
                "--indexes-dir",
                setup_build_env["indexes"],
            ],
        ):
            with patch("tensortruth.build_db.build_module") as mock_build:
                build_main()

        mock_build.assert_called()

    def test_cannot_combine_modules_with_group_flags(self, setup_build_env):
        """Test that --modules cannot be used with --all/--libraries/etc."""

        with patch(
            "sys.argv",
            [
                "tensor-truth-build",
                "--modules",
                "test_lib",
                "--all",
                "--library-docs-dir",
                setup_build_env["docs"],
                "--sources-config",
                setup_build_env["config"],
            ],
        ):
            result = build_main()

        # Should return error
        assert result == 1
