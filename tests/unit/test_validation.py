"""
Unit tests for validation utilities.

Tests the critical validation logic that prevents build errors by ensuring
modules exist on disk before building indexes.
"""

import json

import pytest

from tensortruth.utils.validation import validate_module_for_build, validate_sources


@pytest.mark.unit
class TestValidateModuleForBuild:
    """Tests for validate_module_for_build function."""

    def test_module_not_found_raises_error(self, temp_library_dir):
        """Test validation fails when module directory doesn't exist."""
        sources_config = {
            "libraries": {"nonexistent_module": {"type": "sphinx", "doc_root": "docs"}},
            "papers": {},
            "books": {},
        }

        with pytest.raises(ValueError) as exc_info:
            validate_module_for_build(
                "nonexistent_module", str(temp_library_dir), sources_config
            )

        error_msg = str(exc_info.value)
        assert "not found" in error_msg
        assert "tensor-truth-docs nonexistent_module" in error_msg

    def test_empty_module_directory_raises_error(self, temp_library_dir):
        """Test validation fails when module directory is empty."""
        # Create empty directory
        module_dir = temp_library_dir / "library_empty_module"
        module_dir.mkdir()

        sources_config = {
            "libraries": {"empty_module": {"type": "sphinx", "doc_root": "docs"}},
            "papers": {},
            "books": {},
        }

        with pytest.raises(ValueError) as exc_info:
            validate_module_for_build(
                "empty_module", str(temp_library_dir), sources_config
            )

        error_msg = str(exc_info.value)
        assert "directory is empty" in error_msg

    def test_module_not_in_config_warns(self, temp_library_dir, caplog):
        """Test validation warns when module not in config but directory exists."""
        # Create module directory with docs
        module_dir = temp_library_dir / "library_test_module"
        module_dir.mkdir()
        (module_dir / "doc.md").write_text("# Documentation")

        # Config with module listed (to pass get_document_type check)
        sources_config = {
            "libraries": {"test_module": {"type": "sphinx", "doc_root": "docs"}},
            "papers": {},
            "books": {},
        }

        # Should not raise, but should warn
        validate_module_for_build("test_module", str(temp_library_dir), sources_config)


@pytest.mark.unit
class TestValidateSources:
    """Tests for validate_sources function - focused on critical functionality."""

    # =========================================================================
    # Exit Code Tests (Critical)
    # =========================================================================

    def test_returns_1_on_schema_errors(self, tmp_path, caplog):
        """Test returns exit code 1 when config has schema errors."""
        config_file = tmp_path / "sources.json"
        library_docs_dir = tmp_path / "library_docs"
        library_docs_dir.mkdir()

        # Invalid config: library missing 'type' field
        config = {
            "libraries": {"pytorch": {"doc_root": "https://pytorch.org/docs/stable/"}},
            "papers": {},
            "books": {},
        }

        config_file.write_text(json.dumps(config))

        exit_code = validate_sources(str(config_file), str(library_docs_dir))

        assert exit_code == 1
        assert "Missing 'type' field" in caplog.text

    def test_returns_0_on_missing_docs(self, tmp_path, caplog):
        """Test returns 0 when config valid but docs missing (incomplete state)."""
        config_file = tmp_path / "sources.json"
        library_docs_dir = tmp_path / "library_docs"
        library_docs_dir.mkdir()

        # Valid config but no docs fetched
        config = {
            "libraries": {
                "pytorch": {
                    "type": "sphinx",
                    "doc_root": "https://pytorch.org/docs/stable/",
                    "version": "2.9",
                }
            },
            "papers": {},
            "books": {},
        }

        config_file.write_text(json.dumps(config))

        exit_code = validate_sources(str(config_file), str(library_docs_dir))

        assert exit_code == 0  # Incomplete but not an error
        assert "VALIDATION INCOMPLETE" in caplog.text

    def test_returns_0_on_full_success(self, tmp_path, caplog):
        """Test returns 0 when everything is valid and complete."""
        import logging

        caplog.set_level(logging.INFO)

        config_file = tmp_path / "sources.json"
        library_docs_dir = tmp_path / "library_docs"
        library_docs_dir.mkdir()

        # Create library directory
        (library_docs_dir / "library_pytorch").mkdir()
        (library_docs_dir / "library_pytorch" / "index.html").write_text(
            "<html></html>"
        )

        config = {
            "libraries": {
                "pytorch": {
                    "type": "sphinx",
                    "doc_root": "https://pytorch.org/docs/stable/",
                    "version": "2.9",
                }
            },
            "papers": {},
            "books": {},
        }

        config_file.write_text(json.dumps(config))

        exit_code = validate_sources(str(config_file), str(library_docs_dir))

        assert exit_code == 0
        assert "VALIDATION PASSED" in caplog.text

    # =========================================================================
    # Deprecated Field Detection (Critical - prevents regressions)
    # =========================================================================

    def test_detects_deprecated_url_in_papers(self, tmp_path, caplog):
        """Test detects deprecated 'url' field in papers (should be 'source')."""
        config_file = tmp_path / "sources.json"
        library_docs_dir = tmp_path / "library_docs"
        library_docs_dir.mkdir()

        config = {
            "libraries": {},
            "papers": {
                "test_category": {
                    "type": "arxiv",
                    "items": {
                        "1706.03762": {
                            "title": "Attention Is All You Need",
                            "arxiv_id": "1706.03762",
                            "url": "https://arxiv.org/abs/1706.03762",  # DEPRECATED
                            "authors": "Vaswani et al.",
                            "year": "2017",
                        }
                    },
                }
            },
            "books": {},
        }

        config_file.write_text(json.dumps(config))

        exit_code = validate_sources(str(config_file), str(library_docs_dir))

        assert exit_code == 1
        assert "deprecated 'url' field" in caplog.text
        assert "should be 'source'" in caplog.text

    def test_detects_deprecated_url_in_books(self, tmp_path, caplog):
        """Test detects deprecated 'url' field in books (should be 'source')."""
        config_file = tmp_path / "sources.json"
        library_docs_dir = tmp_path / "library_docs"
        library_docs_dir.mkdir()

        config = {
            "libraries": {},
            "papers": {},
            "books": {
                "linear_algebra": {
                    "type": "pdf_book",
                    "title": "Linear Algebra",
                    "authors": ["Author Name"],
                    "url": "https://example.com/book.pdf",  # DEPRECATED
                    "category": "Math",
                    "split_method": "toc",
                }
            },
        }

        config_file.write_text(json.dumps(config))

        exit_code = validate_sources(str(config_file), str(library_docs_dir))

        assert exit_code == 1
        assert "deprecated 'url' field" in caplog.text

    # =========================================================================
    # Directory Naming Validation (Critical - catches common bugs)
    # =========================================================================

    def test_validates_library_directory_naming(self, tmp_path, caplog):
        """Test validates correct library directory naming: library_{name}."""
        import logging

        caplog.set_level(logging.INFO)

        config_file = tmp_path / "sources.json"
        library_docs_dir = tmp_path / "library_docs"
        library_docs_dir.mkdir()

        # Create correct directory structure
        (library_docs_dir / "library_pytorch").mkdir()
        (library_docs_dir / "library_pytorch" / "index.html").write_text(
            "<html></html>"
        )

        config = {
            "libraries": {
                "pytorch": {
                    "type": "sphinx",
                    "doc_root": "https://pytorch.org/docs/stable/",
                    "version": "2.9",
                }
            },
            "papers": {},
            "books": {},
        }

        config_file.write_text(json.dumps(config))

        exit_code = validate_sources(str(config_file), str(library_docs_dir))

        assert exit_code == 0
        assert "pytorch" in caplog.text

    def test_detects_orphaned_directories(self, tmp_path, caplog):
        """Test detects directories not in config (orphaned)."""
        config_file = tmp_path / "sources.json"
        library_docs_dir = tmp_path / "library_docs"
        library_docs_dir.mkdir()

        # Create orphaned directory
        (library_docs_dir / "unknown_library").mkdir()

        config = {"libraries": {}, "papers": {}, "books": {}}

        config_file.write_text(json.dumps(config))

        exit_code = validate_sources(str(config_file), str(library_docs_dir))

        assert exit_code == 0  # Orphans are warnings, not errors
        assert "Orphaned" in caplog.text
        assert "unknown_library" in caplog.text

    # =========================================================================
    # Schema Validation (Critical - prevents invalid configs)
    # =========================================================================

    def test_detects_invalid_library_type(self, tmp_path, caplog):
        """Test rejects invalid library type (not sphinx/doxygen)."""
        config_file = tmp_path / "sources.json"
        library_docs_dir = tmp_path / "library_docs"
        library_docs_dir.mkdir()

        config = {
            "libraries": {
                "custom_lib": {
                    "type": "javadoc",  # INVALID
                    "doc_root": "https://example.com/docs/",
                }
            },
            "papers": {},
            "books": {},
        }

        config_file.write_text(json.dumps(config))

        exit_code = validate_sources(str(config_file), str(library_docs_dir))

        assert exit_code == 1
        assert "Invalid type 'javadoc'" in caplog.text
        assert "expected: sphinx or doxygen" in caplog.text

    def test_detects_invalid_split_method(self, tmp_path, caplog):
        """Test rejects invalid book split_method (not toc/none/manual)."""
        config_file = tmp_path / "sources.json"
        library_docs_dir = tmp_path / "library_docs"
        library_docs_dir.mkdir()

        config = {
            "libraries": {},
            "papers": {},
            "books": {
                "test_book": {
                    "type": "pdf_book",
                    "title": "Test Book",
                    "authors": ["Author"],
                    "source": "https://example.com/book.pdf",
                    "category": "Test",
                    "split_method": "auto",  # INVALID
                }
            },
        }

        config_file.write_text(json.dumps(config))

        exit_code = validate_sources(str(config_file), str(library_docs_dir))

        assert exit_code == 1
        assert "Invalid split_method 'auto'" in caplog.text
        assert "expected: toc, none, or manual" in caplog.text

    def test_detects_missing_required_paper_fields(self, tmp_path, caplog):
        """Test detects when paper is missing required fields."""
        config_file = tmp_path / "sources.json"
        library_docs_dir = tmp_path / "library_docs"
        library_docs_dir.mkdir()

        config = {
            "libraries": {},
            "papers": {
                "test_category": {
                    "type": "arxiv",
                    "items": {
                        "1706.03762": {
                            "title": "Attention Is All You Need",
                            # Missing: arxiv_id, source, authors, year
                        }
                    },
                }
            },
            "books": {},
        }

        config_file.write_text(json.dumps(config))

        exit_code = validate_sources(str(config_file), str(library_docs_dir))

        assert exit_code == 1
        assert "Missing 'arxiv_id' field" in caplog.text
        assert "Missing 'source' field" in caplog.text
        assert "Missing 'authors' field" in caplog.text
        assert "Missing 'year' field" in caplog.text
