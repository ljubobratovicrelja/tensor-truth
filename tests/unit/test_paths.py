"""
Unit tests for path resolution utilities.

Tests the priority system (CLI args > ENV vars > defaults) which is critical
for ensuring users can customize paths in multiple ways.
"""

import os

import pytest

from tensortruth.cli_paths import (
    get_base_indexes_dir,
    get_library_docs_dir,
    get_sources_config_path,
)


@pytest.mark.unit
class TestLibraryDocsDir:
    """Tests for get_library_docs_dir function with priority resolution."""

    def test_default_path_when_no_overrides(self):
        """Test default path is ~/.tensortruth/library_docs."""
        # Clear environment variable if it exists
        env_backup = os.environ.pop("TENSOR_TRUTH_DOCS_DIR", None)
        try:
            result = get_library_docs_dir()

            assert "library_docs" in str(result)
            assert ".tensortruth" in str(result)
        finally:
            if env_backup:
                os.environ["TENSOR_TRUTH_DOCS_DIR"] = env_backup

    def test_cli_override_takes_priority(self, tmp_path, monkeypatch):
        """Test CLI argument takes highest priority over env var."""
        from pathlib import Path

        cli_path = tmp_path / "cli_override"
        env_path = tmp_path / "env_override"

        # Set environment variable
        monkeypatch.setenv("TENSOR_TRUTH_DOCS_DIR", str(env_path))

        # CLI override should win
        result = get_library_docs_dir(override=str(cli_path))

        assert Path(result) == cli_path.absolute()

    def test_env_var_override_when_no_cli(self, tmp_path, monkeypatch):
        """Test environment variable is used when no CLI arg."""
        from pathlib import Path

        env_path = tmp_path / "env_override"
        monkeypatch.setenv("TENSOR_TRUTH_DOCS_DIR", str(env_path))

        result = get_library_docs_dir()

        assert Path(result) == env_path.absolute()


@pytest.mark.unit
class TestSourcesConfigPath:
    """Tests for get_sources_config_path function."""

    def test_does_not_create_file(self, tmp_path):
        """Test that the file itself is NOT auto-created (only returns path)."""
        from pathlib import Path

        new_file = tmp_path / "new_sources.json"
        assert not new_file.exists()

        result = get_sources_config_path(override=str(new_file))

        # File should not be created, only path returned
        assert not Path(result).exists()
        assert result == str(new_file.absolute())


@pytest.mark.unit
class TestBaseIndexesDir:
    """Tests for get_base_indexes_dir function."""

    def test_priority_resolution(self, tmp_path, monkeypatch):
        """Test full priority chain: CLI > ENV > default."""
        from pathlib import Path

        cli_path = tmp_path / "cli_indexes"
        env_path = tmp_path / "env_indexes"

        # Test 1: CLI wins over ENV
        monkeypatch.setenv("TENSOR_TRUTH_INDEXES_DIR", str(env_path))
        result = get_base_indexes_dir(override=str(cli_path))
        assert Path(result) == cli_path.absolute()

        # Test 2: ENV wins over default
        result = get_base_indexes_dir()
        assert Path(result) == env_path.absolute()

        # Test 3: Default when nothing set
        monkeypatch.delenv("TENSOR_TRUTH_INDEXES_DIR")
        result = get_base_indexes_dir()
        assert ".tensortruth" in result
        assert "indexes" in result


@pytest.mark.unit
class TestPathPriorityIntegration:
    """Integration tests for path priority system across all path functions."""

    def test_all_functions_respect_cli_priority(self, tmp_path):
        """Test that all path functions prioritize CLI args consistently."""
        from pathlib import Path

        cli_docs = tmp_path / "cli_docs"
        cli_config = tmp_path / "cli_config.json"
        cli_indexes = tmp_path / "cli_indexes"

        docs_result = get_library_docs_dir(override=str(cli_docs))
        config_result = get_sources_config_path(override=str(cli_config))
        indexes_result = get_base_indexes_dir(override=str(cli_indexes))

        assert Path(docs_result) == cli_docs.absolute()
        assert Path(config_result) == cli_config.absolute()
        assert Path(indexes_result) == cli_indexes.absolute()

    def test_all_functions_respect_env_priority(self, tmp_path, monkeypatch):
        """Test that all path functions respect environment variables."""
        from pathlib import Path

        env_docs = tmp_path / "env_docs"
        env_config = tmp_path / "env_config.json"
        env_indexes = tmp_path / "env_indexes"

        monkeypatch.setenv("TENSOR_TRUTH_DOCS_DIR", str(env_docs))
        monkeypatch.setenv("TENSOR_TRUTH_SOURCES_CONFIG", str(env_config))
        monkeypatch.setenv("TENSOR_TRUTH_INDEXES_DIR", str(env_indexes))

        docs_result = get_library_docs_dir()
        config_result = get_sources_config_path()
        indexes_result = get_base_indexes_dir()

        assert Path(docs_result) == env_docs.absolute()
        assert Path(config_result) == env_config.absolute()
        assert Path(indexes_result) == env_indexes.absolute()

    def test_mixed_overrides_work_independently(self, tmp_path, monkeypatch):
        """Test that each function can use different override methods."""
        from pathlib import Path

        cli_docs = tmp_path / "cli_docs"
        env_config = tmp_path / "env_config.json"

        # Set only config env var
        monkeypatch.setenv("TENSOR_TRUTH_SOURCES_CONFIG", str(env_config))

        # Use CLI for docs, ENV for config, default for indexes
        docs_result = get_library_docs_dir(override=str(cli_docs))
        config_result = get_sources_config_path()
        indexes_result = get_base_indexes_dir()

        assert Path(docs_result) == cli_docs.absolute()
        assert Path(config_result) == env_config.absolute()
        assert ".tensortruth" in indexes_result and "indexes" in indexes_result
