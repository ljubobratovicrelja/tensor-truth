"""
Unit tests for sources configuration management.

Tests the load/update/list operations that manage the user's sources.json file,
which is critical for tracking what libraries and papers are available.
"""

import json

import pytest

from tensortruth.utils.sources_config import (
    list_sources,
    load_user_sources,
    update_sources_config,
)


@pytest.mark.unit
class TestLoadUserSources:
    """Tests for load_user_sources function."""

    def test_load_invalid_json_returns_empty_config(self, tmp_path):
        """Test loading malformed JSON raises JSONDecodeError."""
        config_file = tmp_path / "bad.json"
        config_file.write_text("{ invalid json content")

        with pytest.raises(json.JSONDecodeError):
            load_user_sources(str(config_file))


@pytest.mark.unit
class TestUpdateSourcesConfig:
    """Tests for update_sources_config function."""

    def test_update_existing_library(self, tmp_path):
        """Test updating an existing library entry."""
        config_file = tmp_path / "config.json"
        initial_data = {
            "libraries": {"pytorch": {"version": "1.13", "type": "sphinx"}},
            "papers": {},
        }
        config_file.write_text(json.dumps(initial_data))

        updated_lib = {
            "version": "2.0",
            "type": "sphinx",
            "doc_root": "https://pytorch.org/docs/stable/",
        }

        update_sources_config(str(config_file), "libraries", "pytorch", updated_lib)

        loaded = json.loads(config_file.read_text())
        assert loaded["libraries"]["pytorch"]["version"] == "2.0"
        assert "doc_root" in loaded["libraries"]["pytorch"]

    def test_preserves_formatting(self, tmp_path):
        """Test that JSON is written with readable formatting."""
        config_file = tmp_path / "config.json"
        # Create initial empty config
        config_file.write_text(json.dumps({"libraries": {}, "papers": {}, "books": {}}))

        lib_config = {
            "version": "2.0",
            "type": "sphinx",
            "doc_root": "https://pytorch.org/docs/stable/",
        }

        update_sources_config(str(config_file), "libraries", "pytorch", lib_config)

        content = config_file.read_text()
        # Should have indentation (pretty-printed)
        assert "  " in content or "\t" in content
        assert "\n" in content


@pytest.mark.unit
class TestListSources:
    """Tests for list_sources function."""

    def test_handles_missing_fields_gracefully(self, capsys):
        """Test that missing optional fields don't crash the listing."""
        config = {
            "libraries": {"minimal_lib": {"type": "sphinx"}},  # Missing version
            "papers": {"minimal_papers": {"items": []}},  # Missing type and description
        }

        # Should not raise
        list_sources(config)

        captured = capsys.readouterr()
        assert "minimal_lib" in captured.out
        assert "minimal_papers" in captured.out


@pytest.mark.unit
class TestSourcesConfigIntegration:
    """Integration tests for sources config workflows."""

    def test_full_workflow_create_update_load(self, tmp_path):
        """Test complete workflow: create, update, and load config."""
        config_file = tmp_path / "workflow_test.json"
        # Create initial empty config
        config_file.write_text(json.dumps({"libraries": {}, "papers": {}, "books": {}}))

        # Step 1: Create initial config with one library
        lib1 = {"version": "1.0", "type": "sphinx"}
        update_sources_config(str(config_file), "libraries", "lib1", lib1)

        # Step 2: Add another library
        lib2 = {"version": "2.0", "type": "doxygen"}
        update_sources_config(str(config_file), "libraries", "lib2", lib2)

        # Step 3: Add a paper category
        papers = {
            "type": "arxiv",
            "items": [{"arxiv_id": "1234.5678"}],
        }
        update_sources_config(str(config_file), "papers", "category1", papers)

        # Step 4: Load and verify
        loaded = load_user_sources(str(config_file))

        assert len(loaded["libraries"]) == 2
        assert "lib1" in loaded["libraries"]
        assert "lib2" in loaded["libraries"]
        assert "category1" in loaded["papers"]
        assert loaded["libraries"]["lib1"]["version"] == "1.0"
        assert loaded["libraries"]["lib2"]["type"] == "doxygen"
