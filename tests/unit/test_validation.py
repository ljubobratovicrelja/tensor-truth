"""
Unit tests for validation utilities.

Tests the critical validation logic that prevents build errors by ensuring
modules exist on disk before building indexes.
"""

import pytest

from tensortruth.utils.validation import validate_module_for_build


@pytest.mark.unit
class TestValidateModuleForBuild:
    """Tests for validate_module_for_build function."""

    def test_module_not_found_raises_error(self, temp_library_dir):
        """Test validation fails when module directory doesn't exist."""
        sources_config = {"libraries": {}, "papers": {}}

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
        module_dir = temp_library_dir / "empty_module"
        module_dir.mkdir()

        sources_config = {"libraries": {}, "papers": {}}

        with pytest.raises(ValueError) as exc_info:
            validate_module_for_build(
                "empty_module", str(temp_library_dir), sources_config
            )

        error_msg = str(exc_info.value)
        assert "directory is empty" in error_msg

    def test_module_not_in_config_warns_but_passes(self, temp_library_dir, caplog):
        """Test validation warns but doesn't fail when module not in config."""
        import logging

        # Create module directory with docs
        module_dir = temp_library_dir / "undocumented_module"
        module_dir.mkdir()
        (module_dir / "doc.md").write_text("# Documentation")

        # Empty config (module not listed)
        sources_config = {"libraries": {}, "papers": {}}

        # Should not raise, but should log warning
        with caplog.at_level(logging.WARNING):
            validate_module_for_build(
                "undocumented_module", str(temp_library_dir), sources_config
            )

        # Check warning was logged
        assert any(
            "not found in sources config" in record.message for record in caplog.records
        )
        assert any(
            "Metadata may be incomplete" in record.message for record in caplog.records
        )
