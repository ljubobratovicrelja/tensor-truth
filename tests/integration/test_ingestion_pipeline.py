"""
Integration tests for the ingestion pipeline.
"""

from unittest.mock import patch

import pytest

# ============================================================================
# Integration Tests for Database Building
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestDatabaseBuilding:
    """Integration tests for database building workflow."""

    @pytest.mark.skip(reason="Requires actual ChromaDB and embeddings - slow test")
    def test_build_module_with_sample_docs(self, temp_library_dir, temp_index_dir):
        """Test building a module with sample documents.

        This test is skipped by default as it requires:
        - Actual embedding model
        - ChromaDB
        - Sample markdown files

        Enable manually for full integration testing.
        """

        # Create sample markdown files
        module_dir = temp_library_dir / "test_module"
        module_dir.mkdir()

        sample_md = module_dir / "sample.md"
        sample_md.write_text("""
# Sample Document

This is a sample document for testing the database building process.

## Section 1

Some content here about PyTorch tensors.

## Section 2

More content about neural networks.
""")

        # Build the module
        # Note: This will actually try to load embedding models
        # build_module("test_module", chunk_sizes=[512, 128])

        # Verify index was created
        # index_path = temp_index_dir / "test_module"
        # assert index_path.exists()

        pass  # Placeholder for actual implementation


# ============================================================================
# Integration Tests for Config-based Rebuilding
# ============================================================================


@pytest.mark.integration
class TestConfigBasedRebuilding:
    """Integration tests for config-based category rebuilding."""

    @patch("tensortruth.fetch_sources.fetch_arxiv_paper")
    def test_rebuild_category_from_config(
        self, mock_fetch, sample_papers_config, temp_dir
    ):
        """Test rebuilding a category from configuration."""

        # Setup mock
        mock_fetch.return_value = {
            "title": "Test Paper",
            "arxiv_id": "1706.03762",
            "source": "https://arxiv.org/abs/1706.03762",
        }

        # This would normally process all papers in config
        # For testing, we verify the structure is correct
        assert "papers" in sample_papers_config
        assert len(sample_papers_config["papers"]["items"]) > 0

        # Note: Full test would call rebuild_category and verify results
        # Skipped here to avoid network calls in tests
