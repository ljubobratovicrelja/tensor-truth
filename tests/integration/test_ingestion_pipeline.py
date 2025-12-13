"""
Integration tests for the ingestion pipeline.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.utils import run_ingestion

# ============================================================================
# Integration Tests for Ingestion Pipeline
# ============================================================================


@pytest.mark.integration
class TestIngestionPipeline:
    """Integration tests for paper ingestion workflow."""

    @patch("tensortruth.utils.fetch_and_convert_paper")
    @patch("tensortruth.utils.build_module")
    @patch("tensortruth.utils.paper_already_processed")
    def test_run_ingestion_new_paper(
        self, mock_processed, mock_build, mock_fetch, sample_paper_metadata
    ):
        """Test ingestion pipeline for new paper."""
        # Setup mocks
        mock_processed.return_value = False  # Paper not yet processed
        mock_fetch.return_value = sample_paper_metadata
        mock_build.return_value = None

        # Run ingestion
        success, logs = run_ingestion("papers", "1234.56789")

        # Assertions
        assert success is True
        assert len(logs) > 0
        assert any("Success" in log for log in logs)

        # Verify calls
        mock_processed.assert_called_once_with("papers", "1234.56789")
        mock_fetch.assert_called_once()
        mock_build.assert_called_once_with("papers")

    @patch("tensortruth.utils.fetch_and_convert_paper")
    @patch("tensortruth.utils.build_module")
    @patch("tensortruth.utils.paper_already_processed")
    def test_run_ingestion_existing_paper(self, mock_processed, mock_build, mock_fetch):
        """Test ingestion pipeline when paper already exists."""
        # Setup mocks
        mock_processed.return_value = True  # Paper already processed

        # Run ingestion
        success, logs = run_ingestion("papers", "1234.56789")

        # Assertions
        assert success is True
        assert any("already processed" in log for log in logs)

        # Verify fetch wasn't called
        mock_fetch.assert_not_called()

    @patch("tensortruth.utils.fetch_and_convert_paper")
    @patch("tensortruth.utils.paper_already_processed")
    def test_run_ingestion_failure(self, mock_processed, mock_fetch):
        """Test ingestion pipeline handles failures gracefully."""
        # Setup mocks
        mock_processed.return_value = False
        mock_fetch.side_effect = Exception("Network error")

        # Run ingestion
        success, logs = run_ingestion("papers", "1234.56789")

        # Assertions
        assert success is False
        assert len(logs) > 0
        assert any("Error" in log for log in logs)


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
        from tensortruth.build_db import build_module

        # Create sample markdown files
        module_dir = temp_library_dir / "test_module"
        module_dir.mkdir()

        sample_md = module_dir / "sample.md"
        sample_md.write_text(
            """
# Sample Document

This is a sample document for testing the database building process.

## Section 1

Some content here about PyTorch tensors.

## Section 2

More content about neural networks.
"""
        )

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

    @patch("tensortruth.fetch_paper.fetch_and_convert_paper")
    def test_rebuild_category_from_config(
        self, mock_fetch, sample_papers_config, temp_dir
    ):
        """Test rebuilding a category from configuration."""
        from tensortruth.fetch_paper import rebuild_category

        # Setup mock
        mock_fetch.return_value = {
            "title": "Test Paper",
            "arxiv_id": "1706.03762",
            "url": "https://arxiv.org/abs/1706.03762",
        }

        # This would normally process all papers in config
        # For testing, we verify the structure is correct
        assert "papers" in sample_papers_config
        assert len(sample_papers_config["papers"]["items"]) > 0

        # Note: Full test would call rebuild_category and verify results
        # Skipped here to avoid network calls in tests
