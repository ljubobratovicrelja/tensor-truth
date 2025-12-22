"""
Unit tests for tensortruth.app_utils.helpers module.
"""

import tarfile
from unittest.mock import patch

import pytest

from tensortruth.app_utils.helpers import (
    HF_FILENAME,
    HF_REPO_ID,
    download_and_extract_indexes,
)


@pytest.mark.unit
class TestDownloadAndExtractIndexes:
    """Tests for download_and_extract_indexes function with HuggingFace Hub."""

    def test_download_and_extract_success(self, tmp_path):
        """Test successful download, extraction, and cleanup."""
        # Create a temporary tarball to simulate download
        tarball_path = tmp_path / HF_FILENAME
        test_index_dir = tmp_path / "test_indexes"
        test_index_dir.mkdir()

        # Create test index content
        test_file = test_index_dir / "test.txt"
        test_file.write_text("test content")

        # Create tarball
        with tarfile.open(tarball_path, "w:") as tar:
            tar.add(test_index_dir, arcname="indexes")

        # Create HF cache directory to verify cleanup
        hf_cache_dir = tmp_path / ".cache"
        hf_cache_dir.mkdir()
        (hf_cache_dir / "cache.json").write_text("{}")

        # Mock hf_hub_download to return the tarball path
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            mock_download.return_value = str(tarball_path)

            # Execute download and extract
            result = download_and_extract_indexes(tmp_path)

            # Verify download was called with correct parameters
            mock_download.assert_called_once_with(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                repo_type="dataset",
                local_dir=tmp_path,
                local_dir_use_symlinks=False,
            )

            # Verify extraction succeeded
            assert result is True

            # Verify files were extracted
            extracted_file = tmp_path / "indexes" / "test.txt"
            assert extracted_file.exists()
            assert extracted_file.read_text() == "test content"

            # Verify cleanup happened
            assert not tarball_path.exists()
            assert not hf_cache_dir.exists()

    def test_cleanup_on_extraction_failure(self, tmp_path):
        """Test cleanup happens in finally block even when extraction fails."""
        # Create invalid tarball
        tarball_path = tmp_path / HF_FILENAME
        tarball_path.write_text("invalid")

        # Create cache directory
        hf_cache_dir = tmp_path / ".cache"
        hf_cache_dir.mkdir()

        with patch("huggingface_hub.hf_hub_download") as mock_download:
            mock_download.return_value = str(tarball_path)

            with pytest.raises(Exception):
                download_and_extract_indexes(tmp_path)

            # Verify cleanup happened despite failure
            assert not tarball_path.exists()
            assert not hf_cache_dir.exists()

    def test_download_failure_raises_exception(self, tmp_path):
        """Test that download failure raises exception."""
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            mock_download.side_effect = Exception("Network error")

            with pytest.raises(Exception) as exc_info:
                download_and_extract_indexes(tmp_path)

            assert "Network error" in str(exc_info.value)
