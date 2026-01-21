"""
Unit tests for tensortruth.app_utils.helpers module.
"""

import tarfile
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.app_utils.helpers import (
    HF_FILENAME,
    HF_REPO_ID,
    _clear_retriever_cache,
    download_and_extract_indexes,
    free_memory,
)


@pytest.mark.unit
class TestDownloadAndExtractIndexes:
    """Tests for download_and_extract_indexes function with HuggingFace Hub."""

    def test_download_and_extract_success(self, tmp_path):
        """Test successful download, extraction, and cleanup."""
        # Create a temporary tarball to simulate download
        tarball_path = tmp_path / HF_FILENAME

        # Create realistic index structure: indexes/module_name/file
        test_indexes_dir = tmp_path / "test_indexes" / "indexes"
        test_module_dir = test_indexes_dir / "library_pytorch"
        test_module_dir.mkdir(parents=True)

        # Create test index content
        test_file = test_module_dir / "test.txt"
        test_file.write_text("test content")

        # Create tarball with indexes/library_pytorch/test.txt
        with tarfile.open(tarball_path, "w:") as tar:
            tar.add(test_indexes_dir, arcname="indexes")

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
            )

            # Verify extraction succeeded
            assert result is True

            # Verify files were extracted to versioned path: indexes/{model_id}/module/
            # Default embedding model is bge-m3
            extracted_file = tmp_path / "indexes" / "bge-m3" / "library_pytorch" / "test.txt"
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


@pytest.mark.unit
class TestFreeMemory:
    """Tests for free_memory function and cache clearing."""

    def test_free_memory_clears_retriever_cache(self):
        """Test that free_memory calls clear_cache on the engine's retriever."""
        mock_retriever = MagicMock()
        mock_engine = MagicMock()
        mock_engine._retriever = mock_retriever

        with patch("tensortruth.app_utils.helpers.gc.collect"):
            with patch("tensortruth.app_utils.helpers.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = False

                free_memory(engine=mock_engine)

        mock_retriever.clear_cache.assert_called_once()

    def test_free_memory_handles_missing_retriever(self):
        """Test that free_memory handles engines without _retriever attribute."""
        mock_engine = MagicMock(spec=[])  # No _retriever attribute

        with patch("tensortruth.app_utils.helpers.gc.collect"):
            with patch("tensortruth.app_utils.helpers.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = False

                # Should not raise
                free_memory(engine=mock_engine)

    def test_free_memory_handles_retriever_without_clear_cache(self):
        """Test that free_memory handles retrievers without clear_cache method."""
        mock_retriever = MagicMock(spec=[])  # No clear_cache method
        mock_engine = MagicMock()
        mock_engine._retriever = mock_retriever

        with patch("tensortruth.app_utils.helpers.gc.collect"):
            with patch("tensortruth.app_utils.helpers.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = False

                # Should not raise
                free_memory(engine=mock_engine)

    def test_free_memory_clears_streamlit_session_retriever_cache(self):
        """Test that free_memory clears cache from st.session_state engine."""
        mock_retriever = MagicMock()
        mock_engine = MagicMock()
        mock_engine._retriever = mock_retriever

        mock_session_state = {"engine": mock_engine}

        with patch("tensortruth.app_utils.helpers.gc.collect"):
            with patch("tensortruth.app_utils.helpers.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = False

                with patch.dict(
                    "sys.modules",
                    {"streamlit": MagicMock(session_state=mock_session_state)},
                ):
                    free_memory(engine=None)

        mock_retriever.clear_cache.assert_called_once()

    def test_free_memory_handles_cache_clear_exception(self):
        """Test that free_memory continues cleanup even if clear_cache raises."""
        mock_retriever = MagicMock()
        mock_retriever.clear_cache.side_effect = RuntimeError("Cache error")
        mock_engine = MagicMock()
        mock_engine._retriever = mock_retriever

        with patch("tensortruth.app_utils.helpers.gc.collect") as mock_gc:
            with patch("tensortruth.app_utils.helpers.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_torch.backends.mps.is_available.return_value = False

                # Should not raise despite cache error
                free_memory(engine=mock_engine)

        # gc.collect should still be called
        mock_gc.assert_called_once()
        # CUDA cache should still be cleared
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_free_memory_clears_settings_embed_model(self):
        """Test that free_memory clears LlamaIndex Settings._embed_model."""
        mock_settings = MagicMock()
        mock_settings._embed_model = "some_model"

        with patch("tensortruth.app_utils.helpers.gc.collect"):
            with patch("tensortruth.app_utils.helpers.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = False

                with patch.dict(
                    "sys.modules",
                    {"llama_index.core": MagicMock(Settings=mock_settings)},
                ):
                    free_memory(engine=None)

        assert mock_settings._embed_model is None

    def test_free_memory_calls_clear_marker_converter(self):
        """Test that free_memory calls clear_marker_converter."""
        with patch("tensortruth.app_utils.helpers.gc.collect"):
            with patch("tensortruth.app_utils.helpers.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = False

                with patch(
                    "tensortruth.utils.pdf.clear_marker_converter"
                ) as mock_clear:
                    free_memory(engine=None)

        mock_clear.assert_called_once()

    def test_free_memory_calls_cuda_synchronize_before_empty_cache(self):
        """Test that free_memory calls torch.cuda.synchronize() before empty_cache()."""
        call_order = []

        with patch("tensortruth.app_utils.helpers.gc.collect"):
            with patch("tensortruth.app_utils.helpers.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_torch.backends.mps.is_available.return_value = False
                mock_torch.cuda.synchronize.side_effect = lambda: call_order.append(
                    "synchronize"
                )
                mock_torch.cuda.empty_cache.side_effect = lambda: call_order.append(
                    "empty_cache"
                )

                free_memory(engine=None)

        assert call_order == ["synchronize", "empty_cache"]

    def test_free_memory_resets_loaded_config(self):
        """Test that free_memory resets loaded_config in session_state."""
        import streamlit as st

        # Set up real session_state with loaded_config
        if not hasattr(st, "session_state"):
            pytest.skip("Streamlit session_state not available in test mode")

        original_config = st.session_state.get("loaded_config")

        try:
            st.session_state["loaded_config"] = ("module", "hash")

            with patch("tensortruth.app_utils.helpers.gc.collect"):
                with patch("tensortruth.app_utils.helpers.torch") as mock_torch:
                    mock_torch.cuda.is_available.return_value = False
                    mock_torch.backends.mps.is_available.return_value = False

                    free_memory(engine=None)

            assert st.session_state.get("loaded_config") is None
        finally:
            # Restore original state
            if original_config is not None:
                st.session_state["loaded_config"] = original_config
            elif "loaded_config" in st.session_state:
                del st.session_state["loaded_config"]


@pytest.mark.unit
class TestClearRetrieverCache:
    """Tests for _clear_retriever_cache helper function."""

    def test_clear_retriever_cache_with_valid_engine(self):
        """Test _clear_retriever_cache calls clear_cache on retriever."""
        mock_retriever = MagicMock()
        mock_engine = MagicMock()
        mock_engine._retriever = mock_retriever

        _clear_retriever_cache(mock_engine)

        mock_retriever.clear_cache.assert_called_once()

    def test_clear_retriever_cache_with_none_engine(self):
        """Test _clear_retriever_cache handles None engine."""
        # Should not raise
        _clear_retriever_cache(None)

    def test_clear_retriever_cache_suppresses_exceptions(self):
        """Test _clear_retriever_cache suppresses exceptions."""
        mock_retriever = MagicMock()
        mock_retriever.clear_cache.side_effect = Exception("Test error")
        mock_engine = MagicMock()
        mock_engine._retriever = mock_retriever

        # Should not raise
        _clear_retriever_cache(mock_engine)
