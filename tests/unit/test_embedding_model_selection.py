"""
Unit tests for embedding model selection MVP.

Tests the new functionality for:
- Index metadata system
- ModelManager singleton
- Path helpers for embedding model directories
- Migration utility
"""

import json
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# Test Index Metadata System
# ============================================================================


@pytest.mark.unit
class TestSanitizeModelId:
    """Tests for sanitize_model_id function."""

    def test_extracts_model_name_from_org_path(self):
        """Test extracting model name from org/model format."""
        from tensortruth.indexing.metadata import sanitize_model_id

        assert sanitize_model_id("BAAI/bge-m3") == "bge-m3"
        assert (
            sanitize_model_id("sentence-transformers/all-MiniLM-L6-v2")
            == "all-minilm-l6-v2"
        )
        assert sanitize_model_id("BAAI/bge-reranker-v2-m3") == "bge-reranker-v2-m3"

    def test_handles_simple_model_names(self):
        """Test simple model names without org prefix."""
        from tensortruth.indexing.metadata import sanitize_model_id

        assert sanitize_model_id("bge-m3") == "bge-m3"
        assert sanitize_model_id("BGE-M3") == "bge-m3"

    def test_replaces_unsafe_characters(self):
        """Test that unsafe filesystem characters are replaced."""
        from tensortruth.indexing.metadata import sanitize_model_id

        assert sanitize_model_id("model@version") == "model-version"
        assert sanitize_model_id("model:latest") == "model-latest"

    def test_removes_consecutive_dashes(self):
        """Test that consecutive dashes are collapsed."""
        from tensortruth.indexing.metadata import sanitize_model_id

        result = sanitize_model_id("model--name")
        assert "--" not in result

    def test_strips_leading_trailing_dashes(self):
        """Test that leading/trailing dashes are removed."""
        from tensortruth.indexing.metadata import sanitize_model_id

        assert not sanitize_model_id("-model-").startswith("-")
        assert not sanitize_model_id("-model-").endswith("-")


@pytest.mark.unit
class TestResolveEmbeddingModelName:
    """Tests for resolve_embedding_model_name function."""

    def test_full_path_returned_as_is(self):
        """Test that a full HuggingFace path is returned unchanged."""
        from tensortruth.indexing.metadata import resolve_embedding_model_name

        assert resolve_embedding_model_name("BAAI/bge-m3") == "BAAI/bge-m3"

    def test_sanitized_id_resolved(self):
        """Test that a sanitized model ID is resolved to the full path."""
        from tensortruth.indexing.metadata import resolve_embedding_model_name

        assert resolve_embedding_model_name("bge-m3") == "BAAI/bge-m3"

    def test_unknown_id_returned_as_is(self):
        """Test that an unknown sanitized ID is returned unchanged."""
        from tensortruth.indexing.metadata import resolve_embedding_model_name

        assert resolve_embedding_model_name("totally-unknown") == "totally-unknown"


@pytest.mark.unit
class TestWriteAndReadIndexMetadata:
    """Tests for write_index_metadata and read_index_metadata functions."""

    def test_write_creates_metadata_file(self, tmp_path):
        """Test that write_index_metadata creates the metadata file."""
        from tensortruth.indexing.metadata import (
            INDEX_METADATA_FILENAME,
            write_index_metadata,
        )

        write_index_metadata(
            index_dir=tmp_path,
            embedding_model="BAAI/bge-m3",
            chunk_sizes=[2048, 512, 256],
        )

        metadata_file = tmp_path / INDEX_METADATA_FILENAME
        assert metadata_file.exists()

    def test_metadata_contains_required_fields(self, tmp_path):
        """Test that written metadata contains all required fields."""
        from tensortruth.indexing.metadata import (
            INDEX_METADATA_FILENAME,
            write_index_metadata,
        )

        write_index_metadata(
            index_dir=tmp_path,
            embedding_model="BAAI/bge-m3",
            chunk_sizes=[2048, 512, 256],
        )

        with open(tmp_path / INDEX_METADATA_FILENAME) as f:
            metadata = json.load(f)

        assert metadata["embedding_model"] == "BAAI/bge-m3"
        assert metadata["embedding_model_id"] == "bge-m3"
        assert metadata["chunk_sizes"] == [2048, 512, 256]
        assert "created_at" in metadata
        assert "index_version" in metadata

    def test_read_returns_written_metadata(self, tmp_path):
        """Test that read_index_metadata returns what was written."""
        from tensortruth.indexing.metadata import (
            read_index_metadata,
            write_index_metadata,
        )

        write_index_metadata(
            index_dir=tmp_path,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_sizes=[1024, 256],
        )

        metadata = read_index_metadata(tmp_path)

        assert metadata is not None
        assert metadata["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert metadata["embedding_model_id"] == "all-minilm-l6-v2"
        assert metadata["chunk_sizes"] == [1024, 256]

    def test_read_returns_none_for_missing_file(self, tmp_path):
        """Test that read_index_metadata returns None if no metadata file."""
        from tensortruth.indexing.metadata import read_index_metadata

        result = read_index_metadata(tmp_path)
        assert result is None

    def test_extra_metadata_is_included(self, tmp_path):
        """Test that extra_metadata parameter is merged into output."""
        from tensortruth.indexing.metadata import (
            read_index_metadata,
            write_index_metadata,
        )

        write_index_metadata(
            index_dir=tmp_path,
            embedding_model="BAAI/bge-m3",
            chunk_sizes=[2048, 512],
            extra_metadata={"custom_field": "custom_value"},
        )

        metadata = read_index_metadata(tmp_path)
        assert metadata["custom_field"] == "custom_value"


@pytest.mark.unit
class TestGetEmbeddingModelFromIndex:
    """Tests for get_embedding_model_from_index function."""

    def test_returns_model_name(self, tmp_path):
        """Test that it returns the correct model name."""
        from tensortruth.indexing.metadata import (
            get_embedding_model_from_index,
            write_index_metadata,
        )

        write_index_metadata(tmp_path, "BAAI/bge-m3", [2048])
        result = get_embedding_model_from_index(tmp_path)
        assert result == "BAAI/bge-m3"

    def test_returns_none_for_no_metadata(self, tmp_path):
        """Test that it returns None when no metadata exists."""
        from tensortruth.indexing.metadata import get_embedding_model_from_index

        result = get_embedding_model_from_index(tmp_path)
        assert result is None


@pytest.mark.unit
class TestIsValidIndexDir:
    """Tests for is_valid_index_dir function."""

    def test_returns_false_for_nonexistent_dir(self, tmp_path):
        """Test that non-existent directory returns False."""
        from tensortruth.indexing.metadata import is_valid_index_dir

        result = is_valid_index_dir(tmp_path / "nonexistent")
        assert result is False

    def test_returns_false_for_empty_dir(self, tmp_path):
        """Test that empty directory returns False."""
        from tensortruth.indexing.metadata import is_valid_index_dir

        result = is_valid_index_dir(tmp_path)
        assert result is False

    def test_returns_true_for_valid_chromadb_dir(self, tmp_path):
        """Test that directory with chroma.sqlite3 returns True."""
        from tensortruth.indexing.metadata import is_valid_index_dir

        (tmp_path / "chroma.sqlite3").touch()
        result = is_valid_index_dir(tmp_path)
        assert result is True


@pytest.mark.unit
class TestDetectLegacyIndex:
    """Tests for detect_legacy_index function."""

    def test_returns_false_for_invalid_index(self, tmp_path):
        """Test that invalid index directory returns False."""
        from tensortruth.indexing.metadata import detect_legacy_index

        result = detect_legacy_index(tmp_path)
        assert result is False

    def test_returns_true_for_legacy_index(self, tmp_path):
        """Test that valid index without metadata is detected as legacy."""
        from tensortruth.indexing.metadata import detect_legacy_index

        # Create a valid ChromaDB marker
        (tmp_path / "chroma.sqlite3").touch()

        result = detect_legacy_index(tmp_path)
        assert result is True

    def test_returns_false_for_index_with_metadata(self, tmp_path):
        """Test that index with metadata is not detected as legacy."""
        from tensortruth.indexing.metadata import (
            detect_legacy_index,
            write_index_metadata,
        )

        # Create a valid ChromaDB marker and metadata
        (tmp_path / "chroma.sqlite3").touch()
        write_index_metadata(tmp_path, "BAAI/bge-m3", [2048])

        result = detect_legacy_index(tmp_path)
        assert result is False


# ============================================================================
# Test ModelManager Singleton
# ============================================================================


@pytest.mark.unit
class TestModelManagerSingleton:
    """Tests for ModelManager singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        from tensortruth.services.model_manager import ModelManager

        ModelManager.reset_instance()

    def test_singleton_returns_same_instance(self):
        """Test that get_instance returns the same object."""
        from tensortruth.services.model_manager import ModelManager

        instance1 = ModelManager.get_instance()
        instance2 = ModelManager.get_instance()

        assert instance1 is instance2

    def test_reset_instance_clears_singleton(self):
        """Test that reset_instance creates a new singleton."""
        from tensortruth.services.model_manager import ModelManager

        instance1 = ModelManager.get_instance()
        ModelManager.reset_instance()
        instance2 = ModelManager.get_instance()

        assert instance1 is not instance2

    def test_get_status_returns_dict(self):
        """Test that get_status returns a valid status dict."""
        from tensortruth.services.model_manager import ModelManager

        manager = ModelManager.get_instance()
        status = manager.get_status()

        assert "embedder" in status
        assert "reranker" in status
        assert "default_device" in status

    def test_initial_state_has_no_models_loaded(self):
        """Test that initial state has no models loaded."""
        from tensortruth.services.model_manager import ModelManager

        manager = ModelManager.get_instance()
        status = manager.get_status()

        assert status["embedder"]["loaded"] is False
        assert status["reranker"]["loaded"] is False

    def test_set_default_device(self):
        """Test that set_default_device updates the default."""
        from tensortruth.services.model_manager import ModelManager

        manager = ModelManager.get_instance()
        manager.set_default_device("cuda")

        assert manager.get_status()["default_device"] == "cuda"


@pytest.mark.unit
class TestModelManagerEmbedder:
    """Tests for ModelManager embedder functionality (mocked)."""

    def setup_method(self):
        """Reset singleton before each test."""
        from tensortruth.services.model_manager import ModelManager

        ModelManager.reset_instance()

    @patch("tensortruth.services.model_manager.HuggingFaceEmbedding")
    def test_get_embedder_loads_model(self, mock_hf_embedding):
        """Test that get_embedder loads the model."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance = MagicMock()
        mock_hf_embedding.return_value = mock_instance

        manager = ModelManager.get_instance()
        embedder = manager.get_embedder(model_name="BAAI/bge-m3", device="cpu")

        assert embedder is mock_instance
        mock_hf_embedding.assert_called_once()

    @patch("tensortruth.services.model_manager.HuggingFaceEmbedding")
    def test_get_embedder_reuses_loaded_model(self, mock_hf_embedding):
        """Test that get_embedder reuses already loaded model."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance = MagicMock()
        mock_hf_embedding.return_value = mock_instance

        manager = ModelManager.get_instance()
        embedder1 = manager.get_embedder(model_name="BAAI/bge-m3", device="cpu")
        embedder2 = manager.get_embedder(model_name="BAAI/bge-m3", device="cpu")

        assert embedder1 is embedder2
        assert mock_hf_embedding.call_count == 1

    @patch("tensortruth.services.model_manager.HuggingFaceEmbedding")
    def test_get_embedder_swaps_when_different_model(self, mock_hf_embedding):
        """Test that get_embedder swaps model when different model requested."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_hf_embedding.side_effect = [mock_instance1, mock_instance2]

        manager = ModelManager.get_instance()
        embedder1 = manager.get_embedder(model_name="BAAI/bge-m3", device="cpu")
        embedder2 = manager.get_embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )

        assert embedder1 is not embedder2
        assert mock_hf_embedding.call_count == 2

    @patch("tensortruth.services.model_manager.HuggingFaceEmbedding")
    def test_get_embedder_swaps_when_device_changes(self, mock_hf_embedding):
        """Test that get_embedder reloads model when device changes (cpu -> cuda)."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_hf_embedding.side_effect = [mock_instance1, mock_instance2]

        manager = ModelManager.get_instance()
        # Load on CPU
        embedder1 = manager.get_embedder(model_name="BAAI/bge-m3", device="cpu")
        # Request same model but on CUDA - should trigger reload
        embedder2 = manager.get_embedder(model_name="BAAI/bge-m3", device="cuda")

        assert embedder1 is not embedder2
        assert mock_hf_embedding.call_count == 2

        # Verify the device was passed correctly in each call
        calls = mock_hf_embedding.call_args_list
        assert calls[0].kwargs["device"] == "cpu"
        assert calls[1].kwargs["device"] == "cuda"

    @patch("tensortruth.services.model_manager.HuggingFaceEmbedding")
    def test_get_embedder_no_reload_same_model_same_device(self, mock_hf_embedding):
        """Test that requesting same model and device does not reload."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance = MagicMock()
        mock_hf_embedding.return_value = mock_instance

        manager = ModelManager.get_instance()
        embedder1 = manager.get_embedder(model_name="BAAI/bge-m3", device="cuda")
        embedder2 = manager.get_embedder(model_name="BAAI/bge-m3", device="cuda")
        embedder3 = manager.get_embedder(model_name="BAAI/bge-m3", device="cuda")

        assert embedder1 is embedder2 is embedder3
        assert mock_hf_embedding.call_count == 1


@pytest.mark.unit
class TestModelManagerReranker:
    """Tests for ModelManager reranker functionality (mocked)."""

    def setup_method(self):
        """Reset singleton before each test."""
        from tensortruth.services.model_manager import ModelManager

        ModelManager.reset_instance()

    @patch("tensortruth.services.model_manager.SentenceTransformerRerank")
    def test_get_reranker_loads_model(self, mock_reranker):
        """Test that get_reranker loads the model."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance = MagicMock()
        mock_reranker.return_value = mock_instance

        manager = ModelManager.get_instance()
        reranker = manager.get_reranker(
            model_name="BAAI/bge-reranker-v2-m3", device="cpu"
        )

        assert reranker is mock_instance
        mock_reranker.assert_called_once()

    @patch("tensortruth.services.model_manager.SentenceTransformerRerank")
    def test_get_reranker_reuses_loaded_model(self, mock_reranker):
        """Test that get_reranker reuses already loaded model."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance = MagicMock()
        mock_reranker.return_value = mock_instance

        manager = ModelManager.get_instance()
        reranker1 = manager.get_reranker(
            model_name="BAAI/bge-reranker-v2-m3", device="cpu"
        )
        reranker2 = manager.get_reranker(
            model_name="BAAI/bge-reranker-v2-m3", device="cpu"
        )

        assert reranker1 is reranker2
        assert mock_reranker.call_count == 1

    @patch("tensortruth.services.model_manager.SentenceTransformerRerank")
    def test_get_reranker_swaps_when_different_model(self, mock_reranker):
        """Test that get_reranker swaps when different model requested."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_reranker.side_effect = [mock_instance1, mock_instance2]

        manager = ModelManager.get_instance()
        reranker1 = manager.get_reranker(
            model_name="BAAI/bge-reranker-v2-m3", device="cpu"
        )
        reranker2 = manager.get_reranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

        assert reranker1 is not reranker2
        assert mock_reranker.call_count == 2

    @patch("tensortruth.services.model_manager.SentenceTransformerRerank")
    def test_get_reranker_swaps_when_device_changes(self, mock_reranker):
        """Test that get_reranker reloads when device changes."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_reranker.side_effect = [mock_instance1, mock_instance2]

        manager = ModelManager.get_instance()
        reranker1 = manager.get_reranker(
            model_name="BAAI/bge-reranker-v2-m3", device="cpu"
        )
        reranker2 = manager.get_reranker(
            model_name="BAAI/bge-reranker-v2-m3", device="cuda"
        )

        assert reranker1 is not reranker2
        assert mock_reranker.call_count == 2

    @patch("tensortruth.services.model_manager.SentenceTransformerRerank")
    def test_get_reranker_top_n_change_no_reload(self, mock_reranker):
        """Test that changing top_n triggers a reload.

        Note: LlamaIndex's SentenceTransformerRerank may cache top_n internally,
        so we reload the model when top_n changes to ensure correct behavior.
        """
        from tensortruth.services.model_manager import ModelManager

        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_reranker.side_effect = [mock_instance1, mock_instance2]

        manager = ModelManager.get_instance()
        reranker1 = manager.get_reranker(
            model_name="BAAI/bge-reranker-v2-m3", top_n=5, device="cpu"
        )
        reranker2 = manager.get_reranker(
            model_name="BAAI/bge-reranker-v2-m3", top_n=10, device="cpu"
        )

        # Different instances due to top_n change
        assert reranker1 is not reranker2
        assert mock_reranker.call_count == 2
        # Verify both were created with correct top_n
        assert mock_reranker.call_args_list[0].kwargs["top_n"] == 5
        assert mock_reranker.call_args_list[1].kwargs["top_n"] == 10


# ============================================================================
# Test Path Helpers for Embedding Model Directories
# ============================================================================


@pytest.mark.unit
class TestGetIndexesDirForModel:
    """Tests for get_indexes_dir_for_model function."""

    def test_creates_model_subdirectory(self, tmp_path, monkeypatch):
        """Test that model subdirectory is created."""
        # Avoid using actual home directory
        monkeypatch.setenv("TENSOR_TRUTH_INDEXES_DIR", str(tmp_path))

        from tensortruth.app_utils.paths import get_indexes_dir_for_model

        result = get_indexes_dir_for_model("BAAI/bge-m3")

        assert result.exists()
        assert result.name == "bge-m3"
        assert result.parent == tmp_path

    def test_sanitizes_model_name(self, tmp_path, monkeypatch):
        """Test that model name is sanitized for filesystem."""
        monkeypatch.setenv("TENSOR_TRUTH_INDEXES_DIR", str(tmp_path))

        from tensortruth.app_utils.paths import get_indexes_dir_for_model

        result = get_indexes_dir_for_model("sentence-transformers/all-MiniLM-L6-v2")

        assert result.name == "all-minilm-l6-v2"

    def test_respects_base_dir_override(self, tmp_path):
        """Test that base_dir_override is respected."""
        from tensortruth.app_utils.paths import get_indexes_dir_for_model

        custom_base = tmp_path / "custom_indexes"
        custom_base.mkdir()

        result = get_indexes_dir_for_model(
            "BAAI/bge-m3", base_dir_override=str(custom_base)
        )

        assert result.parent == custom_base


@pytest.mark.unit
class TestGetModuleIndexDir:
    """Tests for get_module_index_dir function."""

    def test_returns_correct_path_structure(self, tmp_path, monkeypatch):
        """Test that returned path has correct structure."""
        monkeypatch.setenv("TENSOR_TRUTH_INDEXES_DIR", str(tmp_path))

        from tensortruth.app_utils.paths import get_module_index_dir

        result = get_module_index_dir("library_pytorch_2.9", "BAAI/bge-m3")

        # Should be: tmp_path/bge-m3/library_pytorch_2.9
        assert result.name == "library_pytorch_2.9"
        assert result.parent.name == "bge-m3"


# ============================================================================
# Test Migration Utility
# ============================================================================


@pytest.mark.unit
class TestDetectLegacyIndexes:
    """Tests for detect_legacy_indexes function."""

    def test_returns_empty_for_nonexistent_dir(self, tmp_path):
        """Test that nonexistent directory returns empty list."""
        from tensortruth.indexing.migration import detect_legacy_indexes

        result = detect_legacy_indexes(tmp_path / "nonexistent")
        assert result == []

    def test_returns_empty_for_empty_dir(self, tmp_path):
        """Test that empty directory returns empty list."""
        from tensortruth.indexing.migration import detect_legacy_indexes

        result = detect_legacy_indexes(tmp_path)
        assert result == []

    def test_detects_legacy_indexes(self, tmp_path):
        """Test that legacy indexes are detected."""
        from tensortruth.indexing.migration import detect_legacy_indexes

        # Create a legacy index (no metadata, has chroma.sqlite3)
        legacy_index = tmp_path / "library_pytorch"
        legacy_index.mkdir()
        (legacy_index / "chroma.sqlite3").touch()

        result = detect_legacy_indexes(tmp_path)

        assert len(result) == 1
        assert result[0] == legacy_index

    def test_ignores_versioned_indexes(self, tmp_path):
        """Test that versioned indexes are not detected as legacy."""
        from tensortruth.indexing.metadata import write_index_metadata
        from tensortruth.indexing.migration import detect_legacy_indexes

        # Create a versioned structure
        versioned = tmp_path / "bge-m3" / "library_pytorch"
        versioned.mkdir(parents=True)
        (versioned / "chroma.sqlite3").touch()
        write_index_metadata(versioned, "BAAI/bge-m3", [2048])

        result = detect_legacy_indexes(tmp_path)

        # The versioned index should not be detected as legacy
        assert len(result) == 0


@pytest.mark.unit
class TestMigrateLegacyIndexes:
    """Tests for migrate_legacy_indexes function."""

    def test_dry_run_does_not_move_files(self, tmp_path):
        """Test that dry_run=True doesn't actually move files."""
        from tensortruth.indexing.migration import migrate_legacy_indexes

        # Create a legacy index
        legacy_index = tmp_path / "library_pytorch"
        legacy_index.mkdir()
        (legacy_index / "chroma.sqlite3").touch()

        migrated, failed = migrate_legacy_indexes(tmp_path, dry_run=True)

        assert "library_pytorch" in migrated
        assert legacy_index.exists()  # Still in original location

    def test_migration_moves_to_correct_location(self, tmp_path):
        """Test that migration moves index to versioned structure."""
        from tensortruth.indexing.migration import migrate_legacy_indexes

        # Create a legacy index
        legacy_index = tmp_path / "library_pytorch"
        legacy_index.mkdir()
        (legacy_index / "chroma.sqlite3").touch()

        migrated, failed = migrate_legacy_indexes(tmp_path)

        assert "library_pytorch" in migrated
        assert not legacy_index.exists()  # Moved from original location
        assert (tmp_path / "bge-m3" / "library_pytorch" / "chroma.sqlite3").exists()

    def test_migration_writes_metadata(self, tmp_path):
        """Test that migration writes metadata file."""
        from tensortruth.indexing.metadata import INDEX_METADATA_FILENAME
        from tensortruth.indexing.migration import migrate_legacy_indexes

        # Create a legacy index
        legacy_index = tmp_path / "library_pytorch"
        legacy_index.mkdir()
        (legacy_index / "chroma.sqlite3").touch()

        migrate_legacy_indexes(tmp_path)

        metadata_path = (
            tmp_path / "bge-m3" / "library_pytorch" / INDEX_METADATA_FILENAME
        )
        assert metadata_path.exists()


@pytest.mark.unit
class TestGetMigrationStatus:
    """Tests for get_migration_status function."""

    def test_returns_correct_structure(self, tmp_path):
        """Test that status dict has correct structure."""
        from tensortruth.indexing.migration import get_migration_status

        status = get_migration_status(tmp_path)

        assert "has_legacy" in status
        assert "legacy_count" in status
        assert "legacy_modules" in status
        assert "versioned_by_model" in status
        assert "total_versioned" in status

    def test_detects_legacy_indexes(self, tmp_path):
        """Test that legacy indexes are counted."""
        from tensortruth.indexing.migration import get_migration_status

        # Create legacy indexes
        for name in ["library_pytorch", "library_numpy"]:
            idx = tmp_path / name
            idx.mkdir()
            (idx / "chroma.sqlite3").touch()

        status = get_migration_status(tmp_path)

        assert status["has_legacy"] is True
        assert status["legacy_count"] == 2

    def test_counts_versioned_by_model(self, tmp_path):
        """Test that versioned indexes are counted by model."""
        from tensortruth.indexing.metadata import write_index_metadata
        from tensortruth.indexing.migration import get_migration_status

        # Create versioned indexes for two models
        for model_id in ["bge-m3", "all-minilm-l6-v2"]:
            model_dir = tmp_path / model_id
            for name in ["library_pytorch", "library_numpy"]:
                idx = model_dir / name
                idx.mkdir(parents=True)
                (idx / "chroma.sqlite3").touch()
                write_index_metadata(idx, f"test/{model_id}", [2048])

        status = get_migration_status(tmp_path)

        assert status["versioned_by_model"]["bge-m3"] == 2
        assert status["versioned_by_model"]["all-minilm-l6-v2"] == 2
        assert status["total_versioned"] == 4


# ============================================================================
# Test Modules Endpoint with Versioned Indexes
# ============================================================================


@pytest.mark.unit
class TestModulesEndpointVersionedStructure:
    """Tests for /modules endpoint with versioned index structure."""

    @pytest.fixture
    def mock_config_service(self):
        """Create a mock config service."""
        mock_service = MagicMock()
        mock_config = MagicMock()
        mock_config.rag.default_embedding_model = "BAAI/bge-m3"
        mock_service.load.return_value = mock_config
        return mock_service

    def test_finds_modules_in_versioned_structure(self, tmp_path, mock_config_service):
        """Test that modules are found in indexes/{model_id}/{module}/ structure."""
        from tensortruth.api.routes.modules import list_modules

        # Create versioned index structure
        model_dir = tmp_path / "bge-m3"
        for name in ["library_pytorch_2.9", "library_numpy_2.3", "book_deep_learning"]:
            idx = model_dir / name
            idx.mkdir(parents=True)
            (idx / "chroma.sqlite3").touch()

        with patch(
            "tensortruth.api.routes.modules.get_indexes_dir", return_value=tmp_path
        ):
            import asyncio

            result = asyncio.run(list_modules(mock_config_service))

        module_names = [m.name for m in result.modules]
        assert len(module_names) == 3
        assert "library_pytorch_2.9" in module_names
        assert "library_numpy_2.3" in module_names
        assert "book_deep_learning" in module_names

    def test_ignores_modules_for_other_embedding_models(
        self, tmp_path, mock_config_service
    ):
        """Test that only modules for configured embedding model are returned."""
        from tensortruth.api.routes.modules import list_modules

        # Create indexes for configured model (bge-m3)
        bge_dir = tmp_path / "bge-m3"
        (bge_dir / "library_pytorch").mkdir(parents=True)
        (bge_dir / "library_pytorch" / "chroma.sqlite3").touch()

        # Create indexes for different model (should be ignored)
        other_dir = tmp_path / "all-minilm-l6-v2"
        (other_dir / "library_numpy").mkdir(parents=True)
        (other_dir / "library_numpy" / "chroma.sqlite3").touch()

        with patch(
            "tensortruth.api.routes.modules.get_indexes_dir", return_value=tmp_path
        ):
            import asyncio

            result = asyncio.run(list_modules(mock_config_service))

        module_names = [m.name for m in result.modules]
        assert len(module_names) == 1
        assert "library_pytorch" in module_names
        assert "library_numpy" not in module_names

    def test_returns_empty_for_flat_structure(self, tmp_path, mock_config_service):
        """Test that flat structure indexes are not returned (migration required)."""
        from tensortruth.api.routes.modules import list_modules

        # Create legacy flat structure (no model subdirectory)
        # These should NOT be returned - migration is required
        for name in ["library_pytorch", "library_numpy"]:
            idx = tmp_path / name
            idx.mkdir()
            (idx / "chroma.sqlite3").touch()

        with patch(
            "tensortruth.api.routes.modules.get_indexes_dir", return_value=tmp_path
        ):
            import asyncio

            result = asyncio.run(list_modules(mock_config_service))

        # No modules returned because they're in flat structure
        assert len(result.modules) == 0

    def test_returns_empty_when_no_indexes(self, tmp_path, mock_config_service):
        """Test that empty list is returned when no indexes exist."""
        from tensortruth.api.routes.modules import list_modules

        with patch(
            "tensortruth.api.routes.modules.get_indexes_dir", return_value=tmp_path
        ):
            import asyncio

            result = asyncio.run(list_modules(mock_config_service))

        assert len(result.modules) == 0

    def test_switches_models_based_on_config(self, tmp_path):
        """Test that changing config embedding model switches which modules are shown."""
        from tensortruth.api.routes.modules import list_modules

        # Create indexes for two models
        bge_dir = tmp_path / "bge-m3"
        (bge_dir / "library_bge").mkdir(parents=True)
        (bge_dir / "library_bge" / "chroma.sqlite3").touch()

        minilm_dir = tmp_path / "all-minilm-l6-v2"
        (minilm_dir / "library_minilm").mkdir(parents=True)
        (minilm_dir / "library_minilm" / "chroma.sqlite3").touch()

        # Test with bge-m3 config
        mock_service_bge = MagicMock()
        mock_config_bge = MagicMock()
        mock_config_bge.rag.default_embedding_model = "BAAI/bge-m3"
        mock_service_bge.load.return_value = mock_config_bge

        with patch(
            "tensortruth.api.routes.modules.get_indexes_dir", return_value=tmp_path
        ):
            import asyncio

            result_bge = asyncio.run(list_modules(mock_service_bge))

        assert [m.name for m in result_bge.modules] == ["library_bge"]

        # Test with minilm config
        mock_service_minilm = MagicMock()
        mock_config_minilm = MagicMock()
        mock_config_minilm.rag.default_embedding_model = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        mock_service_minilm.load.return_value = mock_config_minilm

        with patch(
            "tensortruth.api.routes.modules.get_indexes_dir", return_value=tmp_path
        ):
            result_minilm = asyncio.run(list_modules(mock_service_minilm))

        assert [m.name for m in result_minilm.modules] == ["library_minilm"]


# ============================================================================
# Test Embedding Model Configuration
# ============================================================================


@pytest.mark.unit
class TestEmbeddingModelConfig:
    """Tests for EmbeddingModelConfig dataclass and defaults."""

    def test_default_config_has_sensible_values(self):
        """Test that default config has reasonable default values."""
        from tensortruth.app_utils.config_schema import EmbeddingModelConfig

        config = EmbeddingModelConfig()

        assert config.batch_size_cuda == 128
        assert config.batch_size_cpu == 16
        assert config.torch_dtype is None
        assert config.padding_side is None
        assert config.flash_attention is False
        assert config.trust_remote_code is True

    def test_custom_values_override_defaults(self):
        """Test that custom values can be provided."""
        from tensortruth.app_utils.config_schema import EmbeddingModelConfig

        config = EmbeddingModelConfig(
            batch_size_cuda=8,
            batch_size_cpu=4,
            torch_dtype="float16",
            padding_side="left",
            flash_attention=True,
        )

        assert config.batch_size_cuda == 8
        assert config.batch_size_cpu == 4
        assert config.torch_dtype == "float16"
        assert config.padding_side == "left"
        assert config.flash_attention is True


@pytest.mark.unit
class TestDefaultEmbeddingModelConfigs:
    """Tests for DEFAULT_EMBEDDING_MODEL_CONFIGS dictionary."""

    def test_bge_m3_config_exists(self):
        """Test that BGE-M3 config is defined."""
        from tensortruth.app_utils.config_schema import DEFAULT_EMBEDDING_MODEL_CONFIGS

        assert "BAAI/bge-m3" in DEFAULT_EMBEDDING_MODEL_CONFIGS


@pytest.mark.unit
class TestRAGConfigGetEmbeddingModelConfig:
    """Tests for RAGConfig.get_embedding_model_config method."""

    def test_returns_user_config_when_present(self):
        """Test that user-configured model config is returned."""
        from tensortruth.app_utils.config_schema import RAGConfig

        custom_config = {
            "batch_size_cuda": 32,
            "batch_size_cpu": 8,
            "torch_dtype": "bfloat16",
            "padding_side": "right",
            "flash_attention": False,
            "trust_remote_code": False,
        }

        rag_config = RAGConfig(
            embedding_model_configs={"my-custom/model": custom_config}
        )

        result = rag_config.get_embedding_model_config("my-custom/model")

        assert result.batch_size_cuda == 32
        assert result.batch_size_cpu == 8
        assert result.torch_dtype == "bfloat16"
        assert result.padding_side == "right"
        assert result.flash_attention is False
        assert result.trust_remote_code is False

    def test_falls_back_to_builtin_defaults(self):
        """Test that built-in defaults are used when user config is missing."""
        from tensortruth.app_utils.config_schema import RAGConfig

        rag_config = RAGConfig(embedding_model_configs={})

        result = rag_config.get_embedding_model_config("BAAI/bge-m3")

        # Should get values from DEFAULT_EMBEDDING_MODEL_CONFIGS
        assert result.batch_size_cuda == 128
        assert result.batch_size_cpu == 16
        assert result.trust_remote_code is True

    def test_returns_generic_defaults_for_unknown_model(self):
        """Test that generic defaults are returned for unknown models."""
        from tensortruth.app_utils.config_schema import RAGConfig

        rag_config = RAGConfig(embedding_model_configs={})

        result = rag_config.get_embedding_model_config("unknown/model")

        # Should get generic defaults
        assert result.batch_size_cuda == 128
        assert result.batch_size_cpu == 16
        assert result.torch_dtype is None
        assert result.padding_side is None
        assert result.flash_attention is False

    def test_user_config_overrides_builtin_defaults(self):
        """Test that user config takes precedence over built-in defaults."""
        from tensortruth.app_utils.config_schema import RAGConfig

        # User wants different settings for BGE-M3
        user_config = {
            "batch_size_cuda": 64,  # Different from built-in default of 128
            "batch_size_cpu": 8,
            "torch_dtype": "float32",
            "padding_side": "left",
            "flash_attention": True,
            "trust_remote_code": True,
        }

        rag_config = RAGConfig(embedding_model_configs={"BAAI/bge-m3": user_config})

        result = rag_config.get_embedding_model_config("BAAI/bge-m3")

        assert result.batch_size_cuda == 64
        assert result.torch_dtype == "float32"
        assert result.flash_attention is True


@pytest.mark.unit
class TestModelManagerEmbeddingModelConfig:
    """Tests for ModelManager's _get_embedding_model_config method."""

    def setup_method(self):
        """Reset singleton before each test."""
        from tensortruth.services.model_manager import ModelManager

        ModelManager.reset_instance()

    def test_returns_config_for_known_model(self):
        """Test that config is returned for known models."""
        from tensortruth.services.model_manager import ModelManager

        manager = ModelManager.get_instance()
        config = manager._get_embedding_model_config("BAAI/bge-m3")

        # Should get BGE-M3 config
        assert config.batch_size_cuda == 128
        assert config.batch_size_cpu == 16
        assert config.trust_remote_code is True

    def test_returns_default_config_for_unknown_model(self):
        """Test that default config is returned for unknown models."""
        from tensortruth.services.model_manager import ModelManager

        manager = ModelManager.get_instance()
        config = manager._get_embedding_model_config("completely-unknown/model")

        # Should get generic defaults
        assert config.batch_size_cuda == 128
        assert config.torch_dtype is None

    @patch("tensortruth.services.model_manager.HuggingFaceEmbedding")
    def test_load_embedder_uses_model_config(self, mock_hf_embedding):
        """Test that _load_embedder applies model-specific config."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance = MagicMock()
        mock_hf_embedding.return_value = mock_instance

        manager = ModelManager.get_instance()
        manager._load_embedder("BAAI/bge-m3", "cpu")

        # Check that HuggingFaceEmbedding was called with BGE-M3 settings
        call_kwargs = mock_hf_embedding.call_args.kwargs

        # Batch size should be 16 for BGE-M3 on CPU
        assert call_kwargs["embed_batch_size"] == 16

        # Should not have special tokenizer_kwargs for BGE-M3
        assert call_kwargs.get("tokenizer_kwargs") is None

    @patch("tensortruth.services.model_manager.HuggingFaceEmbedding")
    def test_load_embedder_uses_default_for_bge(self, mock_hf_embedding):
        """Test that _load_embedder uses default settings for BGE-M3."""
        from tensortruth.services.model_manager import ModelManager

        mock_instance = MagicMock()
        mock_hf_embedding.return_value = mock_instance

        manager = ModelManager.get_instance()
        manager._load_embedder("BAAI/bge-m3", "cuda")

        # Check that HuggingFaceEmbedding was called with BGE-M3 settings
        call_kwargs = mock_hf_embedding.call_args.kwargs

        # Batch size should be large for BGE-M3 on CUDA
        assert call_kwargs["embed_batch_size"] == 128

        # Should not have special tokenizer_kwargs for BGE-M3
        assert call_kwargs.get("tokenizer_kwargs") is None
