"""Unit tests for indexing.builder module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from tensortruth.core.types import DocumentType
from tensortruth.indexing.builder import build_module, extract_metadata


@pytest.mark.unit
class TestExtractMetadata:
    """Tests for extract_metadata function."""

    @pytest.fixture
    def mock_documents(self):
        """Create mock documents."""
        doc = Mock()
        doc.metadata = {"file_path": "/path/to/doc.md"}
        return [doc]

    @pytest.fixture
    def sources_config(self):
        """Sample sources config."""
        return {
            "libraries": {
                "pytorch": {
                    "type": "sphinx",
                    "version": "2.0",
                    "doc_root": "https://pytorch.org/docs/",
                }
            },
            "papers": {
                "dl_foundations": {
                    "type": "arxiv",
                    "display_name": "DL Foundations",
                    "items": {
                        "1706.03762": {
                            "title": "Attention Is All You Need",
                            "arxiv_id": "1706.03762",
                            "source": "https://arxiv.org/abs/1706.03762",
                            "authors": "Vaswani et al.",
                            "year": "2017",
                        }
                    },
                }
            },
            "books": {
                "linear_algebra": {
                    "type": "pdf_book",
                    "title": "Linear Algebra",
                    "authors": ["Author"],
                    "source": "https://example.com/book.pdf",
                    "category": "math",
                    "split_method": "toc",
                }
            },
        }

    @patch("tensortruth.indexing.builder.extract_library_module_metadata")
    @patch("tensortruth.indexing.builder.extract_library_metadata_from_config")
    def test_extract_metadata_library(
        self,
        mock_get_lib_meta,
        mock_extract_lib_meta,
        mock_documents,
        sources_config,
    ):
        """Test metadata extraction for library documents."""
        mock_get_lib_meta.return_value = {
            "title": "PyTorch",
            "library_display_name": "PyTorch 2.0",
        }
        mock_extract_lib_meta.return_value = {
            "title": "PyTorch Intro",
            "display_name": "PyTorch 2.0 > intro",
            "library_display_name": "PyTorch 2.0",
        }

        extract_metadata(
            "pytorch", DocumentType.LIBRARY, sources_config, mock_documents
        )

        # Verify extraction was called
        mock_get_lib_meta.assert_called_once_with("pytorch", sources_config)
        mock_extract_lib_meta.assert_called_once()

        # Verify metadata was added to document
        assert "title" in mock_documents[0].metadata
        assert "display_name" in mock_documents[0].metadata

    @patch("tensortruth.indexing.builder.extract_arxiv_metadata_from_config")
    def test_extract_metadata_papers(
        self, mock_extract_arxiv, mock_documents, sources_config
    ):
        """Test metadata extraction for paper documents."""
        mock_extract_arxiv.return_value = {
            "title": "Attention Is All You Need",
            "formatted_authors": "Vaswani et al.",
            "display_name": "Attention Is All You Need, Vaswani et al.",
        }

        extract_metadata(
            "dl_foundations", DocumentType.PAPERS, sources_config, mock_documents
        )

        mock_extract_arxiv.assert_called_once()
        assert "title" in mock_documents[0].metadata

    @patch("tensortruth.indexing.builder.extract_book_chapter_metadata")
    @patch("tensortruth.indexing.builder.get_book_metadata_from_config")
    def test_extract_metadata_book(
        self,
        mock_get_book,
        mock_extract_chapter,
        mock_documents,
        sources_config,
    ):
        """Test metadata extraction for book documents."""
        mock_get_book.return_value = {
            "title": "Linear Algebra",
            "authors": ["Author"],
            "formatted_authors": "Author",
            "book_display_name": "Linear Algebra, Author",
        }
        mock_extract_chapter.return_value = {
            "title": "Linear Algebra",
            "display_name": "Linear Algebra Ch.1, Author",
            "book_display_name": "Linear Algebra, Author",
        }

        extract_metadata(
            "linear_algebra", DocumentType.BOOK, sources_config, mock_documents
        )

        mock_get_book.assert_called_once_with("linear_algebra", sources_config)
        mock_extract_chapter.assert_called_once()

    def test_extract_metadata_progress_callback(self, mock_documents, sources_config):
        """Test that progress callback is called."""
        callback = Mock()

        with patch(
            "tensortruth.indexing.builder.extract_arxiv_metadata_from_config"
        ) as mock_extract:
            mock_extract.return_value = {
                "title": "Test",
                "display_name": "Test Paper",
            }

            extract_metadata(
                "dl_foundations",
                DocumentType.PAPERS,
                sources_config,
                mock_documents,
                progress_callback=callback,
            )

            # Callback should be called at least once
            assert callback.call_count >= 1

    def test_extract_metadata_handles_errors_gracefully(
        self, mock_documents, sources_config
    ):
        """Test that extraction continues even if metadata extraction fails."""
        with patch(
            "tensortruth.indexing.builder.extract_arxiv_metadata_from_config"
        ) as mock_extract:
            mock_extract.side_effect = Exception("Test error")

            # Should not raise, just log warning
            extract_metadata(
                "dl_foundations", DocumentType.PAPERS, sources_config, mock_documents
            )


@pytest.mark.unit
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
                    "doc_root": "https://example.com/docs/",
                }
            },
        }

    @pytest.fixture
    def temp_docs_dir(self, tmp_path):
        """Create temporary docs directory with content."""
        docs_dir = tmp_path / "library_docs"
        docs_dir.mkdir()

        lib_dir = docs_dir / "library_test_lib"
        lib_dir.mkdir()
        (lib_dir / "intro.md").write_text("# Introduction\n\nTest content.")
        (lib_dir / "api.md").write_text("# API Reference\n\nAPI docs.")

        return str(docs_dir)

    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_build_module_success(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test successful module build."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()

        indexes_dir = str(tmp_path / "indexes")

        result = build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
        )

        assert result is True
        # With versioned structure, index is in indexes/{model_id}/module/
        assert Path(indexes_dir, "bge-m3", "library_test_lib").exists()
        mock_index.assert_called_once()

    def test_build_module_missing_source_dir(
        self, tmp_path, mock_sources_config, caplog
    ):
        """Test that missing source directory returns False."""
        library_docs_dir = str(tmp_path / "nonexistent_docs")
        indexes_dir = str(tmp_path / "indexes")

        result = build_module(
            "test_lib",
            library_docs_dir,
            indexes_dir,
            mock_sources_config,
        )

        assert result is False
        assert "missing" in caplog.text.lower()

    @patch("tensortruth.indexing.builder.SimpleDirectoryReader")
    def test_build_module_no_documents(
        self, mock_reader, tmp_path, temp_docs_dir, mock_sources_config, caplog
    ):
        """Test that empty directory returns False."""
        mock_reader.return_value.load_data.return_value = []

        indexes_dir = str(tmp_path / "indexes")

        result = build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
        )

        assert result is False
        assert "no documents" in caplog.text.lower()

    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_build_module_custom_chunk_sizes(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test build with custom chunk sizes."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()

        indexes_dir = str(tmp_path / "indexes")

        result = build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            chunk_sizes=[4096, 1024, 256],
        )

        assert result is True

    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    def test_build_module_custom_device(
        self,
        mock_embed,
        mock_index,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test build with custom device."""
        mock_embed.return_value = MagicMock()

        indexes_dir = str(tmp_path / "indexes")

        result = build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            device="cuda",
        )

        assert result is True
        # Verify _create_embed_model was called with custom device
        mock_embed.assert_called_with("BAAI/bge-m3", "cuda")

    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_build_module_progress_callback(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that progress callback is called."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()

        indexes_dir = str(tmp_path / "indexes")
        callback = Mock()

        result = build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            progress_callback=callback,
        )

        assert result is True
        # Callback should be called for different stages
        assert callback.call_count >= 2  # At least metadata and embedding stages

    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_build_module_replaces_old_index(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that old index is replaced."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()

        indexes_dir = str(tmp_path / "indexes")
        # With versioned structure, index is in indexes/{model_id}/module/
        index_path = Path(indexes_dir, "bge-m3", "library_test_lib")

        # Create existing index in versioned path
        index_path.mkdir(parents=True)
        old_file = index_path / "old_data.txt"
        old_file.write_text("old index data")

        result = build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
        )

        assert result is True
        # Old file should be gone (directory is replaced)
        assert not old_file.exists()


@pytest.mark.unit
class TestChunkOverlap:
    """Tests for chunk overlap functionality (Phase 1)."""

    def test_default_chunk_overlap_constant_exists(self):
        """Test that DEFAULT_CHUNK_OVERLAP constant is defined."""
        from tensortruth.indexing.builder import DEFAULT_CHUNK_OVERLAP

        assert DEFAULT_CHUNK_OVERLAP == 64

    @pytest.fixture
    def mock_sources_config(self):
        """Mock sources configuration."""
        return {
            "libraries": {
                "test_lib": {
                    "type": "sphinx",
                    "version": "1.0",
                    "doc_root": "https://example.com/docs/",
                }
            },
        }

    @pytest.fixture
    def temp_docs_dir(self, tmp_path):
        """Create temporary docs directory with content."""
        docs_dir = tmp_path / "library_docs"
        docs_dir.mkdir()

        lib_dir = docs_dir / "library_test_lib"
        lib_dir.mkdir()
        (lib_dir / "intro.md").write_text("# Introduction\n\nTest content.")

        return str(docs_dir)

    @patch("tensortruth.indexing.builder.HierarchicalNodeParser")
    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_build_module_passes_chunk_overlap_to_parser(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        mock_parser_class,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that build_module passes chunk_overlap to HierarchicalNodeParser."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()
        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.return_value = []
        mock_parser_class.from_defaults.return_value = mock_parser

        indexes_dir = str(tmp_path / "indexes")

        build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            chunk_overlap=128,
        )

        # Verify chunk_overlap was passed to the parser
        mock_parser_class.from_defaults.assert_called_once()
        call_kwargs = mock_parser_class.from_defaults.call_args.kwargs
        assert call_kwargs.get("chunk_overlap") == 128

    @patch("tensortruth.indexing.builder.HierarchicalNodeParser")
    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_build_module_default_chunk_overlap(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        mock_parser_class,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that build_module uses default chunk_overlap when not specified."""
        from tensortruth.indexing.builder import DEFAULT_CHUNK_OVERLAP

        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()
        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.return_value = []
        mock_parser_class.from_defaults.return_value = mock_parser

        indexes_dir = str(tmp_path / "indexes")

        build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
        )

        # Verify default chunk_overlap was used
        mock_parser_class.from_defaults.assert_called_once()
        call_kwargs = mock_parser_class.from_defaults.call_args.kwargs
        assert call_kwargs.get("chunk_overlap") == DEFAULT_CHUNK_OVERLAP

    @patch("tensortruth.indexing.builder.HierarchicalNodeParser")
    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_build_module_custom_chunk_overlap(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        mock_parser_class,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test build with custom chunk overlap value."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()
        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.return_value = []
        mock_parser_class.from_defaults.return_value = mock_parser

        indexes_dir = str(tmp_path / "indexes")

        build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            chunk_overlap=512,
        )

        call_kwargs = mock_parser_class.from_defaults.call_args.kwargs
        assert call_kwargs.get("chunk_overlap") == 512

    @patch("tensortruth.indexing.builder.write_index_metadata")
    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_chunk_overlap_recorded_in_metadata(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        mock_write_metadata,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that chunk_overlap is recorded in index metadata."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()

        indexes_dir = str(tmp_path / "indexes")

        build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            chunk_overlap=256,
        )

        # Verify write_index_metadata was called with chunk_overlap
        mock_write_metadata.assert_called_once()
        call_kwargs = mock_write_metadata.call_args.kwargs
        assert call_kwargs.get("chunk_overlap") == 256


@pytest.mark.unit
class TestSemanticChunking:
    """Tests for semantic chunking functionality (Phase 2)."""

    def test_chunking_strategy_enum_values(self):
        """Test that ChunkingStrategy enum has expected values."""
        from tensortruth.indexing.builder import ChunkingStrategy

        assert hasattr(ChunkingStrategy, "HIERARCHICAL")
        assert hasattr(ChunkingStrategy, "SEMANTIC")
        assert hasattr(ChunkingStrategy, "SEMANTIC_HIERARCHICAL")
        assert ChunkingStrategy.HIERARCHICAL.value == "hierarchical"
        assert ChunkingStrategy.SEMANTIC.value == "semantic"
        assert ChunkingStrategy.SEMANTIC_HIERARCHICAL.value == "semantic_hierarchical"

    @pytest.fixture
    def mock_sources_config(self):
        """Mock sources configuration."""
        return {
            "libraries": {
                "test_lib": {
                    "type": "sphinx",
                    "version": "1.0",
                    "doc_root": "https://example.com/docs/",
                }
            },
        }

    @pytest.fixture
    def temp_docs_dir(self, tmp_path):
        """Create temporary docs directory with content."""
        docs_dir = tmp_path / "library_docs"
        docs_dir.mkdir()

        lib_dir = docs_dir / "library_test_lib"
        lib_dir.mkdir()
        (lib_dir / "intro.md").write_text("# Introduction\n\nTest content.")

        return str(docs_dir)

    @patch("tensortruth.indexing.builder.HierarchicalNodeParser")
    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_hierarchical_strategy_uses_hierarchical_parser(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        mock_parser_class,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that hierarchical strategy uses HierarchicalNodeParser."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()
        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.return_value = []
        mock_parser_class.from_defaults.return_value = mock_parser

        indexes_dir = str(tmp_path / "indexes")

        from tensortruth.indexing.builder import build_module

        build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            chunking_strategy="hierarchical",
        )

        # Verify HierarchicalNodeParser was used
        mock_parser_class.from_defaults.assert_called_once()

    @patch("tensortruth.indexing.builder.SemanticSplitterNodeParser")
    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_semantic_strategy_uses_semantic_splitter(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        mock_semantic_parser,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that semantic strategy uses SemanticSplitterNodeParser."""
        mock_detect_device.return_value = "cpu"
        mock_embed_model = MagicMock()
        mock_embed.return_value = mock_embed_model
        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.return_value = []
        mock_semantic_parser.return_value = mock_parser

        indexes_dir = str(tmp_path / "indexes")

        from tensortruth.indexing.builder import build_module

        build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            chunking_strategy="semantic",
        )

        # Verify SemanticSplitterNodeParser was used
        mock_semantic_parser.assert_called_once()
        call_kwargs = mock_semantic_parser.call_args.kwargs
        # Should pass embed_model for semantic splitting
        assert "embed_model" in call_kwargs

    @patch("tensortruth.indexing.builder.SemanticSplitterNodeParser")
    @patch("tensortruth.indexing.builder.HierarchicalNodeParser")
    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_semantic_hierarchical_applies_both_parsers(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        mock_hierarchical_parser,
        mock_semantic_parser,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that semantic_hierarchical uses both parsers."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()

        # Semantic parser returns documents
        mock_semantic = MagicMock()
        mock_semantic.get_nodes_from_documents.return_value = [MagicMock()]
        mock_semantic_parser.return_value = mock_semantic

        # Hierarchical parser returns final nodes
        mock_hierarchical = MagicMock()
        mock_hierarchical.get_nodes_from_documents.return_value = []
        mock_hierarchical_parser.from_defaults.return_value = mock_hierarchical

        indexes_dir = str(tmp_path / "indexes")

        from tensortruth.indexing.builder import build_module

        build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            chunking_strategy="semantic_hierarchical",
        )

        # Both parsers should be called
        mock_semantic_parser.assert_called_once()
        mock_hierarchical_parser.from_defaults.assert_called_once()

    @patch("tensortruth.indexing.builder.SemanticSplitterNodeParser")
    @patch("tensortruth.indexing.builder.write_index_metadata")
    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_chunking_strategy_recorded_in_metadata(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        mock_write_metadata,
        mock_semantic_parser,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that chunking_strategy is recorded in index metadata."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()
        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.return_value = []
        mock_semantic_parser.return_value = mock_parser

        indexes_dir = str(tmp_path / "indexes")

        from tensortruth.indexing.builder import build_module

        build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            chunking_strategy="semantic",
        )

        # Verify write_index_metadata was called with chunking_strategy
        mock_write_metadata.assert_called_once()
        call_kwargs = mock_write_metadata.call_args.kwargs
        assert call_kwargs.get("chunking_strategy") == "semantic"

    @patch("tensortruth.indexing.builder.SemanticSplitterNodeParser")
    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_semantic_buffer_size_parameter(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        mock_semantic_parser,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that semantic_buffer_size is passed to SemanticSplitterNodeParser."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()
        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.return_value = []
        mock_semantic_parser.return_value = mock_parser

        indexes_dir = str(tmp_path / "indexes")

        from tensortruth.indexing.builder import build_module

        build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            chunking_strategy="semantic",
            semantic_buffer_size=3,
        )

        call_kwargs = mock_semantic_parser.call_args.kwargs
        assert call_kwargs.get("buffer_size") == 3

    @patch("tensortruth.indexing.builder.SemanticSplitterNodeParser")
    @patch("tensortruth.indexing.builder.VectorStoreIndex")
    @patch("tensortruth.indexing.builder._create_embed_model")
    @patch("tensortruth.indexing.builder.TensorTruthConfig._detect_default_device")
    def test_semantic_breakpoint_threshold_parameter(
        self,
        mock_detect_device,
        mock_embed,
        mock_index,
        mock_semantic_parser,
        tmp_path,
        temp_docs_dir,
        mock_sources_config,
    ):
        """Test that semantic_breakpoint_threshold is passed to SemanticSplitterNodeParser."""
        mock_detect_device.return_value = "cpu"
        mock_embed.return_value = MagicMock()
        mock_parser = MagicMock()
        mock_parser.get_nodes_from_documents.return_value = []
        mock_semantic_parser.return_value = mock_parser

        indexes_dir = str(tmp_path / "indexes")

        from tensortruth.indexing.builder import build_module

        build_module(
            "test_lib",
            temp_docs_dir,
            indexes_dir,
            mock_sources_config,
            chunking_strategy="semantic",
            semantic_breakpoint_threshold=90,
        )

        call_kwargs = mock_semantic_parser.call_args.kwargs
        assert call_kwargs.get("breakpoint_percentile_threshold") == 90
