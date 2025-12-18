"""Unit tests for session index builder."""

import shutil
from unittest.mock import Mock, patch

import pytest

from tensortruth.session_index import SessionIndexBuilder


@pytest.fixture
def session_id():
    """Test session ID."""
    return "test_session_123"


@pytest.fixture
def temp_session_dirs(tmp_path, session_id):
    """Create temporary session directories."""
    session_dir = tmp_path / "sessions" / session_id
    markdown_dir = session_dir / "markdown"
    index_dir = session_dir / "index"

    markdown_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    return {"session": session_dir, "markdown": markdown_dir, "index": index_dir}


@pytest.fixture
def mock_paths(temp_session_dirs):
    """Mock path functions to return temp directories."""
    with (
        patch("tensortruth.session_index.get_session_index_dir") as mock_index,
        patch("tensortruth.session_index.get_session_markdown_dir") as mock_md,
    ):
        mock_index.return_value = temp_session_dirs["index"]
        mock_md.return_value = temp_session_dirs["markdown"]
        yield {"index": mock_index, "markdown": mock_md}


@pytest.fixture
def session_builder(session_id, mock_paths):
    """Create SessionIndexBuilder with mocked paths."""
    return SessionIndexBuilder(session_id)


@pytest.fixture
def sample_markdown_files(temp_session_dirs):
    """Create sample markdown files for testing."""
    markdown_dir = temp_session_dirs["markdown"]

    files = []
    for i in range(3):
        md_file = markdown_dir / f"document_{i}.md"
        md_file.write_text(
            f"# Document {i}\n\nThis is test content for document {i}.\n"
        )
        files.append(md_file)

    return files


class TestSessionIndexBuilderInit:
    """Test SessionIndexBuilder initialization."""

    def test_stores_session_id(self, session_id, mock_paths):
        """Should store session ID."""
        builder = SessionIndexBuilder(session_id)
        assert builder.session_id == session_id

    def test_sets_directory_paths(self, session_builder, temp_session_dirs):
        """Should set correct directory paths."""
        assert session_builder.session_index_dir == temp_session_dirs["index"]
        assert session_builder.session_markdown_dir == temp_session_dirs["markdown"]


class TestIndexExists:
    """Test index existence check."""

    def test_returns_false_when_no_index(self, session_builder):
        """Should return False if index doesn't exist."""
        # Clear index directory
        if session_builder.session_index_dir.exists():
            shutil.rmtree(session_builder.session_index_dir)
        session_builder.session_index_dir.mkdir(parents=True)

        assert not session_builder.index_exists()

    def test_returns_false_without_chroma_db(self, session_builder):
        """Should return False if chroma.sqlite3 missing."""
        # Create only docstore
        (session_builder.session_index_dir / "docstore.json").write_text("{}")

        assert not session_builder.index_exists()

    def test_returns_false_without_docstore(self, session_builder):
        """Should return False if docstore.json missing."""
        # Create only chroma db
        (session_builder.session_index_dir / "chroma.sqlite3").write_text("")

        assert not session_builder.index_exists()

    def test_returns_true_with_both_files(self, session_builder):
        """Should return True if both required files exist."""
        (session_builder.session_index_dir / "chroma.sqlite3").write_text("")
        (session_builder.session_index_dir / "docstore.json").write_text("{}")

        assert session_builder.index_exists()


class TestBuildIndex:
    """Test index building."""

    @patch("tensortruth.session_index.StorageContext")
    @patch("tensortruth.session_index.VectorStoreIndex")
    @patch("tensortruth.session_index.get_embed_model")
    @patch("tensortruth.session_index.chromadb.PersistentClient")
    @patch("tensortruth.session_index.get_leaf_nodes")
    @patch("tensortruth.session_index.HierarchicalNodeParser")
    @patch("tensortruth.session_index.SimpleDirectoryReader")
    def test_loads_markdown_files(
        self,
        mock_reader,
        mock_parser,
        mock_get_leaf,
        mock_chroma,
        mock_embed,
        mock_index,
        mock_storage,
        session_builder,
        sample_markdown_files,
    ):
        """Should load all markdown files."""

        # Mock document loading
        mock_docs = [Mock(text=f"doc {i}") for i in range(len(sample_markdown_files))]
        mock_reader.return_value.load_data.return_value = mock_docs

        # Mock node parsing
        mock_parser_instance = Mock()
        mock_parser_instance.get_nodes_from_documents.return_value = [Mock()]
        mock_parser.from_defaults.return_value = mock_parser_instance
        mock_get_leaf.return_value = [Mock()]

        # Mock the rest of the pipeline
        mock_embed.return_value = Mock()
        mock_chroma.return_value = Mock()
        mock_index.return_value = Mock()
        mock_storage.from_defaults.return_value = Mock()

        session_builder.build_index(sample_markdown_files)

        # Should have created reader for each file
        assert mock_reader.call_count == len(sample_markdown_files)

    @patch("tensortruth.session_index.StorageContext")
    @patch("tensortruth.session_index.VectorStoreIndex")
    @patch("tensortruth.session_index.get_embed_model")
    @patch("tensortruth.session_index.chromadb.PersistentClient")
    @patch("tensortruth.session_index.get_leaf_nodes")
    @patch("tensortruth.session_index.HierarchicalNodeParser")
    @patch("tensortruth.session_index.SimpleDirectoryReader")
    def test_uses_hierarchical_chunking(
        self,
        mock_reader,
        mock_parser,
        mock_get_leaf,
        mock_chroma,
        mock_embed,
        mock_index,
        mock_storage,
        session_builder,
        sample_markdown_files,
    ):
        """Should use HierarchicalNodeParser with correct chunk sizes."""
        mock_reader.return_value.load_data.return_value = [Mock(text="content")]
        mock_parser_instance = Mock()
        mock_parser_instance.get_nodes_from_documents.return_value = [Mock()]
        mock_parser.from_defaults.return_value = mock_parser_instance
        mock_get_leaf.return_value = [Mock()]
        mock_storage.from_defaults.return_value = Mock()

        session_builder.build_index(sample_markdown_files, chunk_sizes=[2048, 512, 128])

        mock_parser.from_defaults.assert_called_once_with(chunk_sizes=[2048, 512, 128])

    @patch("tensortruth.session_index.StorageContext")
    @patch("tensortruth.session_index.VectorStoreIndex")
    @patch("tensortruth.session_index.get_embed_model")
    @patch("tensortruth.session_index.chromadb.PersistentClient")
    @patch("tensortruth.session_index.get_leaf_nodes")
    @patch("tensortruth.session_index.HierarchicalNodeParser")
    @patch("tensortruth.session_index.SimpleDirectoryReader")
    def test_creates_chroma_db(
        self,
        mock_reader,
        mock_parser,
        mock_get_leaf,
        mock_chroma,
        mock_embed,
        mock_index,
        mock_storage,
        session_builder,
        sample_markdown_files,
        temp_session_dirs,
    ):
        """Should create ChromaDB in session index directory."""
        mock_reader.return_value.load_data.return_value = [Mock(text="content")]

        # Mock node parsing
        mock_parser_instance = Mock()
        mock_parser_instance.get_nodes_from_documents.return_value = [Mock()]
        mock_parser.from_defaults.return_value = mock_parser_instance
        mock_get_leaf.return_value = [Mock()]
        mock_storage.from_defaults.return_value = Mock()

        mock_embed.return_value = Mock()

        session_builder.build_index(sample_markdown_files)

        mock_chroma.assert_called_once_with(path=str(temp_session_dirs["index"]))

    @patch("tensortruth.session_index.SimpleDirectoryReader")
    def test_raises_on_no_markdown_files(self, mock_reader, session_builder):
        """Should raise error if no markdown files provided."""
        with pytest.raises(ValueError, match="No markdown files found"):
            session_builder.build_index([])

    @patch("tensortruth.session_index.StorageContext")
    @patch("tensortruth.session_index.VectorStoreIndex")
    @patch("tensortruth.session_index.get_embed_model")
    @patch("tensortruth.session_index.chromadb.PersistentClient")
    @patch("tensortruth.session_index.get_leaf_nodes")
    @patch("tensortruth.session_index.HierarchicalNodeParser")
    @patch("tensortruth.session_index.SimpleDirectoryReader")
    def test_cleans_existing_index(
        self,
        mock_reader,
        mock_parser,
        mock_get_leaf,
        mock_chroma,
        mock_embed,
        mock_index,
        mock_storage,
        session_builder,
        sample_markdown_files,
    ):
        """Should remove old index before building new one."""
        mock_reader.return_value.load_data.return_value = [Mock(text="content")]

        # Mock node parsing
        mock_parser_instance = Mock()
        mock_parser_instance.get_nodes_from_documents.return_value = [Mock()]
        mock_parser.from_defaults.return_value = mock_parser_instance
        mock_get_leaf.return_value = [Mock()]
        mock_storage.from_defaults.return_value = Mock()

        mock_embed.return_value = Mock()

        # Create fake old index files
        old_file = session_builder.session_index_dir / "old_data.json"
        old_file.write_text("{}")
        assert old_file.exists()

        session_builder.build_index(sample_markdown_files)

        # Old files should be gone
        assert not old_file.exists()

    def test_uses_default_chunk_sizes(self, session_builder, sample_markdown_files):
        """Should use default chunk sizes if none provided."""
        with (
            patch("tensortruth.session_index.StorageContext") as mock_storage,
            patch("tensortruth.session_index.HierarchicalNodeParser") as mock_parser,
            patch("tensortruth.session_index.SimpleDirectoryReader") as mock_reader,
            patch("tensortruth.session_index.get_leaf_nodes") as mock_get_leaf,
            patch("tensortruth.session_index.VectorStoreIndex"),
            patch("tensortruth.session_index.get_embed_model"),
            patch("tensortruth.session_index.chromadb.PersistentClient"),
        ):
            mock_reader.return_value.load_data.return_value = [Mock(text="content")]
            mock_parser_instance = Mock()
            mock_parser_instance.get_nodes_from_documents.return_value = [Mock()]
            mock_parser.from_defaults.return_value = mock_parser_instance
            mock_get_leaf.return_value = [Mock()]
            mock_storage.from_defaults.return_value = Mock()

            session_builder.build_index(sample_markdown_files)

            # Should use default [2048, 512, 256]
            mock_parser.from_defaults.assert_called_once_with(
                chunk_sizes=[2048, 512, 256]
            )


class TestRebuildIndex:
    """Test index rebuilding."""

    @patch.object(SessionIndexBuilder, "build_index")
    def test_calls_build_with_no_files(self, mock_build, session_builder):
        """Should call build_index with None for markdown_files."""
        session_builder.rebuild_index()

        mock_build.assert_called_once_with(markdown_files=None, chunk_sizes=None)

    @patch.object(SessionIndexBuilder, "build_index")
    def test_passes_chunk_sizes(self, mock_build, session_builder):
        """Should pass custom chunk sizes to build_index."""
        custom_sizes = [4096, 1024, 256]
        session_builder.rebuild_index(chunk_sizes=custom_sizes)

        mock_build.assert_called_once_with(
            markdown_files=None, chunk_sizes=custom_sizes
        )


class TestDeleteIndex:
    """Test index deletion."""

    def test_deletes_index_directory(self, session_builder):
        """Should remove entire index directory."""
        # Create some files in index
        (session_builder.session_index_dir / "chroma.sqlite3").write_text("")
        (session_builder.session_index_dir / "docstore.json").write_text("{}")
        assert session_builder.session_index_dir.exists()

        session_builder.delete_index()

        assert not session_builder.session_index_dir.exists()

    def test_handles_nonexistent_index(self, session_builder):
        """Should not crash if index doesn't exist."""
        # Remove index directory
        if session_builder.session_index_dir.exists():
            shutil.rmtree(session_builder.session_index_dir)

        # Should not raise
        session_builder.delete_index()


class TestGetIndexSize:
    """Test index size calculation."""

    def test_returns_zero_for_nonexistent_index(self, session_builder):
        """Should return 0 if index doesn't exist."""
        if session_builder.session_index_dir.exists():
            shutil.rmtree(session_builder.session_index_dir)

        assert session_builder.get_index_size() == 0

    def test_calculates_total_size(self, session_builder):
        """Should sum all file sizes in index directory."""
        # Create files with known sizes
        (session_builder.session_index_dir / "file1.txt").write_bytes(b"x" * 100)
        (session_builder.session_index_dir / "file2.txt").write_bytes(b"y" * 200)

        # Create subdirectory with file
        subdir = session_builder.session_index_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_bytes(b"z" * 50)

        size = session_builder.get_index_size()
        assert size == 350  # 100 + 200 + 50


class TestGetDocumentCount:
    """Test document count."""

    def test_returns_zero_for_no_documents(self, session_builder):
        """Should return 0 if no markdown files."""
        assert session_builder.get_document_count() == 0

    def test_counts_markdown_files(self, session_builder, sample_markdown_files):
        """Should count all .md files in markdown directory."""
        assert session_builder.get_document_count() == len(sample_markdown_files)

    def test_ignores_non_markdown_files(self, session_builder, sample_markdown_files):
        """Should only count .md files."""
        # Add a non-markdown file
        (session_builder.session_markdown_dir / "readme.txt").write_text("not markdown")

        # Should still only count .md files
        assert session_builder.get_document_count() == len(sample_markdown_files)
