"""Unit tests for DocumentService."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.services.document_service import DocumentService


@pytest.fixture
def temp_scope_dir(tmp_path):
    """Create a temporary scope directory."""
    scope_dir = tmp_path / "test_scope"
    scope_dir.mkdir(parents=True, exist_ok=True)
    return scope_dir


@pytest.fixture
def session_service(temp_scope_dir):
    """Create a DocumentService with session scope."""
    return DocumentService(
        scope_id="sess_123",
        scope_dir=temp_scope_dir,
        scope_type="session",
    )


@pytest.fixture
def project_service(temp_scope_dir):
    """Create a DocumentService with project scope."""
    return DocumentService(
        scope_id="proj_456",
        scope_dir=temp_scope_dir,
        scope_type="project",
    )


class TestUploadText:
    """Test text/markdown file upload."""

    def test_upload_text_creates_markdown_file(self, session_service, temp_scope_dir):
        """Should create a markdown file in the markdown directory."""
        content = b"Hello, this is plain text content."
        result = session_service.upload_text(content, "notes.txt")

        md_path = Path(result.path)
        assert md_path.exists()
        assert md_path.parent == temp_scope_dir / "markdown"
        assert md_path.suffix == ".md"

        file_content = md_path.read_text()
        assert "Hello, this is plain text content." in file_content
        assert "# Document: notes.txt" in file_content

    def test_upload_text_header_session_scope(self, session_service):
        """Should include 'Session Upload' in header for session scope."""
        content = b"Some text"
        result = session_service.upload_text(content, "file.txt")

        file_content = Path(result.path).read_text()
        assert "# Source: Session Upload" in file_content

    def test_upload_text_header_project_scope(self, project_service):
        """Should include 'Project Upload' in header for project scope."""
        content = b"Some text"
        result = project_service.upload_text(content, "file.txt")

        file_content = Path(result.path).read_text()
        assert "# Source: Project Upload" in file_content

    def test_upload_text_returns_metadata(self, session_service):
        """Should return proper PDFMetadata."""
        content = b"Content here"
        result = session_service.upload_text(content, "readme.md")

        assert result.pdf_id.startswith("doc_")
        assert result.filename == "readme.md"
        assert result.file_size == len(content)
        assert result.page_count == 0


class TestUploadDocument:
    """Test document dispatch by file type."""

    def test_dispatches_pdf(self, session_service):
        """Should route .pdf to upload()."""
        with patch.object(session_service, "upload") as mock_upload:
            mock_upload.return_value = MagicMock()
            session_service.upload_document(b"pdf bytes", "paper.pdf")
            mock_upload.assert_called_once_with(b"pdf bytes", "paper.pdf")

    def test_dispatches_txt(self, session_service):
        """Should route .txt to upload_text()."""
        with patch.object(session_service, "upload_text") as mock_text:
            mock_text.return_value = MagicMock()
            session_service.upload_document(b"text bytes", "notes.txt")
            mock_text.assert_called_once_with(b"text bytes", "notes.txt")

    def test_dispatches_md(self, session_service):
        """Should route .md to upload_text()."""
        with patch.object(session_service, "upload_text") as mock_text:
            mock_text.return_value = MagicMock()
            session_service.upload_document(b"md bytes", "readme.md")
            mock_text.assert_called_once_with(b"md bytes", "readme.md")

    def test_dispatches_markdown(self, session_service):
        """Should route .markdown to upload_text()."""
        with patch.object(session_service, "upload_text") as mock_text:
            mock_text.return_value = MagicMock()
            session_service.upload_document(b"md bytes", "doc.markdown")
            mock_text.assert_called_once_with(b"md bytes", "doc.markdown")

    def test_rejects_unsupported(self, session_service):
        """Should raise ValueError for unsupported file types."""
        with pytest.raises(ValueError, match="Unsupported file type: .docx"):
            session_service.upload_document(b"bytes", "file.docx")


class TestGetIndexBuilder:
    """Test index builder creation."""

    def test_uses_explicit_dirs(self, session_service, temp_scope_dir):
        """Should create DocumentIndexBuilder with scope_dir/index and scope_dir/markdown."""
        builder = session_service._get_index_builder()
        assert builder.index_dir == temp_scope_dir / "index"
        assert builder.markdown_dir == temp_scope_dir / "markdown"

    def test_caches_builder(self, session_service):
        """Should return the same builder instance on repeated calls."""
        builder1 = session_service._get_index_builder()
        builder2 = session_service._get_index_builder()
        assert builder1 is builder2


class TestScopeTypePropagation:
    """Test that scope_type is passed through correctly."""

    def test_scope_type_passed_to_handler(self, temp_scope_dir):
        """Should pass scope_type to PDFHandler."""
        service = DocumentService(
            scope_id="proj_789",
            scope_dir=temp_scope_dir,
            scope_type="project",
        )
        assert service._pdf_handler.scope_type == "project"

    def test_scope_type_defaults_to_session(self, temp_scope_dir):
        """Should default scope_type to 'session'."""
        service = DocumentService(
            scope_id="sess_abc",
            scope_dir=temp_scope_dir,
        )
        assert service._pdf_handler.scope_type == "session"
