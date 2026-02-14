"""Unit tests for PDF handler."""

from unittest.mock import patch

import pytest

from tensortruth.pdf_handler import PDFHandler


@pytest.fixture
def temp_session_dir(tmp_path):
    """Create a temporary session directory."""
    session_dir = tmp_path / "test_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


@pytest.fixture
def pdf_handler(temp_session_dir):
    """Create a PDFHandler instance with temp directory."""
    return PDFHandler(temp_session_dir)


@pytest.fixture
def sample_content():
    return b"fake pdf content"


@pytest.fixture
def sample_filename():
    return "test_paper.pdf"


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal valid PDF for testing."""
    pdf_path = tmp_path / "sample.pdf"
    # Minimal PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
197
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


class TestPDFHandlerInit:
    """Test PDFHandler initialization."""

    def test_creates_directories(self, temp_session_dir):
        """Should create documents and markdown directories."""
        handler = PDFHandler(temp_session_dir)
        assert handler.documents_dir.exists()
        assert handler.markdown_dir.exists()
        assert handler.documents_dir == temp_session_dir / "documents"
        assert handler.markdown_dir == temp_session_dir / "markdown"

    def test_handles_existing_directories(self, temp_session_dir):
        """Should not fail if directories already exist."""
        # Create directories first
        (temp_session_dir / "documents").mkdir(parents=True, exist_ok=True)
        (temp_session_dir / "markdown").mkdir(parents=True, exist_ok=True)

        # Should not raise
        handler = PDFHandler(temp_session_dir)
        assert handler.documents_dir.exists()
        assert handler.markdown_dir.exists()

    def test_uses_pdfs_dir_fallback(self, temp_session_dir):
        """Should use existing pdfs/ dir for legacy sessions."""
        (temp_session_dir / "pdfs").mkdir(parents=True, exist_ok=True)

        handler = PDFHandler(temp_session_dir)
        assert handler.documents_dir == temp_session_dir / "pdfs"

    def test_documents_dir_takes_precedence(self, temp_session_dir):
        """Should prefer documents/ over pdfs/ when both exist."""
        (temp_session_dir / "pdfs").mkdir(parents=True, exist_ok=True)
        (temp_session_dir / "documents").mkdir(parents=True, exist_ok=True)

        handler = PDFHandler(temp_session_dir)
        assert handler.documents_dir == temp_session_dir / "documents"

    def test_scope_type_in_header_session(self, temp_session_dir, sample_pdf):
        """Should use 'Session Upload' in header for session scope."""
        handler = PDFHandler(temp_session_dir, scope_type="session")
        with patch("tensortruth.pdf_handler.convert_with_marker") as mock_marker:
            mock_marker.return_value = "Content"
            result = handler.convert_pdf_to_markdown(sample_pdf)
            content = result.read_text()
            assert "# Source: Session Upload" in content

    def test_scope_type_in_header_project(self, temp_session_dir, sample_pdf):
        """Should use 'Project Upload' in header for project scope."""
        handler = PDFHandler(temp_session_dir, scope_type="project")
        with patch("tensortruth.pdf_handler.convert_with_marker") as mock_marker:
            mock_marker.return_value = "Content"
            result = handler.convert_pdf_to_markdown(sample_pdf)
            content = result.read_text()
            assert "# Source: Project Upload" in content


class TestUploadPDF:
    """Test PDF upload functionality."""

    def test_saves_pdf_with_unique_id(
        self, pdf_handler, sample_content, sample_filename
    ):
        """Should save PDF with unique identifier prefix."""
        metadata = pdf_handler.upload_pdf(sample_content, sample_filename)

        assert "id" in metadata
        assert metadata["id"].startswith("pdf_")
        assert "path" in metadata
        assert "filename" in metadata
        assert metadata["filename"] == "test_paper.pdf"

    def test_pdf_file_created(self, pdf_handler, sample_content, sample_filename):
        """Should create PDF file on disk."""
        metadata = pdf_handler.upload_pdf(sample_content, sample_filename)
        pdf_path = metadata["path"]

        assert pdf_path.exists()
        assert pdf_path.read_bytes() == b"fake pdf content"

    @patch.object(PDFHandler, "get_pdf_metadata")
    def test_extracts_file_size(
        self, mock_get_metadata, pdf_handler, sample_content, sample_filename
    ):
        """Should extract file size metadata."""
        # Mock metadata extraction to return valid data
        mock_get_metadata.return_value = {
            "file_size": len(b"fake pdf content"),
            "page_count": 1,
        }

        metadata = pdf_handler.upload_pdf(sample_content, sample_filename)

        assert "file_size" in metadata
        assert metadata["file_size"] == len(b"fake pdf content")


class TestGetPDFMetadata:
    """Test PDF metadata extraction."""

    def test_extracts_page_count(self, pdf_handler, sample_pdf):
        """Should extract page count from PDF."""
        metadata = pdf_handler.get_pdf_metadata(sample_pdf)

        assert "page_count" in metadata
        assert metadata["page_count"] >= 0

    def test_extracts_file_size(self, pdf_handler, sample_pdf):
        """Should extract file size."""
        metadata = pdf_handler.get_pdf_metadata(sample_pdf)

        assert "file_size" in metadata
        assert metadata["file_size"] > 0
        assert metadata["file_size"] == sample_pdf.stat().st_size

    def test_handles_corrupted_pdf(self, pdf_handler, tmp_path):
        """Should handle corrupted PDF gracefully."""
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a valid pdf")

        metadata = pdf_handler.get_pdf_metadata(bad_pdf)

        # Should return default values instead of crashing
        assert metadata["page_count"] == 0
        assert metadata["file_size"] >= 0


class TestConvertPDFToMarkdown:
    """Test PDF to markdown conversion."""

    @patch("tensortruth.pdf_handler.convert_with_marker")
    def test_tries_marker_first(self, mock_marker, pdf_handler, sample_pdf):
        """Should attempt marker-pdf conversion first."""
        mock_marker.return_value = "# Test Content\n\nConverted with marker"

        result = pdf_handler.convert_pdf_to_markdown(sample_pdf, use_marker=True)

        mock_marker.assert_called_once_with(str(sample_pdf))
        assert result.exists()
        assert "Test Content" in result.read_text()

    @patch("tensortruth.pdf_handler.convert_pdf_to_markdown")
    @patch("tensortruth.pdf_handler.convert_with_marker")
    def test_falls_back_to_pymupdf(
        self, mock_marker, mock_pymupdf, pdf_handler, sample_pdf
    ):
        """Should fall back to pymupdf4llm if marker fails."""
        mock_marker.side_effect = Exception("Marker failed")
        mock_pymupdf.return_value = "# Fallback Content"

        result = pdf_handler.convert_pdf_to_markdown(sample_pdf, use_marker=True)

        # Should have tried both
        assert mock_marker.call_count == 1
        # Fallback triggered
        assert result.exists()

    @patch("tensortruth.pdf_handler.convert_with_marker")
    def test_adds_metadata_header(self, mock_marker, pdf_handler, sample_pdf):
        """Should add metadata header to markdown."""
        mock_marker.return_value = "PDF content here"

        result = pdf_handler.convert_pdf_to_markdown(sample_pdf)
        content = result.read_text()

        assert "# Document:" in content
        assert "# Source: Session Upload" in content
        assert "PDF content here" in content

    @patch("tensortruth.pdf_handler.convert_with_marker")
    def test_creates_markdown_file(self, mock_marker, pdf_handler, sample_pdf):
        """Should create markdown file in correct directory."""
        mock_marker.return_value = "Content"

        result = pdf_handler.convert_pdf_to_markdown(sample_pdf)

        assert result.parent == pdf_handler.markdown_dir
        assert result.suffix == ".md"
        assert result.exists()

    @patch("tensortruth.pdf_handler.convert_with_marker")
    @patch("tensortruth.pdf_handler.convert_pdf_to_markdown")
    def test_raises_on_both_failures(
        self, mock_pymupdf, mock_marker, pdf_handler, sample_pdf
    ):
        """Should raise exception if both converters fail."""
        mock_marker.side_effect = Exception("Marker failed")
        mock_pymupdf.side_effect = Exception("PyMuPDF failed")

        with pytest.raises(Exception, match="PDF conversion failed"):
            pdf_handler.convert_pdf_to_markdown(sample_pdf, use_marker=False)


class TestDeletePDF:
    """Test PDF deletion."""

    def test_deletes_pdf_file(self, pdf_handler, sample_content, sample_filename):
        """Should delete PDF file from disk."""
        metadata = pdf_handler.upload_pdf(sample_content, sample_filename)
        pdf_id = metadata["id"]
        pdf_path = metadata["path"]

        assert pdf_path.exists()
        pdf_handler.delete_pdf(pdf_id)
        assert not pdf_path.exists()

    @patch.object(PDFHandler, "get_pdf_metadata")
    @patch("tensortruth.pdf_handler.convert_with_marker")
    def test_deletes_markdown_file(
        self,
        mock_marker,
        mock_get_metadata,
        pdf_handler,
        sample_content,
        sample_filename,
    ):
        """Should delete corresponding markdown file."""
        mock_marker.return_value = "Content"
        mock_get_metadata.return_value = {"file_size": 100, "page_count": 1}

        # Upload and convert
        metadata = pdf_handler.upload_pdf(sample_content, sample_filename)
        pdf_id = metadata["id"]
        md_path = pdf_handler.convert_pdf_to_markdown(metadata["path"])

        assert md_path.exists()
        pdf_handler.delete_pdf(pdf_id)
        assert not md_path.exists()

    def test_handles_nonexistent_pdf(self, pdf_handler):
        """Should not crash if PDF doesn't exist."""
        # Should not raise
        pdf_handler.delete_pdf("pdf_nonexistent")


class TestHelperMethods:
    """Test utility methods."""

    @patch("tensortruth.pdf_handler.convert_with_marker")
    def test_get_all_markdown_files(self, mock_marker, pdf_handler, sample_pdf):
        """Should list all markdown files in session."""
        mock_marker.return_value = "Content"

        # Create multiple markdown files
        pdf_handler.convert_pdf_to_markdown(sample_pdf)

        # Create another manually
        (pdf_handler.markdown_dir / "another.md").write_text("test")

        files = pdf_handler.get_all_markdown_files()
        assert len(files) == 2
        assert all(f.suffix == ".md" for f in files)

    def test_get_pdf_count(self, pdf_handler, sample_content, sample_filename):
        """Should count PDFs in session."""
        assert pdf_handler.get_pdf_count() == 0

        pdf_handler.upload_pdf(sample_content, sample_filename)
        assert pdf_handler.get_pdf_count() == 1

        # Upload another
        pdf_handler.upload_pdf(sample_content, "another.pdf")
        assert pdf_handler.get_pdf_count() == 2
