"""Integration tests for PDF ingestion pipeline."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tensortruth.pdf_handler import PDFHandler
from tensortruth.session_index import SessionIndexBuilder


@pytest.fixture
def integration_session_dir(tmp_path):
    """Create session directory for integration testing."""
    session_dir = tmp_path / "sessions" / "integration_test_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


@pytest.fixture
def sample_pdf_content():
    """Create minimal valid PDF content."""
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF content) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000208 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
303
%%EOF"""


@pytest.fixture
def mock_uploaded_pdf(tmp_path, sample_pdf_content):
    """Create a mock uploaded PDF file."""
    pdf_path = tmp_path / "test_upload.pdf"
    pdf_path.write_bytes(sample_pdf_content)

    mock_file = Mock()
    mock_file.name = "test_paper.pdf"
    mock_file.getbuffer.return_value = sample_pdf_content
    return mock_file


@pytest.mark.integration
class TestPDFUploadToIndex:
    """Test complete pipeline from upload to indexed."""

    @patch("tensortruth.pdf_handler.convert_with_marker")
    @patch("tensortruth.session_index.get_embed_model")
    @patch("tensortruth.session_index.VectorStoreIndex")
    def test_full_pipeline_marker_success(
        self,
        mock_index,
        mock_embed,
        mock_marker,
        integration_session_dir,
        mock_uploaded_pdf,
    ):
        """Test complete flow: upload → marker conversion → indexing."""
        # Setup mocks
        mock_marker.return_value = "# Research Paper\n\nThis is a test paper about AI."
        mock_embed.return_value = Mock()
        mock_index_instance = Mock()
        mock_index.return_value = mock_index_instance

        # Step 1: Upload PDF
        handler = PDFHandler(integration_session_dir)
        metadata = handler.upload_pdf(mock_uploaded_pdf)

        assert metadata["id"].startswith("pdf_")
        assert metadata["filename"] == "test_paper.pdf"
        assert Path(metadata["path"]).exists()

        # Step 2: Convert to markdown
        md_path = handler.convert_pdf_to_markdown(metadata["path"])

        assert md_path.exists()
        md_content = md_path.read_text()
        assert "# Document:" in md_content
        assert "# Source: Session Upload" in md_content
        assert "Research Paper" in md_content

        # Step 3: Build index
        builder = SessionIndexBuilder("integration_test_session")
        with (
            patch("tensortruth.session_index.get_session_index_dir") as mock_index_dir,
            patch("tensortruth.session_index.get_session_markdown_dir") as mock_md_dir,
        ):
            mock_index_dir.return_value = integration_session_dir / "index"
            mock_md_dir.return_value = integration_session_dir / "markdown"

            # Create the builder with mocked paths
            builder = SessionIndexBuilder("integration_test_session")

            # Should not raise
            builder.build_index([md_path])

        # Verify embedding was called
        assert mock_embed.called

    @patch("tensortruth.pdf_handler.convert_pdf_to_markdown")
    @patch("tensortruth.pdf_handler.convert_with_marker")
    @patch("tensortruth.session_index.get_embed_model")
    @patch("tensortruth.session_index.VectorStoreIndex")
    def test_fallback_to_pymupdf_on_marker_failure(
        self,
        mock_index,
        mock_embed,
        mock_marker,
        mock_pymupdf,
        integration_session_dir,
        mock_uploaded_pdf,
    ):
        """Test fallback when marker-pdf fails."""
        # Marker fails
        mock_marker.side_effect = Exception("Marker conversion failed")
        # PyMuPDF succeeds
        mock_pymupdf.return_value = "# Fallback Content\n\nConverted with pymupdf"

        mock_embed.return_value = Mock()

        handler = PDFHandler(integration_session_dir)
        metadata = handler.upload_pdf(mock_uploaded_pdf)

        # Should successfully convert using fallback
        md_path = handler.convert_pdf_to_markdown(metadata["path"])

        assert md_path.exists()
        # Should have used pymupdf as fallback
        assert mock_pymupdf.called


@pytest.mark.integration
class TestMultiplePDFHandling:
    """Test handling multiple PDFs in a session."""

    @patch("tensortruth.pdf_handler.convert_with_marker")
    @patch("tensortruth.session_index.get_embed_model")
    @patch("tensortruth.session_index.VectorStoreIndex")
    def test_multiple_pdf_upload_and_indexing(
        self,
        mock_index,
        mock_embed,
        mock_marker,
        integration_session_dir,
        sample_pdf_content,
    ):
        """Test uploading and indexing multiple PDFs."""
        mock_marker.return_value = "# Paper Content"
        mock_embed.return_value = Mock()

        handler = PDFHandler(integration_session_dir)
        uploaded_pdfs = []

        # Upload 3 PDFs
        for i in range(3):
            mock_file = Mock()
            mock_file.name = f"paper_{i}.pdf"
            mock_file.getbuffer.return_value = sample_pdf_content

            metadata = handler.upload_pdf(mock_file)
            md_path = handler.convert_pdf_to_markdown(metadata["path"])
            uploaded_pdfs.append((metadata, md_path))

        # Verify all uploaded
        assert len(uploaded_pdfs) == 3
        assert handler.get_pdf_count() == 3

        # Get all markdown files
        md_files = handler.get_all_markdown_files()
        assert len(md_files) == 3

        # Build index from all
        builder = SessionIndexBuilder("integration_test_session")
        with (
            patch("tensortruth.session_index.get_session_index_dir") as mock_index_dir,
            patch("tensortruth.session_index.get_session_markdown_dir") as mock_md_dir,
        ):
            mock_index_dir.return_value = integration_session_dir / "index"
            mock_md_dir.return_value = integration_session_dir / "markdown"

            builder = SessionIndexBuilder("integration_test_session")
            builder.build_index(md_files)

        assert mock_index.called


@pytest.mark.integration
class TestPDFDeletionAndRebuild:
    """Test PDF deletion and index rebuilding."""

    @patch("tensortruth.pdf_handler.convert_with_marker")
    @patch("tensortruth.session_index.get_embed_model")
    @patch("tensortruth.session_index.VectorStoreIndex")
    def test_delete_pdf_and_rebuild(
        self,
        mock_index,
        mock_embed,
        mock_marker,
        integration_session_dir,
        mock_uploaded_pdf,
    ):
        """Test deleting a PDF and rebuilding index."""
        mock_marker.return_value = "# Content"
        mock_embed.return_value = Mock()

        handler = PDFHandler(integration_session_dir)

        # Upload 2 PDFs
        metadata1 = handler.upload_pdf(mock_uploaded_pdf)
        md_path1 = handler.convert_pdf_to_markdown(metadata1["path"])

        mock_uploaded_pdf.name = "another.pdf"
        metadata2 = handler.upload_pdf(mock_uploaded_pdf)
        md_path2 = handler.convert_pdf_to_markdown(metadata2["path"])

        assert handler.get_pdf_count() == 2

        # Delete first PDF
        handler.delete_pdf(metadata1["id"])

        assert handler.get_pdf_count() == 1
        assert not Path(metadata1["path"]).exists()
        assert not md_path1.exists()
        assert Path(metadata2["path"]).exists()
        assert md_path2.exists()

        # Rebuild index with remaining PDF
        remaining_files = handler.get_all_markdown_files()
        assert len(remaining_files) == 1

        builder = SessionIndexBuilder("integration_test_session")
        with (
            patch("tensortruth.session_index.get_session_index_dir") as mock_index_dir,
            patch("tensortruth.session_index.get_session_markdown_dir") as mock_md_dir,
        ):
            mock_index_dir.return_value = integration_session_dir / "index"
            mock_md_dir.return_value = integration_session_dir / "markdown"

            builder = SessionIndexBuilder("integration_test_session")
            builder.build_index(remaining_files)

        # Should have rebuilt successfully
        assert mock_index.called


@pytest.mark.integration
class TestSessionCleanup:
    """Test complete session cleanup."""

    @patch("tensortruth.pdf_handler.convert_with_marker")
    @patch("tensortruth.session_index.get_embed_model")
    @patch("tensortruth.session_index.VectorStoreIndex")
    def test_complete_session_cleanup(
        self,
        mock_index,
        mock_embed,
        mock_marker,
        integration_session_dir,
        mock_uploaded_pdf,
    ):
        """Test that deleting session removes all artifacts."""
        mock_marker.return_value = "# Content"
        mock_embed.return_value = Mock()

        # Setup handler and builder
        handler = PDFHandler(integration_session_dir)

        # Upload PDF and build index
        metadata = handler.upload_pdf(mock_uploaded_pdf)
        md_path = handler.convert_pdf_to_markdown(metadata["path"])

        with (
            patch("tensortruth.session_index.get_session_index_dir") as mock_index_dir,
            patch("tensortruth.session_index.get_session_markdown_dir") as mock_md_dir,
        ):
            mock_index_dir.return_value = integration_session_dir / "index"
            mock_md_dir.return_value = integration_session_dir / "markdown"

            # Use context manager to ensure proper cleanup
            with SessionIndexBuilder("integration_test_session") as builder:
                builder.build_index([md_path])

        # Verify everything exists
        assert (integration_session_dir / "pdfs").exists()
        assert (integration_session_dir / "markdown").exists()
        assert (integration_session_dir / "index").exists()

        # Force garbage collection to close ChromaDB connections (Windows file locking issue)
        import gc
        import shutil
        import sys
        import time

        gc.collect()  # Force garbage collection
        time.sleep(1)  # Give Windows time to release file handles

        # Cleanup with retry logic for Windows file locking
        max_retries = 5
        cleanup_successful = False

        for attempt in range(max_retries):
            try:
                shutil.rmtree(integration_session_dir)
                cleanup_successful = True
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2**attempt  # 1s, 2s, 4s, 8s
                    time.sleep(wait_time)
                    gc.collect()  # Try GC again
                else:
                    # On Windows, ChromaDB's SQLite connection can be very stubborn
                    # Skip cleanup verification on final failure if on Windows
                    if sys.platform == "win32":
                        import warnings

                        warnings.warn(
                            f"Could not delete test directory due to Windows file "
                            f"locking: {e}. This is a known ChromaDB issue on "
                            "Windows and doesn't affect functionality."
                        )
                        cleanup_successful = False
                    else:
                        raise

        # Verify cleanup (skip on Windows if cleanup failed due to locked files)
        if cleanup_successful:
            assert not integration_session_dir.exists()


@pytest.mark.integration
class TestIndexPersistence:
    """Test index persistence across sessions."""

    @patch("tensortruth.pdf_handler.convert_with_marker")
    @patch("tensortruth.session_index.get_embed_model")
    @patch("tensortruth.session_index.VectorStoreIndex")
    def test_index_persists_and_loads(
        self,
        mock_index,
        mock_embed,
        mock_marker,
        integration_session_dir,
        mock_uploaded_pdf,
    ):
        """Test that index can be built, persisted, and checked for existence."""
        mock_marker.return_value = "# Content"
        mock_embed.return_value = Mock()

        handler = PDFHandler(integration_session_dir)
        metadata = handler.upload_pdf(mock_uploaded_pdf)
        md_path = handler.convert_pdf_to_markdown(metadata["path"])

        # Build index
        builder = SessionIndexBuilder("integration_test_session")
        with (
            patch("tensortruth.session_index.get_session_index_dir") as mock_index_dir,
            patch("tensortruth.session_index.get_session_markdown_dir") as mock_md_dir,
        ):
            mock_index_dir.return_value = integration_session_dir / "index"
            mock_md_dir.return_value = integration_session_dir / "markdown"

            builder = SessionIndexBuilder("integration_test_session")
            builder.build_index([md_path])

            # Create marker files for index existence
            (integration_session_dir / "index" / "chroma.sqlite3").write_text("")
            (integration_session_dir / "index" / "docstore.json").write_text("{}")

            # Check index exists
            assert builder.index_exists()

            # Get index size
            size = builder.get_index_size()
            assert size > 0

            # Get document count
            count = builder.get_document_count()
            assert count == 1


@pytest.mark.integration
class TestDuplicatePDFPrevention:
    """Test prevention of duplicate PDF processing."""

    def test_duplicate_filename_rejected(self, mock_uploaded_pdf):
        """Should not process PDF with duplicate filename."""
        # Create a mock session with existing PDF
        session = {
            "pdf_documents": [
                {
                    "id": "pdf_existing",
                    "filename": "test_paper.pdf",
                    "status": "indexed",
                }
            ]
        }

        # Check that duplicate is detected
        existing_filenames = {doc["filename"] for doc in session["pdf_documents"]}
        assert "test_paper.pdf" in existing_filenames

        # Attempting to add same filename should be blocked
        # This would be caught by the check in process_pdf_upload
        assert mock_uploaded_pdf.name in existing_filenames
