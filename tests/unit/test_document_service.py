"""Unit tests for DocumentService."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.services.document_service import DocumentService
from tensortruth.utils import progress as progress_mod


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


class TestUploadUrl:
    """Test URL document upload."""

    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    def test_upload_url_creates_markdown_file(
        self, mock_fetch, session_service, temp_scope_dir
    ):
        """Should create a markdown file in the markdown directory."""
        mock_fetch.return_value = (
            "# Hello\n\nSome markdown content here.",
            "Test Page",
        )

        result = session_service.upload_url("https://example.com/docs")

        md_path = Path(result.path)
        assert md_path.exists()
        assert md_path.parent == temp_scope_dir / "markdown"
        assert md_path.suffix == ".md"

        file_content = md_path.read_text()
        assert "Some markdown content here." in file_content

    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    def test_upload_url_header_contains_url(self, mock_fetch, session_service):
        """Should include the source URL in the file header."""
        mock_fetch.return_value = ("# Content\n\nBody text.", "Page Title")

        result = session_service.upload_url("https://pytorch.org/docs/stable")

        file_content = Path(result.path).read_text()
        assert "# URL: https://pytorch.org/docs/stable" in file_content
        assert "# Document: Page Title" in file_content

    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    def test_upload_url_includes_context(self, mock_fetch, session_service):
        """Should prepend context before fetched content when provided."""
        mock_fetch.return_value = ("# Fetched\n\nFetched body.", "Title")

        result = session_service.upload_url(
            "https://example.com/page", context="Focus on section 3"
        )

        file_content = Path(result.path).read_text()
        assert "Focus on section 3" in file_content
        assert "Fetched body." in file_content
        # Context should appear before fetched content
        context_pos = file_content.index("Focus on section 3")
        body_pos = file_content.index("Fetched body.")
        assert context_pos < body_pos

    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    def test_upload_url_returns_metadata(self, mock_fetch, session_service):
        """Should return proper PDFMetadata fields."""
        mock_fetch.return_value = ("# Content\n\nBody.", "My Page Title")

        result = session_service.upload_url("https://example.com/page")

        assert result.pdf_id.startswith("url_")
        assert result.filename == "My Page Title"
        assert result.page_count == 0
        assert result.file_size > 0
        assert Path(result.path).exists()

    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    def test_upload_url_returns_domain_as_filename_when_no_title(
        self, mock_fetch, session_service
    ):
        """Should use domain as filename when page has no title."""
        mock_fetch.return_value = ("# Content\n\nBody.", "")

        result = session_service.upload_url("https://example.com/page")

        assert result.filename == "example.com"

    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    def test_upload_url_propagates_fetch_error(self, mock_fetch, session_service):
        """Should propagate ConnectionError from fetch_url_as_markdown."""
        mock_fetch.side_effect = ConnectionError("HTTP 404 fetching URL")

        with pytest.raises(ConnectionError, match="HTTP 404"):
            session_service.upload_url("https://example.com/missing")

    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    def test_upload_url_scope_labels_session(self, mock_fetch, session_service):
        """Should use 'Session Upload' label for session scope."""
        mock_fetch.return_value = ("# Content\n\nBody.", "Title")

        result = session_service.upload_url("https://example.com")

        file_content = Path(result.path).read_text()
        assert "# Source: Session Upload" in file_content

    @patch("tensortruth.services.document_service.fetch_url_as_markdown")
    def test_upload_url_scope_labels_project(self, mock_fetch, project_service):
        """Should use 'Project Upload' label for project scope."""
        mock_fetch.return_value = ("# Content\n\nBody.", "Title")

        result = project_service.upload_url("https://example.com")

        file_content = Path(result.path).read_text()
        assert "# Source: Project Upload" in file_content


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


class TestRebuildIndexProgress:
    """Regression tests for progress reporting during long build stages.

    A two-PDF rebuild used to report ``Converting a.pdf (1/2)`` at 0% and then
    nothing for the entire Marker conversion (minutes), which the UI renders as
    a frozen build.
    """

    @pytest.fixture(autouse=True)
    def fast_heartbeat(self, monkeypatch):
        monkeypatch.setattr(progress_mod, "DEFAULT_HEARTBEAT_INTERVAL", 0.02)

    def _service_with_two_pdfs(self, session_service, tmp_path, monkeypatch):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        pdf_a = docs_dir / "pdf_aaa_Paper A.pdf"
        pdf_b = docs_dir / "pdf_bbb_Paper B.pdf"
        pdf_a.write_bytes(b"%PDF-1.4")
        pdf_b.write_bytes(b"%PDF-1.4")

        md_path = tmp_path / "markdown" / "pdf_aaa.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("# Paper A")

        monkeypatch.setattr(
            session_service, "get_all_pdf_files", lambda: [pdf_a, pdf_b]
        )
        monkeypatch.setattr(session_service, "get_all_markdown_files", lambda: [])
        monkeypatch.setattr(
            session_service, "index_from_markdown", lambda *a, **k: None
        )
        monkeypatch.setattr(session_service, "_persist_metadata", lambda: None)
        monkeypatch.setattr(session_service, "_get_index_builder", lambda: MagicMock())
        return pdf_a, pdf_b, md_path

    def test_progress_advances_while_conversion_is_blocked(
        self, session_service, tmp_path, monkeypatch
    ):
        """Progress must move during a conversion, not only after it."""
        pdf_a, _pdf_b, md_path = self._service_with_two_pdfs(
            session_service, tmp_path, monkeypatch
        )

        started = threading.Event()
        release = threading.Event()

        def blocking_convert(pdf):
            if pdf.name == pdf_a.name:
                started.set()
                release.wait(timeout=10)
            return md_path

        monkeypatch.setattr(session_service, "_convert_if_needed", blocking_convert)

        calls = []

        def cb(stage, current, total):
            calls.append((stage, current, total))

        thread = threading.Thread(
            target=lambda: session_service.rebuild_index(None, cb), daemon=True
        )
        thread.start()

        assert started.wait(timeout=5), "conversion never started"

        # While the first conversion is still blocked, the heartbeat must have
        # reported something other than the initial 0%.
        deadline = time.monotonic() + 5
        observed = []
        snapshot = []
        while time.monotonic() < deadline:
            snapshot = list(calls)
            observed = [c[1] for c in snapshot]
            if observed and max(observed) >= 2:
                break
            time.sleep(0.02)

        assert observed, "no progress reported at all"
        assert (
            max(observed) >= 2
        ), f"progress never advanced during a blocked conversion: {observed}"
        assert (
            "Converting" in snapshot[-1][0]
        ), f"last stage was not the conversion: {snapshot[-1][0]}"

        release.set()
        thread.join(timeout=10)
        assert not thread.is_alive(), "rebuild did not finish"

        pcts = [c[1] for c in calls]
        assert pcts[-1] == 100, f"build did not report completion: {pcts[-5:]}"
        assert pcts == sorted(pcts), f"progress went backwards: {pcts}"

    def test_conversion_progress_stays_inside_its_document_slice(
        self, session_service, tmp_path, monkeypatch
    ):
        """The first of two documents owns 0-60% of the 0-60% conversion band."""
        pdf_a, pdf_b, md_path = self._service_with_two_pdfs(
            session_service, tmp_path, monkeypatch
        )

        seen = []

        def slow_convert(pdf):
            if pdf.name == pdf_a.name:
                time.sleep(0.2)
            return md_path

        monkeypatch.setattr(session_service, "_convert_if_needed", slow_convert)
        monkeypatch.setattr(
            session_service,
            "index_from_markdown",
            lambda *a, **k: seen.append(("index",)),
        )

        calls = []
        session_service.rebuild_index(
            None, lambda stage, current, total: calls.append((stage, current))
        )

        assert calls[-1][1] == 100
        # Both documents were converted before indexing started.
        assert any("Paper A" in c[0] for c in calls)
        assert any("Paper B" in c[0] for c in calls)
        assert seen == [("index",)]
        assert pdf_b.exists()
