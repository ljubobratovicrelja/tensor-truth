"""PDF handling service for session-scoped document management.

This service wraps the PDFHandler and SessionIndexBuilder with a unified
interface for PDF upload, conversion, indexing, and cleanup.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from tensortruth.pdf_handler import PDFHandler
from tensortruth.session_index import SessionIndexBuilder

from .models import PDFMetadata


class PDFService:
    """Service for managing PDF uploads and indexing within a session.

    Provides a clean interface for:
    - Uploading PDFs to session storage
    - Converting PDFs to markdown
    - Building vector indexes from PDFs
    - Cleanup of session PDF data
    """

    def __init__(
        self,
        session_id: str,
        session_dir: Union[str, Path],
        metadata_cache: Optional[Dict[str, Dict]] = None,
    ):
        """Initialize PDF service for a session.

        Args:
            session_id: Session identifier.
            session_dir: Path to session directory.
            metadata_cache: Optional pre-loaded metadata cache from session JSON.
        """
        self.session_id = session_id
        self.session_dir = Path(session_dir)
        self.metadata_cache = metadata_cache or {}

        # Initialize handlers
        self._pdf_handler = PDFHandler(self.session_dir)
        self._index_builder: Optional[SessionIndexBuilder] = None

    def _get_index_builder(self) -> SessionIndexBuilder:
        """Get or create the index builder instance.

        Returns:
            SessionIndexBuilder for this session.
        """
        if self._index_builder is None:
            self._index_builder = SessionIndexBuilder(
                self.session_id, metadata_cache=self.metadata_cache
            )
        return self._index_builder

    def upload(self, file_content: bytes, filename: str) -> PDFMetadata:
        """Upload a PDF file to the session.

        Args:
            file_content: Raw PDF file bytes.
            filename: Original filename.

        Returns:
            PDFMetadata with file info.
        """
        result = self._pdf_handler.upload_pdf(file_content, filename)

        return PDFMetadata(
            pdf_id=result["id"],
            filename=result["filename"],
            path=str(result["path"]),
            file_size=result["file_size"],
            page_count=result["page_count"],
        )

    def convert_to_markdown(
        self, pdf_path: Union[str, Path], use_marker: bool = True
    ) -> Optional[Path]:
        """Convert a PDF to markdown.

        Args:
            pdf_path: Path to PDF file.
            use_marker: If True, try marker-pdf first (better for formulas).

        Returns:
            Path to generated markdown file, or None on failure.
        """
        return self._pdf_handler.convert_pdf_to_markdown(Path(pdf_path), use_marker)

    def index_pdfs(
        self,
        pdf_files: Optional[List[Path]] = None,
        chunk_sizes: Optional[List[int]] = None,
    ) -> None:
        """Build vector index from PDF files.

        Uses direct PDF indexing (faster) rather than markdown conversion.

        Args:
            pdf_files: List of PDF paths. If None, indexes all PDFs in session.
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256]).

        Raises:
            ValueError: If no PDF files provided or found.
        """
        if pdf_files is None:
            pdf_files = self.get_all_pdf_files()

        if not pdf_files:
            raise ValueError("No PDF files to index")

        builder = self._get_index_builder()
        builder.build_index_from_pdfs(pdf_files, chunk_sizes)

        # Update our metadata cache from builder
        self.metadata_cache = builder.get_metadata_cache()

    def index_from_markdown(
        self,
        markdown_files: Optional[List[Path]] = None,
        chunk_sizes: Optional[List[int]] = None,
    ) -> None:
        """Build vector index from markdown files.

        Alternative indexing method using pre-converted markdown.

        Args:
            markdown_files: List of markdown paths. If None, uses all in session.
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256]).
        """
        builder = self._get_index_builder()
        builder.build_index(markdown_files, chunk_sizes)
        self.metadata_cache = builder.get_metadata_cache()

    def delete(self, pdf_id: str) -> None:
        """Delete a PDF and its associated files.

        Args:
            pdf_id: PDF identifier (e.g., "pdf_abc123").
        """
        self._pdf_handler.delete_pdf(pdf_id)

        # Remove from metadata cache
        if pdf_id in self.metadata_cache:
            del self.metadata_cache[pdf_id]

    def get_index_path(self) -> Optional[Path]:
        """Get the path to the session's vector index.

        Returns:
            Path to index directory, or None if no index exists.
        """
        builder = self._get_index_builder()
        if builder.index_exists():
            return builder.session_index_dir
        return None

    def index_exists(self) -> bool:
        """Check if a valid index exists for this session.

        Returns:
            True if index exists and is valid.
        """
        builder = self._get_index_builder()
        return builder.index_exists()

    def get_all_pdf_files(self) -> List[Path]:
        """Get all PDF files in the session.

        Returns:
            List of PDF file paths.
        """
        return self._pdf_handler.get_all_pdf_files()

    def get_all_markdown_files(self) -> List[Path]:
        """Get all markdown files in the session.

        Returns:
            List of markdown file paths.
        """
        return self._pdf_handler.get_all_markdown_files()

    def get_pdf_count(self) -> int:
        """Get the number of PDFs in the session.

        Returns:
            PDF count.
        """
        return self._pdf_handler.get_pdf_count()

    def get_metadata_cache(self) -> Dict[str, Dict]:
        """Get the metadata cache for session persistence.

        Returns:
            Metadata cache dict.
        """
        return self.metadata_cache

    def cleanup(self) -> None:
        """Clean up resources and close connections.

        Should be called when done with the session.
        """
        if self._index_builder is not None:
            self._index_builder.close()
            self._index_builder = None

    def delete_index(self) -> None:
        """Delete the session's vector index.

        Removes all indexed data but keeps the original PDFs.
        """
        builder = self._get_index_builder()
        builder.delete_index()

    def rebuild_index(self, chunk_sizes: Optional[List[int]] = None) -> None:
        """Rebuild the index from all PDFs in the session.

        Args:
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256]).
        """
        pdf_files = self.get_all_pdf_files()
        if pdf_files:
            self.index_pdfs(pdf_files, chunk_sizes)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()
        return False
