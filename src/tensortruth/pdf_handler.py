"""PDF upload handler for session-scoped document ingestion."""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf as fitz

from .utils.pdf import convert_pdf_to_markdown, convert_with_marker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFHandler:
    """Manages PDF upload, conversion, and storage for a session."""

    def __init__(self, session_dir: Path):
        """
        Initialize PDF handler for a session.

        Args:
            session_dir: Path to session directory (e.g., ~/.tensortruth/sessions/sess_123/)
        """
        self.session_dir = Path(session_dir)
        self.pdfs_dir = self.session_dir / "pdfs"
        self.markdown_dir = self.session_dir / "markdown"

        # Ensure directories exist
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

    def upload_pdf(self, uploaded_file: Any) -> Dict[str, Any]:
        """
        Save uploaded PDF to session directory.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Dictionary with metadata: {id, path, filename, file_size, page_count}
        """
        pdf_id = f"pdf_{uuid.uuid4().hex[:8]}"
        filename = uploaded_file.name

        # Save PDF to disk
        pdf_path = self.pdfs_dir / f"{pdf_id}_{filename}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Saved PDF: {pdf_path}")

        # Extract metadata
        metadata = self.get_pdf_metadata(pdf_path)
        metadata["id"] = pdf_id
        metadata["path"] = pdf_path
        metadata["filename"] = filename

        return metadata

    def get_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with file_size and page_count
        """
        try:
            # Get file size
            file_size = pdf_path.stat().st_size

            # Get page count using pymupdf
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()

            return {"file_size": file_size, "page_count": page_count}

        except Exception as e:
            logger.error(f"Failed to extract PDF metadata from {pdf_path}: {e}")
            return {"file_size": 0, "page_count": 0}

    def convert_pdf_to_markdown(
        self, pdf_path: Path, use_marker: bool = True
    ) -> Optional[Path]:
        """
        Convert PDF to markdown using marker-pdf (with pymupdf4llm fallback).

        Args:
            pdf_path: Path to PDF file
            use_marker: If True, try marker-pdf first (better for formulas)

        Returns:
            Path to generated markdown file

        Raises:
            Exception if conversion fails with both methods
        """
        # Extract pdf_abc123 from pdf_abc123_filename.pdf
        # Handle both uploaded PDFs (pdf_{uuid}_filename) and direct PDFs (filename)
        stem_parts = pdf_path.stem.split("_")
        if len(stem_parts) >= 2 and stem_parts[0] == "pdf":
            # Uploaded PDF format: pdf_abc123_filename
            pdf_id = f"{stem_parts[0]}_{stem_parts[1]}"
        else:
            # Direct PDF or unknown format: use full stem
            pdf_id = pdf_path.stem
        md_filename = f"{pdf_id}.md"
        md_path = self.markdown_dir / md_filename

        logger.info(f"Converting PDF to markdown: {pdf_path.name}")

        try:
            if use_marker:
                # Try marker-pdf first (better for academic papers with math)
                logger.info("Attempting conversion with marker-pdf...")
                markdown_text = convert_with_marker(str(pdf_path))
            else:
                # Use pymupdf4llm
                logger.info("Using pymupdf4llm for conversion...")
                markdown_text = convert_pdf_to_markdown(
                    str(pdf_path), preserve_math=True, converter="pymupdf"
                )

            # Check if conversion failed
            if not markdown_text or "PDF content extraction failed" in markdown_text:
                raise ValueError("PDF conversion returned empty or error content")

            # Add metadata header
            pdf_name = pdf_path.name
            header = f"# Document: {pdf_name}\n# Source: Session Upload\n\n---\n\n"
            markdown_text = header + markdown_text

            # Write to file
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            logger.info(f"Markdown saved to: {md_path}")
            return md_path

        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            # If marker failed and use_marker was True, try pymupdf as fallback
            if use_marker:
                logger.info("Marker failed, falling back to pymupdf4llm...")
                return self.convert_pdf_to_markdown(pdf_path, use_marker=False)
            else:
                raise Exception(f"PDF conversion failed with both methods: {e}")

    def delete_pdf(self, pdf_id: str) -> None:
        """
        Delete PDF and its corresponding markdown file.

        Args:
            pdf_id: PDF identifier (e.g., "pdf_abc123")
        """
        # Find and delete PDF file
        pdf_files = list(self.pdfs_dir.glob(f"{pdf_id}_*"))
        for pdf_file in pdf_files:
            pdf_file.unlink()
            logger.info(f"Deleted PDF: {pdf_file}")

        # Delete markdown file
        md_file = self.markdown_dir / f"{pdf_id}.md"
        if md_file.exists():
            md_file.unlink()
            logger.info(f"Deleted markdown: {md_file}")

    def get_all_markdown_files(self) -> List[Path]:
        """Get list of all markdown files in session."""
        return list(self.markdown_dir.glob("*.md"))

    def get_all_pdf_files(self) -> List[Path]:
        """Get list of all PDF files in session."""
        return list(self.pdfs_dir.glob("pdf_*.pdf"))

    def get_pdf_count(self) -> int:
        """Get number of PDFs in session."""
        return len(self.get_all_pdf_files())
