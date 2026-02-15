"""Document handling service for scope-based document management.

This service wraps the PDFHandler and DocumentIndexBuilder with a unified
interface for document upload, conversion, indexing, and cleanup.
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

from tensortruth.document_index import DocumentIndexBuilder
from tensortruth.pdf_handler import PDFHandler
from tensortruth.scrapers.url_fetcher import fetch_url_as_markdown

from .metadata_store import MetadataStore
from .models import PDFMetadata

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing document uploads and indexing within a scope.

    Provides a clean interface for:
    - Uploading documents (PDF, text, markdown) to scope storage
    - Converting PDFs to markdown
    - Building vector indexes from documents
    - Cleanup of scope document data
    """

    def __init__(
        self,
        scope_id: str,
        scope_dir: Union[str, Path],
        scope_type: str = "session",
        metadata_cache: Optional[Dict[str, Dict]] = None,
        metadata_store: Optional[MetadataStore] = None,
    ):
        """Initialize document service for a scope.

        Args:
            scope_id: Scope identifier (session or project ID).
            scope_dir: Path to scope directory.
            scope_type: Scope type ("session" or "project").
            metadata_cache: Optional pre-loaded metadata cache from scope JSON.
            metadata_store: Optional persistent metadata store.
        """
        self.scope_id = scope_id
        self.scope_dir = Path(scope_dir)
        self.scope_type = scope_type
        self.metadata_cache = metadata_cache or {}
        self._metadata_store = metadata_store

        self._pdf_handler = PDFHandler(self.scope_dir, scope_type=self.scope_type)
        self._index_builder: Optional[DocumentIndexBuilder] = None

    def _get_index_builder(self) -> DocumentIndexBuilder:
        """Get or create the index builder instance.

        Returns:
            DocumentIndexBuilder for this scope.
        """
        if self._index_builder is None:
            index_dir = self.scope_dir / "index"
            markdown_dir = self.scope_dir / "markdown"
            self._index_builder = DocumentIndexBuilder(
                index_dir=index_dir,
                markdown_dir=markdown_dir,
                metadata_cache=self.metadata_cache,
            )
        return self._index_builder

    def upload(self, file_content: bytes, filename: str) -> PDFMetadata:
        """Upload a PDF file to the scope.

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
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Build vector index from PDF files.

        Uses direct PDF indexing (faster) rather than markdown conversion.

        Args:
            pdf_files: List of PDF paths. If None, indexes all PDFs in scope.
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256]).
            progress_callback: Optional callback(stage, current, total) for progress.

        Raises:
            ValueError: If no PDF files provided or found.
        """
        if pdf_files is None:
            pdf_files = self.get_all_pdf_files()

        if not pdf_files:
            raise ValueError("No PDF files to index")

        builder = self._get_index_builder()
        builder.build_index_from_pdfs(pdf_files, chunk_sizes, progress_callback)

        # Update our metadata cache from builder and persist
        self.metadata_cache = builder.get_metadata_cache()
        self._persist_metadata()

    def index_from_markdown(
        self,
        markdown_files: Optional[List[Path]] = None,
        chunk_sizes: Optional[List[int]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Build vector index from markdown files.

        Alternative indexing method using pre-converted markdown.

        Args:
            markdown_files: List of markdown paths. If None, uses all in scope.
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256]).
            progress_callback: Optional callback(stage, current, total) for progress.
        """
        builder = self._get_index_builder()
        builder.build_index(markdown_files, chunk_sizes, progress_callback)
        self.metadata_cache = builder.get_metadata_cache()
        self._persist_metadata()

    def delete(self, pdf_id: str) -> None:
        """Delete a PDF and its associated files.

        Also removes the document from the vector index if it exists.

        Args:
            pdf_id: PDF identifier (e.g., "pdf_abc123").
        """
        self._pdf_handler.delete_pdf(pdf_id)

        # Remove from vector index if present
        builder = self._get_index_builder()
        if builder.index_exists():
            removed = builder.remove_document(pdf_id)
            if removed:
                logger.info(f"Removed {pdf_id} from vector index")

        # Remove from metadata cache and store
        if pdf_id in self.metadata_cache:
            del self.metadata_cache[pdf_id]
        if self._metadata_store is not None:
            self._metadata_store.delete(pdf_id)
            self._metadata_store.persist_if_dirty()

    def get_index_path(self) -> Optional[Path]:
        """Get the path to the scope's vector index.

        Returns:
            Path to index directory, or None if no index exists.
        """
        builder = self._get_index_builder()
        if builder.index_exists():
            return builder.index_dir
        return None

    def index_exists(self) -> bool:
        """Check if a valid index exists for this scope.

        Returns:
            True if index exists and is valid.
        """
        builder = self._get_index_builder()
        return builder.index_exists()

    def get_all_pdf_files(self) -> List[Path]:
        """Get all PDF files in the scope.

        Returns:
            List of PDF file paths.
        """
        return self._pdf_handler.get_all_pdf_files()

    def get_all_markdown_files(self) -> List[Path]:
        """Get all markdown files in the scope.

        Returns:
            List of markdown file paths.
        """
        return self._pdf_handler.get_all_markdown_files()

    def get_pdf_count(self) -> int:
        """Get the number of PDFs in the scope.

        Returns:
            PDF count.
        """
        return self._pdf_handler.get_pdf_count()

    def set_metadata(self, doc_id: str, metadata: Dict) -> None:
        """Set metadata for a document and persist to disk.

        Used at upload time to inject metadata (e.g. arXiv title/authors)
        so the indexer can skip LLM extraction.
        """
        self.metadata_cache[doc_id] = metadata
        if self._metadata_store is not None:
            self._metadata_store.set(doc_id, metadata)
            self._metadata_store.save()

    def _persist_metadata(self) -> None:
        """Persist current metadata cache to disk via the store."""
        if self._metadata_store is not None:
            self._metadata_store.update_from(self.metadata_cache)
            self._metadata_store.persist_if_dirty()

    def get_metadata_cache(self) -> Dict[str, Dict]:
        """Get the metadata cache for scope persistence.

        Returns:
            Metadata cache dict.
        """
        return self.metadata_cache

    def cleanup(self) -> None:
        """Clean up resources and close connections.

        Should be called when done with the scope.
        """
        self._persist_metadata()
        if self._index_builder is not None:
            self._index_builder.close()
            self._index_builder = None

    def delete_index(self) -> None:
        """Delete the scope's vector index.

        Removes all indexed data but keeps the original documents.
        """
        builder = self._get_index_builder()
        builder.delete_index()

    def _convert_if_needed(self, pdf_path: Path) -> Optional[Path]:
        """Convert a PDF to markdown if not already converted.

        Checks if a markdown file with the matching pdf_id already
        exists in the markdown directory. Skips conversion if so.
        """
        # Extract pdf_id the same way pdf_handler does:
        # "pdf_abc1234_My Paper.pdf" -> "pdf_abc1234"
        stem_parts = pdf_path.stem.split("_")
        if len(stem_parts) >= 2 and stem_parts[0] == "pdf":
            pdf_id = f"{stem_parts[0]}_{stem_parts[1]}"
        else:
            pdf_id = pdf_path.stem

        md_path = self.scope_dir / "markdown" / f"{pdf_id}.md"
        if md_path.exists():
            return md_path
        return self.convert_to_markdown(pdf_path)

    def _get_all_doc_ids(self) -> List[str]:
        """Collect doc_ids from all documents on disk (PDFs + standalone markdown).

        Returns:
            List of doc_id strings (e.g. ["pdf_abc123", "doc_def456"]).
        """
        from tensortruth.api.routes.documents import extract_doc_id

        pdf_files = self.get_all_pdf_files()
        md_files = self.get_all_markdown_files()

        doc_ids: List[str] = []
        pdf_doc_ids: set[str] = set()

        for pdf_path in pdf_files:
            did = extract_doc_id(pdf_path.stem)
            pdf_doc_ids.add(did)
            doc_ids.append(did)

        # Add standalone markdown files (exclude PDF conversion artifacts)
        for md_path in md_files:
            did = extract_doc_id(md_path.stem)
            if did not in pdf_doc_ids:
                doc_ids.append(did)

        return doc_ids

    def get_unindexed_doc_ids(self) -> List[str]:
        """Return doc_ids not yet in the index.

        Returns:
            List of doc_ids that have not been indexed.
        """
        all_ids = set(self._get_all_doc_ids())
        builder = self._get_index_builder()
        indexed_ids = builder.get_indexed_doc_ids()
        return sorted(all_ids - indexed_ids)

    def build_index(
        self,
        chunk_sizes: Optional[List[int]] = None,
        progress_callback: Optional[Callable] = None,
        conversion_method: str = "marker",
    ) -> None:
        """Incremental build: only index documents not yet in the index.

        If settings have changed since the last build, deletes the index
        and does a full rebuild.

        Args:
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256]).
            progress_callback: Optional callback(stage, current, total).
            conversion_method: "marker" or "direct".
        """
        from tensortruth.api.routes.documents import extract_doc_id

        if chunk_sizes is None:
            chunk_sizes = [2048, 512, 256]

        cb = progress_callback or (lambda *a: None)
        builder = self._get_index_builder()

        # If settings changed, delete index and do full rebuild
        if builder.index_exists() and not builder.is_settings_current(
            chunk_sizes, conversion_method
        ):
            logger.info("Settings changed, deleting index for full rebuild")
            builder.delete_index()

        if not builder.index_exists():
            # No existing index — full rebuild
            self.rebuild_index(chunk_sizes, progress_callback, conversion_method)
            return

        # Determine which doc_ids need indexing
        all_ids = self._get_all_doc_ids()
        indexed_ids = builder.get_indexed_doc_ids()
        unindexed = [did for did in all_ids if did not in indexed_ids]

        if not unindexed:
            logger.info("All documents already indexed, nothing to do")
            return

        logger.info(f"Incremental build: {len(unindexed)} new documents to index")

        # Gather files for unindexed doc_ids
        pdf_files = self.get_all_pdf_files()
        md_files = self.get_all_markdown_files()

        # Build a map of doc_id -> file path
        pdf_by_id: Dict[str, Path] = {}
        for pf in pdf_files:
            pdf_by_id[extract_doc_id(pf.stem)] = pf
        md_by_id: Dict[str, Path] = {}
        for mf in md_files:
            md_by_id[extract_doc_id(mf.stem)] = mf

        # Convert unindexed PDFs to markdown if needed (marker path)
        from llama_index.core import SimpleDirectoryReader

        documents = []
        doc_ids = []

        if conversion_method == "direct":
            # Direct PDF path — load PDFs directly
            from llama_index.readers.file import PDFReader

            reader = PDFReader()
            for did in unindexed:
                if did in pdf_by_id:
                    cb(f"Loading {pdf_by_id[did].name}", 10, 100)
                    docs = reader.load_data(pdf_by_id[did])
                    for doc in docs:
                        doc.metadata["file_path"] = str(pdf_by_id[did])
                        doc.metadata["file_name"] = pdf_by_id[did].name
                    documents.extend(docs)
                    doc_ids.extend([did] * len(docs))
                elif did in md_by_id:
                    rdr = SimpleDirectoryReader(input_files=[str(md_by_id[did])])
                    docs = rdr.load_data()
                    documents.extend(docs)
                    doc_ids.extend([did] * len(docs))
        else:
            # Marker path — convert PDFs to markdown first, then load markdown
            total = len(unindexed)
            for i, did in enumerate(unindexed):
                if did in pdf_by_id:
                    cb(f"Converting {pdf_by_id[did].name}", i, total + 1)
                    md_path = self._convert_if_needed(pdf_by_id[did])
                    if md_path:
                        rdr = SimpleDirectoryReader(input_files=[str(md_path)])
                        docs = rdr.load_data()
                        documents.extend(docs)
                        doc_ids.extend([did] * len(docs))
                elif did in md_by_id:
                    rdr = SimpleDirectoryReader(input_files=[str(md_by_id[did])])
                    docs = rdr.load_data()
                    documents.extend(docs)
                    doc_ids.extend([did] * len(docs))

        if not documents:
            logger.warning("No documents loaded for incremental add")
            return

        logger.info(f"Loading {len(documents)} documents for incremental add")
        builder.add_documents(documents, doc_ids, chunk_sizes, progress_callback)

        # Update metadata cache from builder and persist
        self.metadata_cache = builder.get_metadata_cache()
        self._persist_metadata()

    def rebuild_index(
        self,
        chunk_sizes: Optional[List[int]] = None,
        progress_callback: Optional[Callable] = None,
        conversion_method: str = "marker",
    ) -> None:
        """Rebuild the index from all documents in the scope.

        Args:
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256]).
            progress_callback: Optional callback(stage, current, total) for progress.
            conversion_method: "marker" (default) or "direct" for legacy PDF path.
        """
        pdf_files = self.get_all_pdf_files()
        existing_md = self.get_all_markdown_files()
        cb = progress_callback or (lambda *a: None)

        if conversion_method == "direct" or not pdf_files:
            # Legacy path or no PDFs (only markdown docs)
            if pdf_files:
                self.index_pdfs(pdf_files, chunk_sizes, progress_callback)
            elif existing_md:
                self.index_from_markdown(existing_md, chunk_sizes, progress_callback)
            # Save settings hash
            b = self._get_index_builder()
            b._save_settings_hash(chunk_sizes or [2048, 512, 256], conversion_method)
            return

        # Phase 1: Convert PDFs -> markdown (per-PDF progress)
        total_steps = len(pdf_files) + 1
        converted: List[Path] = []
        for i, pdf in enumerate(pdf_files):
            cb(f"Converting {pdf.name}", i, total_steps)
            md = self._convert_if_needed(pdf)
            if md:
                converted.append(md)

        # Combine with pre-existing markdown (text/url uploads)
        all_md_set = {p.resolve() for p in converted}
        for md in existing_md:
            all_md_set.add(md.resolve())
        all_md = list(all_md_set)

        if not all_md:
            return

        # Phase 2: Index from markdown
        cb("Building index", len(pdf_files), total_steps)
        self.index_from_markdown(all_md, chunk_sizes, progress_callback=None)

        cb("Complete", total_steps, total_steps)
        self._persist_metadata()

        # Save settings hash for staleness detection
        builder = self._get_index_builder()
        builder._save_settings_hash(chunk_sizes or [2048, 512, 256], conversion_method)

    def upload_text(self, content: bytes, filename: str) -> PDFMetadata:
        """Upload a text or markdown file.

        Text/markdown files are written directly to the markdown directory
        (no conversion needed). They get a metadata header for consistency.
        """
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        stem = Path(filename).stem
        md_filename = f"{doc_id}_{stem}.md"

        markdown_dir = self.scope_dir / "markdown"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        md_path = markdown_dir / md_filename

        text = content.decode("utf-8")

        scope_label = (
            "Project Upload" if self.scope_type == "project" else "Session Upload"
        )
        header = f"# Document: {filename}\n# Source: {scope_label}\n\n---\n\n"
        md_path.write_text(header + text, encoding="utf-8")

        return PDFMetadata(
            pdf_id=doc_id,
            filename=filename,
            path=str(md_path),
            file_size=len(content),
            page_count=0,
        )

    def upload_url(self, url: str, context: str = "") -> PDFMetadata:
        """Upload content from a URL.

        Fetches the URL, converts HTML to markdown, and writes it to
        the markdown directory with a metadata header.

        Args:
            url: The URL to fetch (must be http or https).
            context: Optional user-provided context to prepend to the content.

        Returns:
            PDFMetadata with document info.

        Raises:
            ValueError: If URL format is invalid or content is unusable.
            ConnectionError: If the URL cannot be fetched.
        """
        markdown_content, page_title = fetch_url_as_markdown(url)

        doc_id = f"url_{uuid.uuid4().hex[:8]}"

        # Create sanitized filename from URL domain+path
        parsed = urlparse(url)
        domain_part = re.sub(r"[^a-zA-Z0-9]", "_", parsed.netloc)
        path_part = re.sub(r"[^a-zA-Z0-9]", "_", parsed.path.strip("/"))
        if path_part:
            sanitized = f"{domain_part}_{path_part}"
        else:
            sanitized = domain_part
        # Truncate to avoid extremely long filenames
        sanitized = sanitized[:80]
        md_filename = f"{doc_id}_{sanitized}.md"

        markdown_dir = self.scope_dir / "markdown"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        md_path = markdown_dir / md_filename

        scope_label = (
            "Project Upload" if self.scope_type == "project" else "Session Upload"
        )
        header = (
            f"# Document: {page_title}\n"
            f"# Source: {scope_label}\n"
            f"# URL: {url}\n"
            f"\n---\n\n"
        )

        body = markdown_content
        if context:
            body = f"{context}\n\n---\n\n{markdown_content}"

        full_content = header + body
        md_path.write_text(full_content, encoding="utf-8")

        display_name = page_title or parsed.netloc

        return PDFMetadata(
            pdf_id=doc_id,
            filename=display_name,
            path=str(md_path),
            file_size=len(full_content),
            page_count=0,
        )

    def upload_document(self, content: bytes, filename: str) -> PDFMetadata:
        """Upload a document, dispatching by file type.

        Supported: .pdf, .txt, .md, .markdown
        """
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return self.upload(content, filename)
        elif ext in (".txt", ".md", ".markdown"):
            return self.upload_text(content, filename)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()
        return False
