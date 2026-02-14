"""Builds and manages vector indexes for uploaded documents."""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.vector_stores.chroma import ChromaVectorStore

from .core.ollama import get_ollama_url
from .rag_engine import get_embed_model
from .utils.metadata import extract_uploaded_pdf_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIndexBuilder:
    """Builds and manages vector indexes for uploaded documents."""

    def __init__(
        self,
        index_dir: Path,
        markdown_dir: Path,
        metadata_cache: Optional[Dict[str, Dict]] = None,
    ):
        """
        Initialize document index builder.

        Args:
            index_dir: Path to the vector index directory
            markdown_dir: Path to the markdown documents directory
            metadata_cache: Optional pre-loaded metadata cache from scope JSON
        """
        self.index_dir = Path(index_dir)
        self.markdown_dir = Path(markdown_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_cache = metadata_cache or {}
        self._chroma_client: Optional[chromadb.ClientAPI] = None

    def _extract_pdf_id_from_filename(self, filename: str) -> str:
        """Extract PDF ID from markdown filename (e.g., 'pdf_abc123.md' -> 'pdf_abc123')."""
        return Path(filename).stem

    def _get_cached_metadata(self, pdf_id: str) -> Optional[Dict]:
        """Get cached metadata for a PDF from the cache."""
        return self.metadata_cache.get(pdf_id)

    def _update_metadata_cache(self, pdf_id: str, metadata: Dict) -> None:
        """Update the metadata cache for a PDF."""
        self.metadata_cache[pdf_id] = metadata

    def get_metadata_cache(self) -> Dict[str, Dict]:
        """Get the complete metadata cache (for saving to scope JSON)."""
        return self.metadata_cache

    def index_exists(self) -> bool:
        """Check if a valid ChromaDB index exists."""
        chroma_db = self.index_dir / "chroma.sqlite3"
        docstore = self.index_dir / "docstore.json"
        return chroma_db.exists() and docstore.exists()

    def build_index_from_pdfs(
        self, pdf_files: List[Path], chunk_sizes: List[int] | None = None
    ) -> None:
        """
        Build ChromaDB vector index directly from PDF files (fast path).

        Args:
            pdf_files: List of PDF file paths
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256])

        Raises:
            ValueError: If no PDF files provided
            Exception: If indexing fails
        """
        if chunk_sizes is None:
            chunk_sizes = [2048, 512, 256]

        if not pdf_files:
            raise ValueError("No PDF files provided")

        logger.info(
            f"Building index from {len(pdf_files)} PDFs (direct mode, no markdown)"
        )

        try:
            # Clean existing index if present
            if self.index_dir.exists():
                logger.info(f"Removing old index: {self.index_dir}")
                shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Load PDFs directly
            from llama_index.readers.file import PDFReader

            reader = PDFReader()
            documents = []

            for pdf_file in pdf_files:
                logger.info(f"Loading PDF: {pdf_file.name}")
                docs = reader.load_data(pdf_file)

                # Set file_path metadata for each document (needed for metadata extraction)
                for doc in docs:
                    doc.metadata["file_path"] = str(pdf_file)
                    doc.metadata["file_name"] = pdf_file.name

                documents.extend(docs)

            if not documents:
                raise ValueError("No documents loaded from PDFs")

            logger.info(f"Loaded {len(documents)} documents from PDFs")

            # Extract metadata (same as markdown path)
            self._extract_and_inject_metadata(documents)

            # Build index (same as markdown path)
            self._build_vector_index(documents, chunk_sizes)

        except Exception as e:
            logger.error(f"Failed to build index from PDFs: {e}")
            raise

    def _extract_and_inject_metadata(self, documents: List) -> None:
        """Extract and inject metadata into documents.

        Args:
            documents: List of LlamaIndex Document objects
        """
        logger.info("Extracting metadata from uploaded PDFs...")

        try:
            ollama_url = get_ollama_url()

            for i, doc in enumerate(documents):
                try:
                    # Get file_path from metadata
                    file_path_str = doc.metadata.get("file_path", "")
                    if not file_path_str or not isinstance(file_path_str, str):
                        logger.debug(
                            f"Skipping metadata extraction for document {i} "
                            "(no valid file_path)"
                        )
                        continue

                    file_path = Path(file_path_str)
                    pdf_id = self._extract_pdf_id_from_filename(file_path.name)

                    # Check cache first
                    cached_metadata = self._get_cached_metadata(pdf_id)

                    if cached_metadata:
                        logger.info(f"  Using cached metadata for {pdf_id}")
                        metadata = cached_metadata
                    else:
                        # Always use LLM extraction for uploaded PDFs
                        # (embedded PDF metadata is often incorrect - publishers
                        # instead of authors, journal names in titles, etc.)
                        logger.info(f"  Extracting metadata for {pdf_id} with LLM...")
                        metadata = extract_uploaded_pdf_metadata(
                            doc=doc,
                            file_path=file_path,
                            ollama_url=ollama_url,
                        )

                        # Cache the metadata
                        self._update_metadata_cache(pdf_id, metadata)

                    # Inject essential metadata fields
                    essential_fields = [
                        "display_name",
                        "authors",
                        "source_url",
                        "doc_type",
                    ]
                    for field in essential_fields:
                        if field in metadata:
                            doc.metadata[field] = metadata[field]

                except Exception as e:
                    logger.warning(f"Failed to extract metadata for document {i}: {e}")

            logger.info(
                f">> Metadata extraction complete for {len(documents)} documents"
            )

        except Exception as e:
            logger.warning(f"Metadata extraction unavailable: {e}")
            logger.info("Continuing index build without metadata enrichment")

    def _build_vector_index(self, documents: List, chunk_sizes: List[int]) -> None:
        """Build the vector index from documents.

        Args:
            documents: List of LlamaIndex Document objects
            chunk_sizes: Hierarchical chunk sizes
        """
        # Parse with hierarchical chunking
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)
        logger.info(f"Parsed {len(nodes)} nodes ({len(leaf_nodes)} leaves)")

        # Create ChromaDB vector store
        self._chroma_client = chromadb.PersistentClient(path=str(self.index_dir))
        collection = self._chroma_client.get_or_create_collection("data")
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # Build index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.docstore.add_documents(nodes)

        # Force CPU for session indexing
        logger.info("Embedding documents on CPU (this may take a while)...")
        embed_model = get_embed_model(device="cpu")

        VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

        # Persist to disk
        storage_context.persist(persist_dir=str(self.index_dir))
        logger.info(f"Document index built: {self.index_dir}")

    def build_index(
        self,
        markdown_files: Optional[List[Path]] = None,
        chunk_sizes: List[int] | None = None,
    ) -> None:
        """
        Build ChromaDB vector index from markdown files.

        Args:
            markdown_files: List of markdown file paths (if None, uses all in markdown dir)
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256])

        Raises:
            ValueError: If no markdown files found
            Exception: If indexing fails
        """
        if chunk_sizes is None:
            chunk_sizes = [2048, 512, 256]

        # Get markdown files
        if markdown_files is None:
            markdown_files = list(self.markdown_dir.glob("*.md"))

        if not markdown_files:
            raise ValueError(f"No markdown files found in {self.markdown_dir}")

        logger.info(
            f"Building index with {len(markdown_files)} documents in {self.index_dir}"
        )

        try:
            # Clean existing index if present
            if self.index_dir.exists():
                logger.info(f"Removing old index: {self.index_dir}")
                shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Load documents
            documents = []
            for md_file in markdown_files:
                logger.info(f"Loading: {md_file.name}")
                reader = SimpleDirectoryReader(input_files=[str(md_file)])
                docs = reader.load_data()
                documents.extend(docs)

            if not documents:
                raise ValueError("No documents loaded from markdown files")

            logger.info(f"Loaded {len(documents)} documents")

            # Extract and inject metadata
            self._extract_and_inject_metadata(documents)

            # Build vector index
            self._build_vector_index(documents, chunk_sizes)

        except Exception as e:
            logger.error(f"Failed to build document index: {e}")
            raise

    def rebuild_index(self, chunk_sizes: List[int] | None = None) -> None:
        """
        Rebuild index from all markdown files in directory.

        Args:
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256])
        """
        logger.info(f"Rebuilding index in {self.index_dir}")
        self.build_index(markdown_files=None, chunk_sizes=chunk_sizes)

    def delete_index(self) -> None:
        """Remove the index directory and all its contents."""
        if self.index_dir.exists():
            logger.info(f"Deleting index: {self.index_dir}")
            shutil.rmtree(self.index_dir)
        else:
            logger.warning(f"Index directory does not exist: {self.index_dir}")

    def get_index_size(self) -> int:
        """
        Get the size of the index directory in bytes.

        Returns:
            Size in bytes, or 0 if index doesn't exist
        """
        if not self.index_dir.exists():
            return 0

        total_size = 0
        for file_path in self.index_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def get_document_count(self) -> int:
        """
        Get the number of documents indexed.

        Returns:
            Number of markdown files, or 0 if none
        """
        return len(list(self.markdown_dir.glob("*.md")))

    def close(self) -> None:
        """
        Explicitly close ChromaDB client connections.

        This is important on Windows where SQLite file handles may remain
        open, preventing directory deletion.
        """
        if self._chroma_client is not None:
            try:
                # ChromaDB doesn't have an explicit close() method,
                # but deleting the reference and forcing GC helps
                del self._chroma_client
                self._chroma_client = None
                logger.debug("ChromaDB client reference released")
            except Exception as e:
                logger.warning(f"Error closing ChromaDB client: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False
