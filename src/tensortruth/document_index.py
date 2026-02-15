"""Builds and manages vector indexes for uploaded documents."""

import hashlib
import logging
import re
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
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

    _DOC_PREFIX_RE = re.compile(r"^(pdf|doc|url)_[a-f0-9]{7,8}_")

    def _extract_doc_id_from_filename(self, filename: str) -> str:
        """Extract the short doc ID prefix from a filename.

        Examples:
            pdf_abc123_My_Paper.pdf -> pdf_abc123
            doc_abcd1234_notes.md   -> doc_abcd1234
            url_12345678_slug.md    -> url_12345678
            simple.md               -> simple
        """
        stem = Path(filename).stem
        m = self._DOC_PREFIX_RE.match(stem)
        if m:
            return m.group(0).rstrip("_")
        return stem

    def _get_cached_metadata(self, pdf_id: str) -> Optional[Dict]:
        """Get cached metadata for a PDF from the cache."""
        return self.metadata_cache.get(pdf_id)

    def _update_metadata_cache(self, pdf_id: str, metadata: Dict) -> None:
        """Update the metadata cache for a PDF."""
        self.metadata_cache[pdf_id] = metadata

    def get_metadata_cache(self) -> Dict[str, Dict]:
        """Get the complete metadata cache (for saving to scope JSON)."""
        return self.metadata_cache

    def _get_chroma_client(self) -> chromadb.ClientAPI:
        """Get or create the shared ChromaDB PersistentClient.

        Handles ChromaDB's internal singleton cache: if the index directory
        was deleted and recreated since the last client was cached, the stale
        singleton causes a 'different settings' ValueError.  We catch that,
        evict the stale entry, and retry.
        """
        if self._chroma_client is None:
            try:
                self._chroma_client = chromadb.PersistentClient(
                    path=str(self.index_dir)
                )
            except ValueError:
                logger.debug("Clearing stale ChromaDB singleton cache and retrying")
                self._evict_chroma_cache()
                self._chroma_client = chromadb.PersistentClient(
                    path=str(self.index_dir)
                )
        return self._chroma_client

    @staticmethod
    def _evict_chroma_cache() -> None:
        """Clear ChromaDB's internal SharedSystemClient singleton cache.

        ChromaDB caches PersistentClient instances by path.  When the index
        directory is deleted and recreated, the stale singleton conflicts with
        a new client for the same path.  This is the only way to recover
        without restarting the process.
        """
        try:
            from chromadb.api.client import SharedSystemClient

            # Attribute name varies across chromadb versions (typo was fixed)
            cache = getattr(
                SharedSystemClient,
                "_identifier_to_system",
                getattr(SharedSystemClient, "_identifer_to_system", None),
            )
            if cache is not None:
                cache.clear()
        except (ImportError, AttributeError):
            pass

    def _release_chroma_client(self) -> None:
        """Release the cached ChromaDB client and evict singleton cache.

        Must be called before deleting / recreating the index directory.
        """
        if self._chroma_client is not None:
            del self._chroma_client
            self._chroma_client = None
        self._evict_chroma_cache()

    def index_exists(self) -> bool:
        """Check if a valid ChromaDB index exists."""
        chroma_db = self.index_dir / "chroma.sqlite3"
        docstore = self.index_dir / "docstore.json"
        return chroma_db.exists() and docstore.exists()

    def build_index_from_pdfs(
        self,
        pdf_files: List[Path],
        chunk_sizes: List[int] | None = None,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """
        Build ChromaDB vector index directly from PDF files (fast path).

        Args:
            pdf_files: List of PDF file paths
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256])
            progress_callback: Optional callback(stage, current, total) for progress

        Raises:
            ValueError: If no PDF files provided
            Exception: If indexing fails
        """
        if chunk_sizes is None:
            chunk_sizes = [2048, 512, 256]

        if not pdf_files:
            raise ValueError("No PDF files provided")

        cb = progress_callback or (lambda *a: None)

        logger.info(
            f"Building index from {len(pdf_files)} PDFs (direct mode, no markdown)"
        )

        try:
            # Clean existing index if present
            if self.index_dir.exists():
                logger.info(f"Removing old index: {self.index_dir}")
                self._release_chroma_client()
                shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: Loading PDFs (10%)
            cb("Loading PDFs", 10, 100)
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

            # Set doc_id on each document so ref_doc_id is tracked in docstore
            for doc in documents:
                file_path_str = doc.metadata.get("file_path", "")
                if file_path_str:
                    doc.doc_id = self._extract_doc_id_from_filename(
                        Path(file_path_str).name
                    )

            # Stage 2: Extracting metadata (40%)
            cb("Extracting metadata", 40, 100)
            self._extract_and_inject_metadata(documents)

            # Stage 3: Embedding documents (70%)
            cb("Embedding documents", 70, 100)
            self._build_vector_index(documents, chunk_sizes)

            # Stage 4: Complete
            cb("Complete", 100, 100)

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
                    pdf_id = self._extract_doc_id_from_filename(file_path.name)

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
        chroma_client = self._get_chroma_client()
        collection = chroma_client.get_or_create_collection("data")
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
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """
        Build ChromaDB vector index from markdown files.

        Args:
            markdown_files: List of markdown file paths (if None, uses all in markdown dir)
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 256])
            progress_callback: Optional callback(stage, current, total) for progress

        Raises:
            ValueError: If no markdown files found
            Exception: If indexing fails
        """
        if chunk_sizes is None:
            chunk_sizes = [2048, 512, 256]

        cb = progress_callback or (lambda *a: None)

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
                self._release_chroma_client()
                shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: Loading documents (10%)
            cb("Loading documents", 10, 100)
            documents = []
            for md_file in markdown_files:
                logger.info(f"Loading: {md_file.name}")
                reader = SimpleDirectoryReader(input_files=[str(md_file)])
                docs = reader.load_data()
                documents.extend(docs)

            if not documents:
                raise ValueError("No documents loaded from markdown files")

            logger.info(f"Loaded {len(documents)} documents")

            # Set doc_id on each document so ref_doc_id is tracked in docstore
            for doc in documents:
                file_path_str = doc.metadata.get("file_path", "")
                if file_path_str:
                    doc.doc_id = self._extract_doc_id_from_filename(
                        Path(file_path_str).name
                    )

            # Stage 2: Extracting metadata (40%)
            cb("Extracting metadata", 40, 100)
            self._extract_and_inject_metadata(documents)

            # Stage 3: Embedding documents (70%)
            cb("Embedding documents", 70, 100)
            self._build_vector_index(documents, chunk_sizes)

            # Stage 4: Complete
            cb("Complete", 100, 100)

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
        self._release_chroma_client()
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

    def get_indexed_doc_ids(self) -> Set[str]:
        """Get doc_ids tracked in the existing index's docstore.

        Returns:
            Set of doc_ids (e.g. {"pdf_abc123", "doc_def456"}), or empty set
            if no index exists or index lacks ref_doc tracking.
        """
        if not self.index_exists():
            return set()

        try:
            chroma_client = self._get_chroma_client()
            collection = chroma_client.get_or_create_collection("data")
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.index_dir),
                vector_store=vector_store,
            )
            ref_doc_info = storage_context.docstore.get_all_ref_doc_info()
            if ref_doc_info is None:
                return set()
            return set(ref_doc_info.keys())
        except Exception as e:
            logger.warning(f"Could not read indexed doc_ids: {e}")
            return set()

    def add_documents(
        self,
        documents: List,
        doc_ids: List[str],
        chunk_sizes: List[int],
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Incrementally add documents to an existing index.

        Args:
            documents: LlamaIndex Document objects to add.
            doc_ids: Corresponding doc_id for each document.
            chunk_sizes: Hierarchical chunk sizes.
            progress_callback: Optional callback(stage, current, total).
        """
        cb = progress_callback or (lambda *a: None)

        # Set doc_id on each document
        for doc, doc_id in zip(documents, doc_ids):
            doc.doc_id = doc_id

        logger.info(f"Incrementally adding {len(documents)} documents to index")

        # Extract metadata
        cb("Extracting metadata", 30, 100)
        self._extract_and_inject_metadata(documents)

        # Load existing index
        cb("Loading existing index", 50, 100)
        chroma_client = self._get_chroma_client()
        collection = chroma_client.get_or_create_collection("data")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(
            persist_dir=str(self.index_dir),
            vector_store=vector_store,
        )

        embed_model = get_embed_model(device="cpu")
        index = load_index_from_storage(storage_context, embed_model=embed_model)

        # Parse nodes
        cb("Embedding documents", 70, 100)
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)
        logger.info(f"Parsed {len(nodes)} nodes ({len(leaf_nodes)} leaves)")

        # Add ALL nodes to docstore (for hierarchy), insert leaves into vector index
        storage_context.docstore.add_documents(nodes)
        index.insert_nodes(leaf_nodes)

        # Persist
        storage_context.persist(persist_dir=str(self.index_dir))
        self._save_settings_hash(chunk_sizes)

        cb("Complete", 100, 100)
        logger.info(f"Incremental add complete: {len(documents)} documents added")

    def remove_document(self, doc_id: str) -> bool:
        """Remove a single document's nodes from the index.

        Args:
            doc_id: The document ID to remove (e.g. "pdf_abc123").

        Returns:
            True if document was found and removed, False otherwise.
        """
        if not self.index_exists():
            return False

        try:
            chroma_client = self._get_chroma_client()
            collection = chroma_client.get_or_create_collection("data")
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.index_dir),
                vector_store=vector_store,
            )

            # Check if doc_id exists in the docstore
            all_ref_info = storage_context.docstore.get_all_ref_doc_info()
            if all_ref_info is None or doc_id not in all_ref_info:
                return False

            # Get all node_ids for this ref_doc
            ref_info = all_ref_info[doc_id]
            node_ids = list(ref_info.node_ids)

            if node_ids:
                # Delete from ChromaDB
                collection.delete(ids=node_ids)
                logger.info(f"Removed {len(node_ids)} nodes from ChromaDB for {doc_id}")

            # Delete from docstore
            storage_context.docstore.delete_ref_doc(doc_id)

            # Persist
            storage_context.persist(persist_dir=str(self.index_dir))
            logger.info(f"Removed document {doc_id} from index")
            return True

        except Exception as e:
            logger.warning(f"Failed to remove document {doc_id} from index: {e}")
            return False

    def is_settings_current(
        self,
        chunk_sizes: List[int],
        conversion_method: str = "marker",
    ) -> bool:
        """Check if the index was built with the same settings.

        Args:
            chunk_sizes: Current chunk sizes to compare.
            conversion_method: Current conversion method to compare.

        Returns:
            True if settings match, False if stale or no hash file exists.
        """
        hash_file = self.index_dir / "settings_hash"
        if not hash_file.exists():
            return False
        stored = hash_file.read_text().strip()
        current = self._compute_settings_hash(chunk_sizes, conversion_method)
        return stored == current

    def _save_settings_hash(
        self,
        chunk_sizes: List[int],
        conversion_method: str = "marker",
    ) -> None:
        """Write a settings hash to the index directory."""
        hash_file = self.index_dir / "settings_hash"
        hash_file.write_text(
            self._compute_settings_hash(chunk_sizes, conversion_method)
        )

    @staticmethod
    def _compute_settings_hash(
        chunk_sizes: List[int],
        conversion_method: str = "marker",
    ) -> str:
        """Compute a deterministic hash of indexing settings."""
        content = f"{chunk_sizes}:{conversion_method}"
        return hashlib.sha256(content.encode()).hexdigest()

    def close(self) -> None:
        """
        Explicitly close ChromaDB client connections.

        Releases the cached client reference and evicts ChromaDB's internal
        singleton cache so the next builder for the same path can create a
        fresh client without 'different settings' errors.
        """
        self._release_chroma_client()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False
