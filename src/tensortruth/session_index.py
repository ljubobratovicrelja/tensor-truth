"""Session-specific vector index builder for uploaded PDFs."""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.vector_stores.chroma import ChromaVectorStore

from .app_utils.paths import get_session_index_dir, get_session_markdown_dir
from .rag_engine import get_embed_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionIndexBuilder:
    """Builds and manages session-specific vector indexes for uploaded PDFs."""

    def __init__(self, session_id: str):
        """
        Initialize session index builder.

        Args:
            session_id: Session identifier (e.g., "sess_abc123")
        """
        self.session_id = session_id
        self.session_index_dir = get_session_index_dir(session_id)
        self.session_markdown_dir = get_session_markdown_dir(session_id)

    def index_exists(self) -> bool:
        """Check if a valid ChromaDB index exists for this session."""
        chroma_db = self.session_index_dir / "chroma.sqlite3"
        docstore = self.session_index_dir / "docstore.json"
        return chroma_db.exists() and docstore.exists()

    def build_index(
        self, markdown_files: Optional[List[Path]] = None, chunk_sizes: List[int] = None
    ) -> None:
        """
        Build ChromaDB vector index from markdown files.

        Args:
            markdown_files: List of markdown file paths (if None, uses all in session markdown dir)
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 128])

        Raises:
            ValueError: If no markdown files found
            Exception: If indexing fails
        """
        if chunk_sizes is None:
            chunk_sizes = [2048, 512, 128]

        # Get markdown files
        if markdown_files is None:
            markdown_files = list(self.session_markdown_dir.glob("*.md"))

        if not markdown_files:
            raise ValueError(f"No markdown files found in {self.session_markdown_dir}")

        logger.info(
            f"Building index for session {self.session_id} with {len(markdown_files)} documents"
        )

        try:
            # Clean existing index if present
            if self.session_index_dir.exists():
                logger.info(f"Removing old index: {self.session_index_dir}")
                shutil.rmtree(self.session_index_dir)
            self.session_index_dir.mkdir(parents=True, exist_ok=True)

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

            # Parse with hierarchical chunking
            node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
            nodes = node_parser.get_nodes_from_documents(documents)
            leaf_nodes = get_leaf_nodes(nodes)
            logger.info(f"Parsed {len(nodes)} nodes ({len(leaf_nodes)} leaves)")

            # Create ChromaDB vector store
            db = chromadb.PersistentClient(path=str(self.session_index_dir))
            collection = db.get_or_create_collection("data")
            vector_store = ChromaVectorStore(chroma_collection=collection)

            # Build index
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            storage_context.docstore.add_documents(nodes)

            logger.info("Embedding documents (this may take a while)...")
            embed_model = get_embed_model()
            VectorStoreIndex(
                leaf_nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True,
            )

            # Persist to disk
            storage_context.persist(persist_dir=str(self.session_index_dir))
            logger.info(
                f"✅ Session index built successfully: {self.session_index_dir}"
            )

        except Exception as e:
            logger.error(f"Failed to build session index: {e}")
            raise

    def rebuild_index(self, chunk_sizes: List[int] = None) -> None:
        """
        Rebuild index from all markdown files in session directory.

        Args:
            chunk_sizes: Hierarchical chunk sizes (default: [2048, 512, 128])
        """
        logger.info(f"Rebuilding index for session {self.session_id}")
        self.build_index(markdown_files=None, chunk_sizes=chunk_sizes)

    def delete_index(self) -> None:
        """Remove the index directory and all its contents."""
        if self.session_index_dir.exists():
            logger.info(f"Deleting index: {self.session_index_dir}")
            shutil.rmtree(self.session_index_dir)
        else:
            logger.warning(f"Index directory does not exist: {self.session_index_dir}")

    def get_index_size(self) -> int:
        """
        Get the size of the index directory in bytes.

        Returns:
            Size in bytes, or 0 if index doesn't exist
        """
        if not self.session_index_dir.exists():
            return 0

        total_size = 0
        for file_path in self.session_index_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def get_document_count(self) -> int:
        """
        Get the number of documents indexed.

        Returns:
            Number of markdown files in session, or 0 if none
        """
        return len(list(self.session_markdown_dir.glob("*.md")))
