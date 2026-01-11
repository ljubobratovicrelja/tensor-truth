"""Core vector index building functionality.

This module provides the core business logic for building hierarchical
vector indexes from documentation with metadata extraction.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..app_utils.config_schema import TensorTruthConfig
from ..core.types import DocumentType
from ..rag_engine import get_embed_model
from ..utils.metadata import (
    extract_arxiv_metadata_from_config,
    extract_book_chapter_metadata,
    extract_library_metadata_from_config,
    extract_library_module_metadata,
    get_book_metadata_from_config,
    get_document_type_from_config,
)

logger = logging.getLogger(__name__)


def extract_metadata(
    module_name: str,
    doc_type: DocumentType,
    sources_config: Dict,
    documents: List[Document],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Extract metadata for a list of documents in a module.

    This function extracts document-specific metadata based on the document
    type and injects it into the document objects for use during indexing.

    Args:
        module_name: Name of the module (e.g., "pytorch", "dl_foundations")
        doc_type: Type of documents being processed (BOOK, LIBRARY, or PAPERS)
        sources_config: Loaded sources.json configuration
        documents: List of LlamaIndex Document objects to process
        progress_callback: Optional callback function(current, total) for progress updates

    Raises:
        ValueError: If document type is unknown or metadata extraction fails critically
    """
    # Prepare root metadata - book info for all chunks, or library for all doc modules etc.
    root_metadata = None

    if doc_type == DocumentType.BOOK:
        logger.info("Using book metadata extraction with caching.")
        root_metadata = get_book_metadata_from_config(module_name, sources_config)
    elif doc_type == DocumentType.LIBRARY:
        logger.info("Using library metadata extraction.")
        root_metadata = extract_library_metadata_from_config(
            module_name, sources_config
        )

    # Extract metadata for each document
    for i, doc in enumerate(documents):
        file_path = Path(doc.metadata.get("file_path", ""))
        try:
            # Extract metadata based on document type
            if doc_type == DocumentType.BOOK:
                # Book extraction with caching
                metadata = extract_book_chapter_metadata(
                    file_path,
                    root_metadata,
                )
            elif doc_type == DocumentType.LIBRARY:
                # Library module extraction, handles per-module URL and display name.
                metadata = extract_library_module_metadata(file_path, root_metadata)
            elif doc_type == DocumentType.PAPERS:
                # Per paper display name, authors, URL etc.
                metadata = extract_arxiv_metadata_from_config(
                    file_path, module_name, sources_config
                )
            else:
                raise ValueError(f"Unknown document type: {doc_type}")

            # Inject only essential metadata fields to avoid chunk size issues
            # (LlamaIndex includes metadata in chunk context)
            essential_fields = [
                "title",
                "formatted_authors",
                "display_name",
                "authors",
                "source_url",
                "doc_type",
                "group_display_name",  # For paper groups UI display
                "book_display_name",  # For book UI display (same across all chapters)
                "library_display_name",  # For library UI display (same across all modules)
            ]

            for field in essential_fields:
                if field in metadata:
                    doc.metadata[field] = metadata[field]

            # Progress reporting
            if (i + 1) % 10 == 0 or (i + 1) == len(documents):
                logger.info(f"  Processed {i + 1}/{len(documents)} documents...")
                if progress_callback:
                    progress_callback(i + 1, len(documents))

        except Exception as e:
            logger.warning(f"Failed to extract metadata for {file_path.name}: {e}")
            # Continue with default metadata

    logger.info(f"Metadata extraction complete for {len(documents)} documents")


def build_module(
    module_name: str,
    library_docs_dir: str,
    indexes_dir: str,
    sources_config: Dict,
    extensions: List[str] = None,
    chunk_sizes: List[int] = None,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> bool:
    """Build vector index for a documentation module.

    This is the core indexing function that:
    1. Loads documents from the source directory
    2. Extracts metadata based on document type
    3. Parses documents into hierarchical chunks
    4. Embeds chunks and creates vector index
    5. Persists the index to disk

    Args:
        module_name: Name of module (e.g., "pytorch", "dl_foundations")
        library_docs_dir: Base directory containing library documentation
        indexes_dir: Base directory for vector indexes
        sources_config: Loaded sources.json configuration
        extensions: List of file extensions to include (default: [".md", ".html", ".pdf"])
        chunk_sizes: Hierarchical chunk sizes for document parsing (default: [2048, 512, 256])
        device: Device for embedding ("cpu", "cuda", "mps"). Auto-detected if None.
        progress_callback: Optional callback function(stage, current, total) for progress updates

    Returns:
        True if build succeeded, False if build failed or was skipped

    Raises:
        ValueError: If module configuration is invalid
    """
    if extensions is None:
        extensions = [".md", ".html", ".pdf"]
    if chunk_sizes is None:
        chunk_sizes = [2048, 512, 256]

    # Get document type from config
    doc_type = get_document_type_from_config(module_name, sources_config)
    logger.info(f"Module '{module_name}' document type: {doc_type}")

    module_dir_name = f"{doc_type.value}_{module_name}"

    source_dir = os.path.join(library_docs_dir, module_dir_name)
    persist_dir = os.path.join(indexes_dir, module_dir_name)

    logger.info(f"\n--- BUILDING MODULE: {module_name} ---")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Target: {persist_dir}")

    # Validate source directory
    if not os.path.exists(source_dir):
        logger.error(f"Source directory missing: {source_dir}")
        return False

    # Remove old index if it exists
    if os.path.exists(persist_dir):
        logger.info(f"Removing old index at {persist_dir}...")
        shutil.rmtree(persist_dir)

    # Load documents
    try:
        documents = SimpleDirectoryReader(
            source_dir,
            recursive=True,
            required_exts=extensions,
            exclude_hidden=False,
        ).load_data()
    except Exception as e:
        logger.error(f"Failed to load documents from {source_dir}: {e}")
        return False

    logger.info(f"Loaded {len(documents)} documents.")

    if len(documents) == 0:
        logger.warning(f"No documents found in {source_dir}. Skipping module.")
        return False

    # Extract metadata
    if progress_callback:
        progress_callback("metadata", 0, len(documents))

    extract_metadata(
        module_name,
        doc_type,
        sources_config,
        documents,
        progress_callback=lambda curr, total: (
            progress_callback("metadata", curr, total) if progress_callback else None
        ),
    )

    # Parse documents into hierarchical nodes
    if progress_callback:
        progress_callback("parsing", 0, len(documents))

    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    logger.info(f"Parsed {len(nodes)} nodes ({len(leaf_nodes)} leaves).")

    # Create isolated vector database
    db = chromadb.PersistentClient(path=persist_dir)
    collection = db.get_or_create_collection("data")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Build index and persist
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes)

    # Detect device if not provided
    if device is None:
        device = TensorTruthConfig._detect_default_device()

    logger.info(f"Embedding on {device.upper()}...")

    if progress_callback:
        progress_callback("embedding", 0, len(leaf_nodes))

    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=get_embed_model(device=device),
        show_progress=True,
    )

    storage_context.persist(persist_dir=persist_dir)
    logger.info(f"Module '{module_name}' built successfully!")

    return True
