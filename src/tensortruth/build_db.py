"""Vector database building utilities for Tensor-Truth.

Builds hierarchical vector indexes from markdown documentation.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.vector_stores.chroma import ChromaVectorStore

from tensortruth.app_utils.config_schema import TensorTruthConfig
from tensortruth.cli_paths import (
    get_base_indexes_dir,
    get_library_docs_dir,
    get_sources_config_path,
)
from tensortruth.fetch_sources import load_user_sources
from tensortruth.rag_engine import get_embed_model
from tensortruth.utils.metadata import (
    DocumentType,
    extract_arxiv_metadata_from_config,
    extract_book_chapter_metadata,
    extract_library_metadata_from_config,
    extract_library_module_metadata,
    get_book_metadata_from_config,
    get_document_type_from_config,
)
from tensortruth.utils.validation import validate_module_for_build

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BUILDER")


def extract_metadata(module_name, doc_type, sources_config, documents):
    """Extract metadata for a list of documents in a module."""

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

            if (i + 1) % 10 == 0 or (i + 1) == len(documents):
                logger.info(f"  Processed {i + 1}/{len(documents)} documents...")

        except Exception as e:
            logger.warning(f"Failed to extract metadata for {file_path.name}: {e}")
            # Continue with default metadata

    logger.info(f"Metadata extraction complete for {len(documents)} documents")


def build_module(
    module_name,
    library_docs_dir,
    indexes_dir,
    sources_config,
    extensions=[".md", ".html", ".pdf"],
    chunk_sizes=[2048, 512, 256],
):
    """Build vector index for a documentation module.

    Args:
        module_name: Name of module subdirectory in library_docs_dir
        library_docs_dir: Base directory containing library documentation
        indexes_dir: Base directory for vector indexes
        extensions: List of file extensions to include
        chunk_sizes: Hierarchical chunk sizes for document parsing
        extract_metadata: Whether to extract document metadata

    Returns:
        None
    """
    # Get document type from config

    doc_type = get_document_type_from_config(module_name, sources_config)
    logger.info(f"Module '{module_name}' document type: {doc_type}")

    module_dir_name = f"{doc_type.value}_{module_name}"

    source_dir = os.path.join(library_docs_dir, module_dir_name)
    persist_dir = os.path.join(indexes_dir, module_dir_name)

    logger.info(f"\n--- BUILDING MODULE: {module_name} ---")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Target: {persist_dir}")

    if not os.path.exists(source_dir):
        logger.error(f"Source directory missing: {source_dir}")
        return

    if os.path.exists(persist_dir):
        logger.info(f"Removing old index at {persist_dir}...")
        shutil.rmtree(persist_dir)

    try:
        documents = SimpleDirectoryReader(
            source_dir,
            recursive=True,
            required_exts=extensions,
            exclude_hidden=False,
        ).load_data()
    except Exception as e:
        logger.error(f"Failed to load documents from {source_dir}: {e}")
        return

    logger.info(f"Loaded {len(documents)} documents.")

    if len(documents) == 0:
        logger.warning(f"No documents found in {source_dir}. Skipping module.")
        return

    # Extract Metadata

    extract_metadata(module_name, doc_type, sources_config, documents)

    # Parse

    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    logger.info(f"Parsed {len(nodes)} nodes ({len(leaf_nodes)} leaves).")

    # Create Isolated DB

    # We use a unique collection name, though it's less critical since folders are separate
    db = chromadb.PersistentClient(path=persist_dir)
    collection = db.get_or_create_collection("data")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Index & Persist

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes)

    device = TensorTruthConfig._detect_default_device()
    logger.info(f"Embedding on {device.upper()}...")

    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=get_embed_model(device=device),
        show_progress=True,
    )

    storage_context.persist(persist_dir=persist_dir)
    logger.info(f"Module '{module_name}' built successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Build vector indexes from documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build specific modules (uses ~/.tensortruth/)
  tensor-truth-build --modules pytorch numpy

  # Build all modules found in library-docs-dir
  tensor-truth-build --all

  # Custom paths
  tensor-truth-build --modules pytorch \\
    --library-docs-dir /data/docs \\
    --indexes-dir /data/indexes

Environment Variables:
  TENSOR_TRUTH_DOCS_DIR       Library docs directory
  TENSOR_TRUTH_SOURCES_CONFIG Sources config path
  TENSOR_TRUTH_INDEXES_DIR    Vector indexes directory
        """,
    )

    # Path configuration arguments
    parser.add_argument(
        "--library-docs-dir",
        help=(
            "Source directory for docs "
            "(default: ~/.tensortruth/library_docs, or $TENSOR_TRUTH_DOCS_DIR)"
        ),
    )

    parser.add_argument(
        "--sources-config",
        help=(
            "Path to sources.json "
            "(default: ~/.tensortruth/sources.json, or $TENSOR_TRUTH_SOURCES_CONFIG)"
        ),
    )

    parser.add_argument(
        "--indexes-dir",
        help=(
            "Output directory for indexes "
            "(default: ~/.tensortruth/indexes, or $TENSOR_TRUTH_INDEXES_DIR)"
        ),
    )

    # Module selection arguments
    parser.add_argument(
        "--modules",
        nargs="+",
        help="Module names to build",
    )

    parser.add_argument(
        "--all", action="store_true", help="Build all modules in library-docs-dir"
    )

    parser.add_argument(
        "--books",
        action="store_true",
        help="Build all book modules found in sources.json.",
    )

    parser.add_argument(
        "--libraries",
        action="store_true",
        help="Build all library modules found in sources.json.",
    )

    parser.add_argument(
        "--papers",
        action="store_true",
        help="Build all paper modules found in sources.json.",
    )

    # Build options
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[2048, 512, 128],
        help="Chunk sizes for hierarchical parsing (default: 2048 512 128)",
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".md", ".html", ".pdf"],
    )

    args = parser.parse_args()

    # Resolve paths (CLI args override env vars override defaults)
    library_docs_dir = get_library_docs_dir(args.library_docs_dir)
    sources_config_path = get_sources_config_path(args.sources_config)
    indexes_dir = get_base_indexes_dir(args.indexes_dir)

    # Load sources config (for validation and metadata)
    sources_config = load_user_sources(sources_config_path)

    # Determine modules to build
    if args.all or args.books or args.libraries or args.papers:

        # Check if modules were also specified
        if args.modules:
            logger.error(
                "Cannot use --modules together with group selectors (all/books/libraries/papers)."
            )
            return 1

        papers = [item for item in sources_config.get("papers", {})]
        libraries = [item for item in sources_config.get("libraries", {})]
        books = [item for item in sources_config.get("books", {})]

        if args.all:
            args.modules = papers + libraries + books
        elif args.books:
            args.modules = books
        elif args.libraries:
            args.modules = libraries
        elif args.papers:
            args.modules = papers

        if not args.modules:
            logger.error(f"No modules found in {library_docs_dir}")
            logger.info("Run: tensor-truth-docs <library-name>")
            return 1

    elif not args.modules:
        logger.error("Must specify --modules or --all")
        parser.print_help()
        return 1

    logger.info("")
    logger.info(f"Modules to build: {args.modules}")
    logger.info(f"Library docs dir: {library_docs_dir}")
    logger.info(f"Indexes dir: {indexes_dir}")
    logger.info(f"Sources config: {sources_config_path}")
    logger.info("")

    # Validate all modules before building
    for module in args.modules:
        try:
            validate_module_for_build(module, library_docs_dir, sources_config)
        except ValueError as e:
            logger.error(f"Validation failed for '{module}': {e}")
            return 1

    # Build each module
    for module in args.modules:

        logger.info("")
        logger.info("=" * 60)
        logger.info(f" Building Module: {module} ")
        logger.info("=" * 60)
        logger.info("")

        build_module(
            module,
            library_docs_dir,
            indexes_dir,
            sources_config,
            args.extensions,
            args.chunk_sizes,
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"COMPLETE: Module {module}")
        logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
