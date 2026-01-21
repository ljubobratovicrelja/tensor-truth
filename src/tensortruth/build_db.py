"""Vector database building CLI for Tensor-Truth.

This is a CLI wrapper around the core indexing functionality in tensortruth.indexing.
Handles argument parsing, path resolution, and batch operations.
"""

import argparse
import logging
import sys

from tensortruth.cli_paths import (
    get_base_indexes_dir,
    get_library_docs_dir,
    get_sources_config_path,
)
from tensortruth.fetch_sources import load_user_sources
from tensortruth.indexing.builder import DEFAULT_EMBEDDING_MODEL, build_module
from tensortruth.indexing.metadata import sanitize_model_id
from tensortruth.utils.validation import validate_module_for_build

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BUILDER")


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

  # Use a specific embedding model
  tensor-truth-build --modules pytorch --embedding-model Qwen/Qwen3-Embedding-0.6B

  # Skip model validation (for offline/private models)
  tensor-truth-build --modules pytorch --embedding-model my-org/my-model --no-validate

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
        default=[2048, 512, 256],
        help="Chunk sizes for hierarchical parsing (default: 2048 512 256)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Overlap tokens between chunks (default: 64). "
        "Must be smaller than smallest chunk size. Prevents information loss at boundaries.",
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".md", ".html", ".pdf"],
    )

    parser.add_argument(
        "--chunking-strategy",
        choices=["hierarchical", "semantic", "semantic_hierarchical"],
        default="hierarchical",
        help="Chunking strategy: hierarchical (fast, default), "
        "semantic (embedding-aware), or semantic_hierarchical (two-pass).",
    )

    parser.add_argument(
        "--semantic-buffer-size",
        type=int,
        default=1,
        help="Buffer size for semantic splitter (default: 1). Higher = larger chunks.",
    )

    parser.add_argument(
        "--semantic-breakpoint-threshold",
        type=int,
        default=95,
        help="Percentile threshold for semantic breaks (default: 95). "
        "Higher = fewer, larger chunks.",
    )

    # Embedding model options
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            f"HuggingFace embedding model to use (default: {DEFAULT_EMBEDDING_MODEL}). "
            "Accepts any HuggingFace model path (e.g., Qwen/Qwen3-Embedding-0.6B)."
        ),
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip HuggingFace model validation (for offline/private models)",
    )

    args = parser.parse_args()

    # Resolve embedding model (CLI arg, then config, then default)
    if args.embedding_model is None:
        try:
            from tensortruth.app_utils.config import load_config

            config = load_config()
            embedding_model = config.rag.default_embedding_model
        except Exception:
            embedding_model = DEFAULT_EMBEDDING_MODEL
    else:
        embedding_model = args.embedding_model

    # Validate embedding model exists on HuggingFace (optional)
    if not args.no_validate:
        try:
            from huggingface_hub import model_info

            logger.info(f"Validating embedding model: {embedding_model}")
            model_info(embedding_model)
            logger.info(f"Model {embedding_model} found on HuggingFace Hub")
        except ImportError:
            logger.warning(
                "huggingface_hub not installed, skipping model validation. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            logger.error(
                f"Embedding model '{embedding_model}' not found on HuggingFace Hub: {e}"
            )
            logger.info(
                "Use --no-validate to skip this check for private/offline models"
            )
            return 1

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

    model_id = sanitize_model_id(embedding_model)

    logger.info("")
    logger.info(f"Modules to build: {args.modules}")
    logger.info(f"Library docs dir: {library_docs_dir}")
    logger.info(f"Indexes dir: {indexes_dir}/{model_id}/")
    logger.info(f"Sources config: {sources_config_path}")
    logger.info(f"Embedding model: {embedding_model}")
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

        success = build_module(
            module,
            library_docs_dir,
            indexes_dir,
            sources_config,
            extensions=args.extensions,
            chunk_sizes=args.chunk_sizes,
            chunk_overlap=args.chunk_overlap,
            chunking_strategy=args.chunking_strategy,
            semantic_buffer_size=args.semantic_buffer_size,
            semantic_breakpoint_threshold=args.semantic_breakpoint_threshold,
            embedding_model=embedding_model,
        )

        logger.info("")
        logger.info("=" * 60)
        if success:
            logger.info(f"COMPLETE: Module {module}")
        else:
            logger.info(f"FAILED: Module {module}")
        logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
