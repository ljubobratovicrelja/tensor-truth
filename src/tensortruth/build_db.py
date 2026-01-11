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
from tensortruth.indexing.builder import build_module
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

        success = build_module(
            module,
            library_docs_dir,
            indexes_dir,
            sources_config,
            extensions=args.extensions,
            chunk_sizes=args.chunk_sizes,
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
