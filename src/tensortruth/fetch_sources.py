"""Documentation and paper fetching utilities.

Handles scraping of library documentation (Sphinx/Doxygen) and ArXiv papers.
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from .cli_paths import get_library_docs_dir, get_sources_config_path
from .core.types import DocType, SourceType
from .scrapers.arxiv import fetch_arxiv_paper, fetch_paper_category
from .scrapers.book import fetch_book, fetch_book_category
from .scrapers.common import process_url
from .scrapers.doxygen import fetch_doxygen_urls
from .scrapers.sphinx import fetch_inventory
from .utils.interactive import interactive_add
from .utils.sources_config import list_sources, load_user_sources, update_sources_config
from .utils.validation import validate_sources

# --- CONFIGURATION ---
MAX_WORKERS = 20  # Safe number for parallel downloads

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scrape_library(
    library_name,
    config,
    output_base_dir,
    max_workers=MAX_WORKERS,
    output_format="markdown",
    enable_cleanup=False,
    min_size=0,
):
    """Scrape documentation for a single library.

    Args:
        library_name: Name of the library (e.g., "pytorch_2.9")
        config: Library configuration dictionary
        output_base_dir: Base directory for output (e.g., ~/.tensortruth/library_docs)
        max_workers: Number of parallel workers
        output_format: Output format ('markdown' or 'html')
        enable_cleanup: Enable aggressive HTML cleanup
        min_size: Minimum file size in characters
    """
    # Create directory with 'library_' prefix to match build_db expectations
    # library_name already includes version (e.g., "pytorch_2.9")
    output_dir = os.path.join(output_base_dir, f"library_{library_name}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Scraping: {library_name} v{config['version']}")
    logger.info(f"Doc Type: {config.get('type', 'sphinx')}")
    logger.info(f"Output Format: {output_format}")
    logger.info(f"Cleanup: {'enabled' if enable_cleanup else 'disabled'}")
    if min_size > 0:
        logger.info(f"Min Size Filter: {min_size} characters")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'=' * 60}\n")

    # 1. Get the list of URLs based on documentation type
    doc_type = config.get(
        "type", "sphinx"
    )  # Changed from doc_type to type for consistency

    if doc_type == "doxygen":
        urls = fetch_doxygen_urls(config)
    elif doc_type == "sphinx":
        urls = fetch_inventory(config)
    else:
        logger.error(f"Unknown doc_type: {doc_type}. Supported: 'sphinx', 'doxygen'")
        return

    if not urls:
        logger.warning(f"No URLs found for {library_name}")
        return

    # 2. Download
    logger.info(f"Downloading {len(urls)} pages...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm for progress bar
        results = list(
            tqdm(
                executor.map(
                    lambda u: process_url(
                        u, config, output_dir, output_format, enable_cleanup, min_size
                    ),
                    urls,
                ),
                total=len(urls),
                desc=library_name,
            )
        )

    successful = sum(1 for r in results if r is True)
    skipped = sum(1 for r in results if r == "skipped")
    failed = len(results) - successful - skipped

    logger.info(f"\n✅ Successfully downloaded {successful}/{len(urls)} pages")
    if skipped > 0:
        logger.info(f"⏭️  Skipped {skipped} files (below {min_size} chars)")
    if failed > 0:
        logger.warning(f"Failed {failed} files")
    logger.info(f"{'=' * 60}\n")


def main():
    """Main entry point for unified source fetching."""
    parser = argparse.ArgumentParser(
        description="Fetch documentation sources (libraries, papers, books)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available sources
  tensor-truth-docs --list

  # Fetch library documentation (uses ~/.tensortruth/)
  tensor-truth-docs pytorch numpy

  # Fetch with custom paths
  tensor-truth-docs pytorch --library-docs-dir /data/docs --sources-config /data/sources.json

  # Fetch papers in a category
  tensor-truth-docs --type papers --category dl_foundations

  # Fetch books (auto-splits by TOC or page chunks)
  tensor-truth-docs --type books book_linear_algebra_cherney
  tensor-truth-docs --type books --category linear_algebra --converter marker
  tensor-truth-docs --type books --all --converter marker

  # Fetch all paper categories
  tensor-truth-docs --type papers --all --converter marker

  # Customize page chunking for books without TOC
  tensor-truth-docs --type books book_deep_learning_goodfellow --pages-per-chunk 20

  # Validate sources
  tensor-truth-docs --validate

Environment Variables:
  TENSOR_TRUTH_DOCS_DIR       Override default library docs directory
  TENSOR_TRUTH_SOURCES_CONFIG Override default sources config path
        """,
    )

    # Path configuration arguments
    parser.add_argument(
        "--library-docs-dir",
        help=(
            "Output directory for docs "
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

    # Action arguments
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available sources and exit",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate sources.json config against filesystem",
    )

    parser.add_argument(
        "--add",
        action="store_true",
        help="Interactive mode to add new sources (libraries, papers, or books)",
    )

    # Source type arguments
    parser.add_argument(
        "--type",
        choices=["library", "paper", "papers", "book", "books"],
        help="Type of source to fetch or add (with --add)",
    )

    parser.add_argument(
        "--category",
        help="Category name (for --type papers or --type books)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all items of the specified type (all books or all paper categories)",
    )

    parser.add_argument(
        "--arxiv-ids",
        nargs="+",
        help="ArXiv IDs to add or fetch (for --type papers)",
    )

    parser.add_argument(
        "--url",
        help="Source URL (for --add mode with books/libraries, skips URL prompt)",
    )

    parser.add_argument(
        "libraries",
        nargs="*",
        help="Library names to scrape (for --type library, or positional)",
    )

    # Fetching options
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers for library scraping (default: {MAX_WORKERS})",
    )

    parser.add_argument(
        "--converter",
        choices=["marker", "pymupdf"],
        default="marker",
        help=(
            "Markdown converter selection for papers/books. "
            "marker (default) provides better quality, pymupdf is faster"
        ),
    )

    parser.add_argument(
        "--format",
        choices=["markdown", "html", "pdf"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Enable aggressive HTML cleanup for library docs (recommended for Doxygen)",
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        metavar="CHARS",
        help="Minimum file size in characters for library docs (skip smaller files)",
    )

    parser.add_argument(
        "--pages-per-chunk",
        type=int,
        default=15,
        metavar="N",
        help=(
            "Pages per chunk for books without TOC or with split_method='none'/'manual' "
            "(default: 15)"
        ),
    )

    parser.add_argument(
        "--max-pages-per-chapter",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Max pages per TOC chapter before splitting into sub-chunks. "
            "Set to 0 for no limit (default: 0)"
        ),
    )

    args = parser.parse_args()

    # Resolve paths (CLI args override env vars override defaults)
    library_docs_dir = get_library_docs_dir(args.library_docs_dir)
    sources_config_path = get_sources_config_path(args.sources_config)

    # Load user's sources config
    # Note: In Phase 2, interactive CLI will populate this file
    try:
        config = load_user_sources(sources_config_path)
    except IOError:
        logger.warning("No sources config found, starting with empty config.")
        config = {
            SourceType.LIBRARIES: {},
            SourceType.PAPERS: {},
            SourceType.BOOKS: {},
        }
    except Exception as e:
        logger.error(f"Failed to load sources config: {e}")
        return 1

    # List mode
    if args.list:
        list_sources(config)
        return 0

    # Validate mode
    if args.validate:
        return validate_sources(sources_config_path, library_docs_dir)

    # Interactive add mode
    if args.add:
        return interactive_add(sources_config_path, library_docs_dir, args)

    # Determine source type
    if args.type == "library" or (not args.type and args.libraries):
        # Library documentation scraping
        libraries_to_scrape = args.libraries
        if not libraries_to_scrape:
            logger.error(
                "No libraries specified. Use --list to see available libraries."
            )
            return 1

        for lib_name in libraries_to_scrape:
            if lib_name not in config[SourceType.LIBRARIES]:
                logger.error(
                    f"Library '{lib_name}' not found in config. "
                    "Use --list to see available libraries."
                )
                continue

            lib_config = config[SourceType.LIBRARIES][lib_name]
            logger.info(f"\n=== Scraping {lib_name} ===")

            try:
                scrape_library(
                    lib_name,
                    lib_config,
                    library_docs_dir,
                    max_workers=args.workers,
                    output_format=args.format,
                    enable_cleanup=args.cleanup,
                    min_size=args.min_size,
                )

                # Auto-write to user's sources.json after successful fetch
                update_sources_config(
                    sources_config_path, SourceType.LIBRARIES, lib_name, lib_config
                )
            except Exception as e:
                logger.error(
                    f"Failed to scrape library {lib_name}: {e}. Continuing with next library..."
                )
                continue

    elif args.type == "papers":
        # Paper fetching
        if args.all:
            # Fetch all paper categories (excluding books)
            paper_categories = {
                name: cfg
                for name, cfg in config.get(SourceType.PAPERS, {}).items()
                if cfg.get("type") != DocType.PDF_BOOK
            }

            if not paper_categories:
                logger.error("No paper categories found in config")
                return 1

            logger.info(f"Fetching all {len(paper_categories)} paper categories")

            for category_name, category_config in paper_categories.items():
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Fetching category: {category_name}")
                logger.info(f"{'=' * 60}")

                try:
                    fetch_paper_category(
                        category_name,
                        category_config,
                        library_docs_dir,
                        output_format=args.format,
                        converter=args.converter,
                    )

                    # Update sources.json
                    update_sources_config(
                        sources_config_path,
                        SourceType.PAPERS,
                        category_name,
                        category_config,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to fetch paper category {category_name}: {e}. "
                        "Continuing with next category..."
                    )
                    continue

            return 0

        if not args.category:
            logger.error("--category required for --type papers (or use --all)")
            return 1

        if args.category not in config[SourceType.PAPERS]:
            if args.arxiv_ids:
                logger.info(
                    f"Category '{args.category}' not found: creating new category."
                )

                # Ask for inputs on display_name and description
                display_name = input(
                    f"Enter display name for category '{args.category}': "
                ).strip()
                if not display_name:
                    display_name = args.category.replace("_", " ").title()
                    logger.info(f"Using default display name: {display_name}")

                description = input(
                    f"Enter description for category '{args.category}': "
                ).strip()
                if not description:
                    description = f"Papers in the {display_name} category"
                    logger.info(f"Using default description: {description}")

                config[SourceType.PAPERS][args.category] = {
                    "type": DocType.ARXIV,
                    "display_name": display_name,
                    "description": description,
                    "items": {},
                }

            else:
                logger.error(
                    f"Paper category '{args.category}' not found. "
                    "Use --list to see available categories."
                )
                return 1

        category_config = config[SourceType.PAPERS][args.category]

        # If specific IDs provided, fetch only those and add to category
        if args.arxiv_ids:
            output_dir = os.path.join(library_docs_dir, f"papers_{args.category}")
            os.makedirs(output_dir, exist_ok=True)

            # Ensure category has items dict
            if "items" not in category_config:
                category_config["items"] = {}

            # Fetch each paper and add to category if not already present
            for arxiv_id in args.arxiv_ids:
                # Check if already in category (using arxiv ID)
                if arxiv_id not in category_config["items"]:
                    # Fetch paper metadata from ArXiv
                    try:
                        import arxiv as arxiv_lib

                        search = arxiv_lib.Search(id_list=[arxiv_id])
                        paper = next(search.results())

                        # Extract authors and year from ArXiv metadata
                        authors = ", ".join([author.name for author in paper.authors])
                        year = str(paper.published.year)

                        # Add to category items dict with arxiv
                        category_config["items"][arxiv_id] = {
                            "title": paper.title,
                            "arxiv_id": arxiv_id,
                            "source": f"https://arxiv.org/abs/{arxiv_id}",
                            "authors": authors,
                            "year": year,
                        }
                        logger.info(
                            f"Added {paper.title} by {authors} ({year}) "
                            f"to category {args.category}"
                        )
                    except Exception as e:
                        logger.warning(f"Could not fetch metadata for {arxiv_id}: {e}")

                # Fetch the paper PDF and/or convert
                fetch_arxiv_paper(
                    arxiv_id,
                    output_dir,
                    output_format=args.format,
                    converter=args.converter,
                )

            # Update sources.json after adding papers with --ids
            update_sources_config(
                sources_config_path, SourceType.PAPERS, args.category, category_config
            )
        else:
            # Fetch entire category
            try:
                fetch_paper_category(
                    args.category,
                    category_config,
                    library_docs_dir,
                    output_format=args.format,
                    converter=args.converter,
                )

                # Auto-write to user's sources.json after successful fetch
                update_sources_config(
                    sources_config_path,
                    SourceType.PAPERS,
                    args.category,
                    category_config,
                )
            except Exception as e:
                logger.error(f"Failed to fetch paper category {args.category}: {e}")
                return 1

    elif args.type == "books":
        # Book fetching
        all_books = config.get(SourceType.BOOKS, {})

        if args.all:
            # Fetch all books
            if not all_books:
                logger.error("No books found in config")
                return 1

            logger.info(f"Fetching all {len(all_books)} books")

            success_count = 0
            for book_name, book_config in all_books.items():
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Fetching: {book_config.get('title')}")
                logger.info(f"{'=' * 60}")

                try:
                    if fetch_book(
                        book_name,
                        book_config,
                        library_docs_dir,
                        converter=args.converter,
                        pages_per_chunk=args.pages_per_chunk,
                        max_pages_per_chapter=args.max_pages_per_chapter,
                    ):
                        success_count += 1
                        # Update sources.json after each successful fetch
                        update_sources_config(
                            sources_config_path,
                            SourceType.BOOKS,
                            book_name,
                            book_config,
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to fetch book {book_name} ({book_config.get('title')}): {e}. "
                        "Continuing with next book..."
                    )
                    continue

            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"Summary: Successfully fetched {success_count}/{len(all_books)} books"
            )
            logger.info(f"{'=' * 60}")
            return 0

        if args.libraries:
            # Fetch specific books by name
            for book_name in args.libraries:
                if book_name not in all_books:
                    logger.error(
                        f"Book '{book_name}' not found. "
                        "Use --list to see available books."
                    )
                    continue

                book_config = all_books[book_name]
                logger.info(f"\n=== Fetching {book_name} ===")

                try:
                    fetch_book(
                        book_name,
                        book_config,
                        library_docs_dir,
                        converter=args.converter,
                        pages_per_chunk=args.pages_per_chunk,
                        max_pages_per_chapter=args.max_pages_per_chapter,
                    )

                    # Update sources.json
                    update_sources_config(
                        sources_config_path, SourceType.BOOKS, book_name, book_config
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to fetch book {book_name}: {e}. "
                        "Continuing with next book..."
                    )
                    continue

        elif args.category:
            # Fetch all books in category
            try:
                fetch_book_category(
                    args.category,
                    config,
                    library_docs_dir,
                    converter=args.converter,
                    pages_per_chunk=args.pages_per_chunk,
                    max_pages_per_chapter=args.max_pages_per_chapter,
                )

                # Update all books in category
                for name, cfg in all_books.items():
                    if cfg.get("category") == args.category:
                        update_sources_config(
                            sources_config_path, SourceType.BOOKS, name, cfg
                        )
            except Exception as e:
                logger.error(f"Failed to fetch book category {args.category}: {e}")
                return 1
        else:
            logger.error(
                "Must specify book names, --category, or --all for --type books"
            )
            return 1

    else:
        logger.error(
            "Must specify --type library, --type papers, or --type books, "
            "or provide library names directly"
        )
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
