"""Documentation and paper fetching utilities.

Handles scraping of library documentation (Sphinx/Doxygen) and ArXiv papers.
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from .cli_paths import get_library_docs_dir, get_sources_config_path
from .scrapers.arxiv import fetch_arxiv_paper, fetch_paper_category
from .scrapers.book import fetch_book, fetch_book_category
from .scrapers.common import process_url
from .scrapers.doxygen import fetch_doxygen_urls
from .scrapers.sphinx import fetch_inventory
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
        library_name: Name of the library
        config: Library configuration dictionary
        output_base_dir: Base directory for output (e.g., ~/.tensortruth/library_docs)
        max_workers: Number of parallel workers
        output_format: Output format ('markdown' or 'html')
        enable_cleanup: Enable aggressive HTML cleanup
        min_size: Minimum file size in characters
    """
    output_dir = os.path.join(output_base_dir, f"{library_name}_{config['version']}")

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


# ============================================================================
# Interactive Add Feature - Helper Functions
# ============================================================================


def validate_url(url: str) -> bool:
    """Validate URL format and accessibility.

    Args:
        url: URL to validate

    Returns:
        True if URL is valid and accessible, False otherwise
    """
    import re

    # Basic regex check
    url_pattern = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    if not url_pattern.match(url):
        return False

    # Try HEAD request to check accessibility
    try:
        import requests

        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code < 400
    except Exception:
        # If HEAD fails, try GET
        try:
            import requests

            response = requests.get(url, timeout=10, allow_redirects=True)
            return response.status_code < 400
        except Exception:
            return False


def sanitize_config_key(name: str) -> str:
    """Sanitize name to valid sources.json key.

    Args:
        name: Name to sanitize

    Returns:
        Sanitized name (lowercase, alphanumeric + underscore)
    """
    import re

    # Convert to lowercase
    name = name.lower()
    # Replace non-alphanumeric with underscore
    name = re.sub(r"[^a-z0-9_-]", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name


def validate_arxiv_id(arxiv_id: str):
    """Validate and normalize ArXiv ID.

    Supports formats:
    - 1234.5678 (new format)
    - arch-ive/1234567 (old format)
    - https://arxiv.org/abs/1234.5678 (URL)

    Args:
        arxiv_id: ArXiv ID to validate

    Returns:
        Normalized ID or None if invalid
    """
    import re

    arxiv_id = arxiv_id.strip()

    # Extract from URL
    if "arxiv.org" in arxiv_id:
        match = re.search(r"(\d{4}\.\d{4,5})", arxiv_id)
        if match:
            return match.group(1)
        match = re.search(r"([a-z\-]+/\d{7})", arxiv_id)
        if match:
            return match.group(1)
        return None

    # Validate format
    # New format: YYMM.NNNNN
    if re.match(r"^\d{4}\.\d{4,5}$", arxiv_id):
        return arxiv_id

    # Old format: arch-ive/YYMMNNN
    if re.match(r"^[a-z\-]+/\d{7}$", arxiv_id):
        return arxiv_id

    logger.warning(f"Invalid ArXiv ID format: {arxiv_id}")
    return None


# ============================================================================
# Paper Addition Functions
# ============================================================================


def add_paper_interactive(sources_config_path, library_docs_dir, args):
    """Interactive ArXiv paper addition (replaces --ids flow).

    Flow:
    1. Prompt for category name
    2. If category doesn't exist, create it with metadata
    3. Prompt for ArXiv IDs (or use --arxiv-ids)
    4. Fetch metadata from ArXiv API
    5. Preview papers to be added
    6. Confirm and save to sources.json
    7. Optionally fetch papers immediately

    Args:
        sources_config_path: Path to sources.json
        library_docs_dir: Base directory for documentation
        args: Command line arguments (may contain --arxiv-ids, --category)

    Returns:
        Exit code (0 = success, 1 = error)
    """
    import re

    print("\n=== Adding ArXiv Papers ===\n")

    # Load existing config
    try:
        config = load_user_sources(sources_config_path)
    except IOError:
        config = {"libraries": {}, "papers": {}, "books": {}}

    # Step 1: Category
    category = args.category if hasattr(args, "category") and args.category else None
    if not category:
        print("Paper categories group related papers together.")
        print("Examples: dl_foundations, computer_vision, nlp, reinforcement_learning")
        category = input("\nEnter category name: ").strip().lower()
        category = sanitize_config_key(category)

    # Step 2: Check if category exists
    category_config = None
    if category in config.get("papers", {}):
        cat = config["papers"][category]
        # Make sure it's not a book
        if cat.get("type") == "pdf_book":
            logger.error(f"'{category}' is a book category, not a paper category")
            return 1

        category_config = cat
        print(f"\n✓ Using existing category: {cat.get('display_name', category)}")
        print(f"  Description: {cat.get('description', 'N/A')}")
        print(f"  Current papers: {len(cat.get('items', {}))}")
    else:
        print(f"\nCategory '{category}' does not exist. Creating new category...")

        display_name = input("Enter display name for category: ").strip()
        if not display_name:
            display_name = category.replace("_", " ").title()
            print(f"Using default: {display_name}")

        description = input("Enter category description: ").strip()
        if not description:
            description = f"Papers in the {display_name} category"
            print(f"Using default: {description}")

        category_config = {
            "type": "arxiv",
            "display_name": display_name,
            "description": description,
            "items": {},
        }

        config["papers"][category] = category_config

    # Step 3: Get ArXiv IDs
    arxiv_ids = (
        args.arxiv_ids if hasattr(args, "arxiv_ids") and args.arxiv_ids else None
    )
    if not arxiv_ids:
        print("\nEnter ArXiv IDs to add (space or comma separated):")
        print("Example: 1706.03762 2010.11929 1512.03385")
        ids_str = input("ArXiv IDs: ").strip()
        # Split by space or comma
        arxiv_ids = re.split(r"[\s,]+", ids_str)

    # Validate IDs
    arxiv_ids = [validate_arxiv_id(aid) for aid in arxiv_ids if aid]
    arxiv_ids = [aid for aid in arxiv_ids if aid is not None]
    if not arxiv_ids:
        logger.error("No valid ArXiv IDs provided")
        return 1

    # Step 4: Fetch metadata
    print(f"\nFetching metadata for {len(arxiv_ids)} papers...")
    papers_to_add = []

    for arxiv_id in arxiv_ids:
        # Check if already exists
        if arxiv_id in category_config.get("items", {}):
            existing = category_config["items"][arxiv_id]
            print(f"⚠️  {arxiv_id} already in category: {existing.get('title')}")
            continue

        # Fetch from ArXiv
        try:
            import arxiv as arxiv_lib

            search = arxiv_lib.Search(id_list=[arxiv_id])
            paper = next(search.results())

            paper_entry = {
                "title": paper.title,
                "arxiv_id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "authors": ", ".join([author.name for author in paper.authors]),
                "year": str(paper.published.year),
            }

            papers_to_add.append((arxiv_id, paper_entry))
            print(f"✓ {arxiv_id}: {paper.title} ({paper.published.year})")

        except Exception as e:
            logger.warning(f"Could not fetch metadata for {arxiv_id}: {e}")
            # Prompt for manual entry
            manual = input(f"Add {arxiv_id} manually? (y/n): ").strip().lower()
            if manual == "y":
                title = input("  Title: ").strip()
                authors = input("  Authors (comma-separated): ").strip()
                year = input("  Year: ").strip()

                paper_entry = {
                    "title": title,
                    "arxiv_id": arxiv_id,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "authors": authors,
                    "year": year,
                }
                papers_to_add.append((arxiv_id, paper_entry))

    if not papers_to_add:
        print("\nNo new papers to add.")
        return 0

    # Step 5: Confirm
    print(f"\n=== Adding {len(papers_to_add)} papers to '{category}' ===")
    for arxiv_id, paper in papers_to_add:
        print(f"  • {paper['title']} ({paper['year']})")

    confirm = input("\nAdd these papers? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return 1

    # Step 6: Add to config
    if "items" not in category_config:
        category_config["items"] = {}

    for arxiv_id, paper_entry in papers_to_add:
        category_config["items"][arxiv_id] = paper_entry

    # Save
    update_sources_config(sources_config_path, "papers", category, category_config)
    print(f"\n✓ Added {len(papers_to_add)} papers to category '{category}'")

    # Step 7: Offer to fetch
    fetch = input("\nFetch papers now? (y/n): ").strip().lower()
    if fetch == "y":
        output_dir = os.path.join(library_docs_dir, f"papers_{category}")
        os.makedirs(output_dir, exist_ok=True)

        converter = input("Converter (marker/pymupdf) [marker]: ").strip() or "marker"

        from .scrapers.arxiv import fetch_arxiv_paper

        for arxiv_id, _ in papers_to_add:
            fetch_arxiv_paper(
                arxiv_id, output_dir, output_format="markdown", converter=converter
            )

    return 0


# ============================================================================
# Main Interactive Entry Point
# ============================================================================


def interactive_add(sources_config_path, library_docs_dir, args):
    """Main interactive entry point for adding sources.

    Prompts user to select source type, then delegates to:
    - add_library_interactive() for libraries (not yet implemented - shows message)
    - add_book_interactive() for books (not yet implemented - shows message)
    - add_paper_interactive() for papers

    Args:
        sources_config_path: Path to sources.json
        library_docs_dir: Base directory for documentation
        args: Parsed command line arguments (for optional skip-prompt args)

    Returns:
        Exit code (0 = success, 1 = error)
    """
    print("\n" + "=" * 60)
    print("Interactive Source Addition")
    print("=" * 60)
    print("\nThis wizard will help you add a new source to Tensor-Truth.")
    print("You can add:")
    print("  1) Library - Documentation for a library (Sphinx/Doxygen)")
    print("  2) Book    - PDF textbook or reference book")
    print("  3) Paper   - ArXiv research paper(s)")
    print()

    # Check if type was provided via CLI
    source_type = None
    if hasattr(args, "type") and args.type:
        # Normalize type
        type_map = {
            "library": "library",
            "libraries": "library",
            "book": "book",
            "books": "book",
            "paper": "paper",
            "papers": "paper",
        }
        source_type = type_map.get(args.type.lower())
        if not source_type:
            logger.error(f"Invalid type: {args.type}")
            logger.error("Valid types: library, book, paper")
            return 1
    else:
        # Interactive type selection
        while True:
            choice = (
                input("What would you like to add? (1/2/3 or library/book/paper): ")
                .strip()
                .lower()
            )

            if choice in ["1", "library"]:
                source_type = "library"
                break
            elif choice in ["2", "book"]:
                source_type = "book"
                break
            elif choice in ["3", "paper"]:
                source_type = "paper"
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3 or library/book/paper.")
                continue

    # Delegate to appropriate handler
    if source_type == "library":
        print("\n⚠️  Library addition is not yet implemented.")
        print("For now, please add libraries manually to sources.json")
        print("See docs/PAPERS.md for the configuration format.")
        return 1
    elif source_type == "book":
        print("\n⚠️  Book addition is not yet implemented.")
        print("For now, please add books manually to sources.json")
        print("See docs/PAPERS.md for the configuration format.")
        return 1
    elif source_type == "paper":
        return add_paper_interactive(sources_config_path, library_docs_dir, args)
    else:
        logger.error(f"Unknown source type: {source_type}")
        return 1


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
        choices=["pymupdf", "marker"],
        default="pymupdf",
        help=(
            "Markdown converter selection for papers/books. "
            "Both are AI powered (pymupdf.layout is used)"
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
        config = {"libraries": {}, "papers": {}, "books": {}}
    except Exception as e:
        logger.error(f"Failed to load sources config: {e}")
        return 1

    # List mode
    if args.list:
        list_sources(config)
        return 0

    # Validate mode
    if args.validate:
        validate_sources(sources_config_path, library_docs_dir)
        return 0

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
            if lib_name not in config["libraries"]:
                logger.error(
                    f"Library '{lib_name}' not found in config. "
                    "Use --list to see available libraries."
                )
                continue

            lib_config = config["libraries"][lib_name]
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
                    sources_config_path, "libraries", lib_name, lib_config
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
                for name, cfg in config.get("papers", {}).items()
                if cfg.get("type") != "pdf_book"
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
                        sources_config_path, "papers", category_name, category_config
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

        if args.category not in config["papers"]:
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

                config["papers"][args.category] = {
                    "type": "arxiv",
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

        category_config = config["papers"][args.category]

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
                            "url": f"https://arxiv.org/abs/{arxiv_id}",
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
                sources_config_path, "papers", args.category, category_config
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
                    sources_config_path, "papers", args.category, category_config
                )
            except Exception as e:
                logger.error(f"Failed to fetch paper category {args.category}: {e}")
                return 1

    elif args.type == "books":
        # Book fetching
        # Books are stored in config["papers"] with type="pdf_book"
        all_books = {
            name: cfg
            for name, cfg in config.get("papers", {}).items()
            if cfg.get("type") == "pdf_book"
        }

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
                            sources_config_path, "papers", book_name, book_config
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
                        sources_config_path, "papers", book_name, book_config
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
                        update_sources_config(sources_config_path, "papers", name, cfg)
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
