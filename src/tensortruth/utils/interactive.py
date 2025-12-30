"""Interactive CLI workflows for adding sources to Tensor-Truth.

Provides guided interactive wizards for adding libraries, books, and papers
to the sources configuration.
"""

import logging
import os
import re
import tempfile

from .detection import detect_css_selector, detect_doc_type, detect_objects_inv
from .pdf import extract_pdf_metadata, generate_book_name
from .sources_config import load_user_sources, update_sources_config
from .validation import prompt_for_url, sanitize_config_key, validate_arxiv_id

logger = logging.getLogger(__name__)


class SourceType:
    """Source configuration section names."""

    LIBRARIES = "libraries"
    PAPERS = "papers"
    BOOKS = "books"


class DocType:
    """Documentation types."""

    SPHINX = "sphinx"
    DOXYGEN = "doxygen"
    ARXIV = "arxiv"
    PDF_BOOK = "pdf_book"


def add_library_interactive(sources_config_path, library_docs_dir, args):
    """Interactive library addition with auto-detection.

    Flow:
    1. Prompt for library URL (or use --url)
    2. Auto-detect doc type (Sphinx/Doxygen)
    3. Auto-detect objects.inv and CSS selector
    4. Prompt for library name, display name, version
    5. Allow manual overrides for auto-detected values
    6. Confirm and save to sources.json
    7. Optionally fetch library immediately

    Args:
        sources_config_path: Path to sources.json
        library_docs_dir: Base directory for documentation
        args: Command line arguments (may contain --url)

    Returns:
        Exit code (0 = success, 1 = error)
    """
    print("\n=== Adding Library Documentation ===\n")

    # Load existing config
    try:
        config = load_user_sources(sources_config_path)
    except IOError:
        config = {
            SourceType.LIBRARIES: {},
            SourceType.PAPERS: {},
            SourceType.BOOKS: {},
        }

    # Step 1: Get URL
    url = args.url if hasattr(args, "url") and args.url else None
    if not url:
        url = prompt_for_url(
            "Enter the root URL of the library documentation:",
            examples=[
                "https://pytorch.org/docs/stable/",
                "https://numpy.org/doc/stable/",
            ],
        )

    # Step 2: Auto-detect doc type
    print("\n⏳ Auto-detecting documentation type...")
    doc_type = detect_doc_type(url)

    if not doc_type:
        print("\nCould not auto-detect type. Please select:")
        print("  1) Sphinx")
        print("  2) Doxygen")
        choice = input("Doc type (1/2): ").strip()
        doc_type = (
            DocType.SPHINX
            if choice == "1"
            else DocType.DOXYGEN if choice == "2" else None
        )

        if not doc_type:
            logger.error("Invalid doc type selection")
            return 1

    # Step 3: Auto-detect configuration based on type
    lib_config = {"type": doc_type, "doc_root": url}

    if doc_type == DocType.SPHINX:
        # Detect objects.inv
        print("\n⏳ Looking for objects.inv...")
        inv_url = detect_objects_inv(url)
        if inv_url:
            lib_config["inventory_url"] = inv_url
            use_inv = (
                input("Use detected inventory URL? (y/n) [y]: ").strip().lower() or "y"
            )
            if use_inv != "y":
                custom_inv = input("Enter custom inventory URL: ").strip()
                if custom_inv:
                    lib_config["inventory_url"] = custom_inv

    # Detect CSS selector
    print("\n⏳ Detecting main content selector...")
    selector = detect_css_selector(url)
    if selector:
        lib_config["selector"] = selector
        use_selector = (
            input(f"Use detected CSS selector '{selector}'? (y/n) [y]: ")
            .strip()
            .lower()
            or "y"
        )
        if use_selector != "y":
            custom_selector = input("Enter custom CSS selector: ").strip()
            if custom_selector:
                lib_config["selector"] = custom_selector
    else:
        # Prompt for manual selector
        custom_selector = input(
            "Enter CSS selector for main content (e.g., div[role='main']): "
        ).strip()
        if custom_selector:
            lib_config["selector"] = custom_selector

    # Step 4: Get library metadata
    print("\n=== Library Metadata ===")

    lib_name = input("\nEnter library config key (e.g., pytorch, numpy): ").strip()
    lib_name = sanitize_config_key(lib_name)

    if lib_name in config.get(SourceType.LIBRARIES, {}):
        logger.error(f"Library '{lib_name}' already exists in config")
        overwrite = input("Overwrite? (y/n): ").strip().lower()
        if overwrite != "y":
            return 1

    version = input("Enter version (e.g., 2.0, stable) [stable]: ").strip()
    if not version:
        version = "stable"

    lib_config["version"] = version

    # Step 5: Preview and confirm
    print("\n=== Library Configuration ===")
    print(f"Config Key:    {lib_name}")
    print(f"Version:       {version}")
    print(f"Type:          {doc_type}")
    print(f"Doc Root:      {url}")
    if "inventory_url" in lib_config:
        print(f"Inventory URL: {lib_config['inventory_url']}")
    if "selector" in lib_config:
        print(f"CSS Selector:  {lib_config['selector']}")

    confirm = input("\nAdd this library? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return 1

    # Step 6: Save
    config.setdefault(SourceType.LIBRARIES, {})[lib_name] = lib_config
    update_sources_config(
        sources_config_path, SourceType.LIBRARIES, lib_name, lib_config
    )
    print(f"\n✓ Added library '{lib_name}' to sources.json")

    # Step 7: Offer to fetch
    fetch = input("\nFetch library documentation now? (y/n): ").strip().lower()
    if fetch == "y":
        try:
            from ..fetch_sources import MAX_WORKERS, scrape_library

            scrape_library(
                lib_name,
                lib_config,
                library_docs_dir,
                max_workers=MAX_WORKERS,
                output_format="markdown",
            )
        except Exception as e:
            logger.error(f"Failed to fetch library: {e}")
            return 1

    return 0


def add_book_interactive(sources_config_path, library_docs_dir, args):
    """Interactive book addition with PDF metadata extraction.

    Flow:
    1. Prompt for book URL (or use --url)
    2. Download PDF temporarily
    3. Extract title and authors from PDF metadata
    4. Prompt to confirm/override metadata
    5. Generate book config key
    6. Prompt for category and split method
    7. Confirm and save to sources.json
    8. Optionally fetch book immediately

    Args:
        sources_config_path: Path to sources.json
        library_docs_dir: Base directory for documentation
        args: Command line arguments (may contain --url, --category)

    Returns:
        Exit code (0 = success, 1 = error)
    """
    print("\n=== Adding Book ===\n")

    # Load existing config
    try:
        config = load_user_sources(sources_config_path)
    except IOError:
        config = {
            SourceType.LIBRARIES: {},
            SourceType.PAPERS: {},
            SourceType.BOOKS: {},
        }

    # Step 1: Get URL
    url = args.url if hasattr(args, "url") and args.url else None
    if not url:
        url = prompt_for_url(
            "Enter the URL of the PDF book:",
            examples=[
                "https://example.com/books/linear_algebra.pdf",
                "https://arxiv.org/pdf/1234.5678.pdf",
            ],
        )

    # Step 2: Download PDF temporarily
    print("\n⏳ Downloading PDF to extract metadata...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_path = tmp_file.name

    from ..utils.pdf import download_pdf_with_headers

    if not download_pdf_with_headers(url, tmp_path):
        logger.error("Failed to download PDF")
        return 1

    # Step 3: Extract metadata
    print("\n⏳ Extracting metadata from PDF...")
    metadata = extract_pdf_metadata(tmp_path)

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    # Step 4: Confirm/override metadata
    title = metadata.get("title")
    authors = metadata.get("authors", [])

    if title:
        print(f"\n✓ Detected title: {title}")
        use_title = input("Use this title? (y/n) [y]: ").strip().lower() or "y"
        if use_title != "y":
            title = input("Enter title: ").strip()
    else:
        print("\n⚠️  Could not detect title from PDF metadata")
        title = input("Enter title: ").strip()

    if authors:
        print(f"✓ Detected authors: {', '.join(authors)}")
        use_authors = input("Use these authors? (y/n) [y]: ").strip().lower() or "y"
        if use_authors != "y":
            authors_str = input("Enter authors (comma-separated): ").strip()
            authors = [a.strip() for a in authors_str.split(",")] if authors_str else []
    else:
        print("⚠️  Could not detect authors from PDF metadata")
        authors_str = input("Enter authors (comma-separated): ").strip()
        authors = [a.strip() for a in authors_str.split(",")] if authors_str else []

    if not title:
        logger.error("Title is required")
        return 1

    # Step 5: Generate config key
    book_name = generate_book_name(title, authors)
    print(f"\n✓ Generated config key: {book_name}")

    custom_name = input("Use this key? (y/n) or enter custom key [y]: ").strip()
    if custom_name.lower() not in ["", "y", "yes"]:
        book_name = sanitize_config_key(custom_name)

    # Check for duplicates
    if book_name in config.get(SourceType.BOOKS, {}):
        logger.error(f"Book '{book_name}' already exists in config")
        overwrite = input("Overwrite? (y/n): ").strip().lower()
        if overwrite != "y":
            return 1

    # Step 6: Get category and split method
    category = args.category if hasattr(args, "category") and args.category else None
    if not category:
        print("\nEnter category for this book (groups related books):")
        print("Examples: linear_algebra, machine_learning, deep_learning")
        category = input("Category: ").strip()
        category = sanitize_config_key(category)

    print("\nSelect split method:")
    print("  1. Split by table of contents (recommended)")
    print("  2. Keep as single document")
    print("  3. Define chapters manually")

    choice = input("\nEnter choice (1-3) [1]: ").strip()
    if not choice:
        choice = "1"

    if choice not in ["1", "2", "3"]:
        logger.error(f"Invalid choice: {choice}. Please select 1, 2, or 3.")
        return 1

    # Map choice to split method
    split_methods = {"1": "toc", "2": "none", "3": "manual"}
    split_method = split_methods[choice]

    # Block manual option with helpful message
    if split_method == "manual":
        print("\n⚠️  Manual chapter definition is not available in interactive mode.")
        print(
            "To define custom chapters, you need to manually edit the sources.json file."
        )
        print("\nExample configuration for splitting a book into 3 equal sections:")
        print(
            """
  "your_book_name": {
    "type": "pdf_book",
    "title": "Your Book Title",
    "authors": ["Author Name"],
    "category": "your_category",
    "source": "https://example.com/book.pdf",
    "split_method": "manual",
    "sections": [
      {"name": "Part 1 (Pages 1-100)", "pages": [1, 100]},
      {"name": "Part 2 (Pages 101-200)", "pages": [101, 200]},
      {"name": "Part 3 (Pages 201-300)", "pages": [201, 300]}
    ]
  }
"""
        )
        print("Please choose a different split method for now:")
        print("  1. Split by table of contents (recommended)")
        print("  2. Keep as single document")

        choice = input("\nEnter choice (1-2) [1]: ").strip()
        if not choice:
            choice = "1"

        if choice not in ["1", "2"]:
            logger.error(f"Invalid choice: {choice}. Please select 1 or 2.")
            return 1

        split_method = split_methods[choice]

    # Step 7: Build config
    book_config = {
        "type": DocType.PDF_BOOK,
        "title": title,
        "authors": authors,
        "category": category,
        "source": url,
        "split_method": split_method,
    }

    # Preview
    print("\n=== Book Configuration ===")
    print(f"Config Key:    {book_name}")
    print(f"Title:         {title}")
    print(f"Authors:       {', '.join(authors)}")
    print(f"Category:      {category}")
    print(f"Split Method:  {split_method}")
    print(f"Source:        {url}")

    confirm = input("\nAdd this book? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return 1

    # Step 8: Save to config["books"]
    config.setdefault(SourceType.BOOKS, {})[book_name] = book_config
    update_sources_config(sources_config_path, SourceType.BOOKS, book_name, book_config)
    print(f"\n✓ Added book '{book_name}' to sources.json")

    # Step 9: Offer to fetch
    fetch = input("\nFetch book now? (y/n): ").strip().lower()
    if fetch == "y":
        try:
            from ..scrapers.book import fetch_book

            converter = input("Converter (marker/pymupdf) [marker]: ").strip()
            if not converter:
                converter = "marker"

            fetch_book(
                book_name,
                book_config,
                library_docs_dir,
                converter=converter,
                pages_per_chunk=15,
                max_pages_per_chapter=0,
            )
        except Exception as e:
            logger.error(f"Failed to fetch book: {e}")
            return 1

    return 0


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
    print("\n=== Adding ArXiv Papers ===\n")

    # Load existing config
    try:
        config = load_user_sources(sources_config_path)
    except IOError:
        config = {
            SourceType.LIBRARIES: {},
            SourceType.PAPERS: {},
            SourceType.BOOKS: {},
        }

    # Step 1: Category
    category = args.category if hasattr(args, "category") and args.category else None
    if not category:
        print("Paper categories group related papers together.")
        print("Examples: dl_foundations, computer_vision, nlp, reinforcement_learning")
        category = input("\nEnter category name: ").strip().lower()
        category = sanitize_config_key(category)

    # Step 2: Check if category exists
    category_config = None
    if category in config.get(SourceType.PAPERS, {}):
        cat = config[SourceType.PAPERS][category]
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
            "type": DocType.ARXIV,
            "display_name": display_name,
            "description": description,
            "items": {},
        }

        config[SourceType.PAPERS][category] = category_config

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
                "source": f"https://arxiv.org/abs/{arxiv_id}",
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
                    "source": f"https://arxiv.org/abs/{arxiv_id}",
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
    update_sources_config(
        sources_config_path, SourceType.PAPERS, category, category_config
    )
    print(f"\n✓ Added {len(papers_to_add)} papers to category '{category}'")

    # Step 7: Offer to fetch
    fetch = input("\nFetch papers now? (y/n): ").strip().lower()
    if fetch == "y":
        output_dir = os.path.join(library_docs_dir, f"papers_{category}")
        os.makedirs(output_dir, exist_ok=True)

        converter = input("Converter (marker/pymupdf) [marker]: ").strip() or "marker"

        from ..scrapers.arxiv import fetch_arxiv_paper

        for arxiv_id, _ in papers_to_add:
            fetch_arxiv_paper(
                arxiv_id, output_dir, output_format="markdown", converter=converter
            )

    return 0


def interactive_add(sources_config_path, library_docs_dir, args):
    """Main interactive entry point for adding sources.

    Prompts user to select source type, then delegates to:
    - add_library_interactive() for libraries
    - add_book_interactive() for books
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
        return add_library_interactive(sources_config_path, library_docs_dir, args)
    elif source_type == "book":
        return add_book_interactive(sources_config_path, library_docs_dir, args)
    elif source_type == "paper":
        return add_paper_interactive(sources_config_path, library_docs_dir, args)
    else:
        logger.error(f"Unknown source type: {source_type}")
        return 1
