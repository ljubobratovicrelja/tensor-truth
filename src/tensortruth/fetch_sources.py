"""Documentation and paper fetching utilities.

Handles scraping of library documentation (Sphinx/Doxygen) and ArXiv papers.
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from tqdm import tqdm

from .cli_paths import get_library_docs_dir, get_sources_config_path
from .scrapers.arxiv import fetch_arxiv_paper, fetch_paper_category
from .scrapers.book import fetch_book, fetch_book_category
from .scrapers.common import process_url
from .scrapers.doxygen import fetch_doxygen_urls
from .scrapers.sphinx import fetch_inventory
from .utils.sources_config import list_sources, load_user_sources, update_sources_config
from .utils.validation import validate_sources

# ============================================================================
# Configuration Constants
# ============================================================================


class SourceType(str, Enum):
    """Source configuration section names."""

    LIBRARIES = "libraries"
    PAPERS = "papers"
    BOOKS = "books"


class DocType(str, Enum):
    """Documentation types."""

    SPHINX = "sphinx"
    DOXYGEN = "doxygen"
    ARXIV = "arxiv"
    PDF_BOOK = "pdf_book"


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


def prompt_for_url(prompt_message: str, examples: list[str] = None) -> str:
    """Prompt user for URL with validation and retry loop.

    Args:
        prompt_message: Message to display when prompting for URL
        examples: Optional list of example URLs to display

    Returns:
        Valid URL string

    Raises:
        SystemExit: If user cancels (exits with code 1)
    """
    print(prompt_message)
    if examples:
        print("Examples:")
        for example in examples:
            print(f"  - {example}")

    url = input("\nURL: ").strip()

    # Retry loop for URL validation
    while not validate_url(url):
        logger.error(f"Invalid or inaccessible URL: {url}")
        url = input("\nTry again (or press Enter to cancel): ").strip()
        if not url:
            print("Cancelled.")
            raise SystemExit(1)

    return url


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
# Library Detection Helpers
# ============================================================================


def detect_doc_type(doc_root: str) -> str:
    """Auto-detect documentation type (Sphinx or Doxygen).

    Args:
        doc_root: Documentation root URL

    Returns:
        "sphinx", "doxygen", or None if unknown
    """
    import requests

    try:
        # Check for Sphinx objects.inv
        inv_url = f"{doc_root.rstrip('/')}/objects.inv"
        response = requests.head(inv_url, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            logger.info("✓ Detected Sphinx docs (found objects.inv)")
            return "sphinx"
    except Exception:
        pass

    try:
        # Check for Doxygen index pages
        response = requests.get(doc_root, timeout=10)
        if response.status_code == 200:
            html = response.text.lower()
            # Common Doxygen indicators
            if "annotated.html" in html or "classes.html" in html or "doxygen" in html:
                logger.info("✓ Detected Doxygen docs")
                return "doxygen"
    except Exception:
        pass

    logger.warning("Could not auto-detect doc type")
    return None


def detect_objects_inv(doc_root: str) -> str:
    """Find objects.inv URL for Sphinx documentation.

    Args:
        doc_root: Documentation root URL

    Returns:
        Full URL to objects.inv or None if not found
    """
    import requests

    # Common locations to check
    locations = [
        "",  # Root
        "_static/",
        "en/latest/",
        "en/stable/",
        "_build/html/",
    ]

    base = doc_root.rstrip("/")

    for loc in locations:
        url = f"{base}/{loc}objects.inv" if loc else f"{base}/objects.inv"
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                logger.info(f"✓ Found objects.inv at: {url}")
                return url
        except Exception:
            continue

    logger.warning("Could not find objects.inv")
    return None


def detect_css_selector(doc_root: str) -> str:
    """Auto-detect CSS selector for main content.

    Args:
        doc_root: Documentation root URL

    Returns:
        CSS selector string or None
    """
    import requests
    from bs4 import BeautifulSoup

    try:
        response = requests.get(doc_root, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Try common selectors in order of preference
        selectors = [
            ("div[role='main']", soup.select("div[role='main']")),
            ("article[role='main']", soup.select("article[role='main']")),
            ("main", soup.select("main")),
            (".document", soup.select(".document")),
            (".content", soup.select(".content")),
        ]

        for selector, elements in selectors:
            if elements:
                logger.info(f"✓ Detected CSS selector: {selector}")
                return selector

        logger.warning("Could not auto-detect CSS selector")
        return None

    except Exception as e:
        logger.warning(f"Error detecting CSS selector: {e}")
        return None


# ============================================================================
# Book Metadata Helpers
# ============================================================================


def download_pdf_with_headers(url: str, output_path: str) -> str:
    """Download PDF with proper headers.

    Args:
        url: PDF URL
        output_path: Output file path

    Returns:
        Path to downloaded file or None on error
    """
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"✓ Downloaded PDF to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        return None


def extract_pdf_metadata(pdf_path: str) -> dict:
    """Extract title and authors from PDF metadata.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dict with 'title' and 'authors' keys
    """
    try:
        import fitz  # PyMuPDF

        with fitz.open(pdf_path) as doc:
            metadata = doc.metadata

            title = metadata.get("title", None)
            author_str = metadata.get("author", "")

            # Parse authors (handle various separator formats)
            authors = []
            if author_str:
                # Try different separators
                for sep in [";", ",", " and "]:
                    if sep in author_str:
                        authors = [a.strip() for a in author_str.split(sep)]
                        break
                else:
                    # Single author
                    authors = [author_str.strip()] if author_str.strip() else []

            logger.info(f"✓ Extracted metadata: title='{title}', authors={authors}")
            return {"title": title, "authors": authors}

    except Exception as e:
        logger.warning(f"Could not extract PDF metadata: {e}")
        return {"title": None, "authors": []}


def generate_book_name(title: str, authors: list) -> str:
    """Generate sanitized book config key.

    Args:
        title: Book title
        authors: List of author names

    Returns:
        Sanitized config key (e.g., "machine_learning_basics_smith")
    """
    # Sanitize title
    title_slug = sanitize_config_key(title) if title else "untitled"

    # Get first author's last name
    author_slug = ""
    if authors:
        # Extract surname: everything except the first name
        # Handles: "Smith", "John Smith", "Ludwig van Beethoven", "Juan de la Cruz"
        first_author = authors[0]
        parts = first_author.split()

        if len(parts) <= 1:
            # Single name, use as-is
            last_name = first_author
        else:
            # Take everything after the first name
            last_name = " ".join(parts[1:])

        author_slug = "_" + sanitize_config_key(last_name)

    # Combine and truncate if too long
    full_name = title_slug + author_slug
    if len(full_name) > 60:
        full_name = full_name[:60].rstrip("_")

    return full_name


# ============================================================================
# Library Addition Functions
# ============================================================================


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


# ============================================================================
# Book Addition Functions
# ============================================================================


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
    import os
    import tempfile

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

    pdf_path = download_pdf_with_headers(url, tmp_path)
    if not pdf_path:
        logger.error("Failed to download PDF")
        return 1

    # Step 3: Extract metadata
    print("\n⏳ Extracting metadata from PDF...")
    metadata = extract_pdf_metadata(pdf_path)

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
    "url": "https://example.com/book.pdf",
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
            from .scrapers.book import fetch_book

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
        return add_library_interactive(sources_config_path, library_docs_dir, args)
    elif source_type == "book":
        return add_book_interactive(sources_config_path, library_docs_dir, args)
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
