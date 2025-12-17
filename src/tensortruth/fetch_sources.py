import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import requests
import sphobjinv as soi
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm

# Optional dependencies for paper fetching
try:
    import arxiv

    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

from .utils.pdf import clean_filename, convert_pdf_to_markdown

# --- CONFIGURATION ---
DEFAULT_CONFIG = "./config/sources.json"
OUTPUT_BASE_DIR = "./library_docs"
MAX_WORKERS = 20  # Safe number for parallel downloads

logging.basicConfig(level=logging.INFO)


def load_config(config_path=DEFAULT_CONFIG):
    """Load unified sources configuration (libraries + papers)."""
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return {"libraries": {}, "papers": {}}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Validate structure
        if "libraries" not in config:
            config["libraries"] = {}
        if "papers" not in config:
            config["papers"] = {}

        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return {"libraries": {}, "papers": {}}


def fetch_inventory(config):
    """Downloads and decodes the Sphinx objects.inv file."""
    print(f"Fetching inventory from {config['inventory_url']}...")
    try:
        inv = soi.Inventory(url=config["inventory_url"])
    except Exception as e:
        logging.error(f"Failed to fetch inventory: {e}")
        return []

    urls = set()
    # Iterate through all objects (functions, classes, methods)
    for obj in inv.objects:
        # We only want Python API docs, not generic labels or C++ docs
        if obj.domain == "py" and obj.role in [
            "function",
            "class",
            "method",
            "module",
            "data",
        ]:
            # Resolve relative URL to absolute
            full_url = os.path.join(config["doc_root"], obj.uri)
            # Remove anchors (#) to avoid duplicates
            clean_url = full_url.split("#")[0]
            urls.add(clean_url)

    print(f"Found {len(urls)} unique API pages.")
    return list(urls)


def fetch_doxygen_urls(config):
    """Extracts documentation URLs from Doxygen index pages."""
    doc_root = config["doc_root"]
    index_pages = config.get("index_pages", ["annotated.html", "modules.html"])

    print(f"Fetching Doxygen URLs from {doc_root}...")
    urls = set()

    for index_page in index_pages:
        index_url = urljoin(doc_root, index_page)
        print(f"  Parsing {index_page}...")

        try:
            resp = requests.get(index_url, timeout=10)
            if resp.status_code != 200:
                logging.warning(f"Failed to fetch {index_url}: {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.content, "html.parser")

            # Doxygen typically has links in tables or div.contents
            # We look for links to .html files (classes, structs, functions, modules)
            for link in soup.find_all("a", href=True):
                href = link["href"]

                # Skip external links, anchors, and non-HTML
                if href.startswith(("http://", "https://", "#", "javascript:")):
                    continue
                if not href.endswith(".html"):
                    continue

                # Skip index pages themselves and common navigation pages
                if href in [
                    "index.html",
                    "pages.html",
                    "annotated.html",
                    "classes.html",
                    "modules.html",
                    "namespaces.html",
                    "files.html",
                    "examples.html",
                ]:
                    continue

                # Build full URL
                full_url = urljoin(doc_root, href)
                urls.add(full_url)

        except Exception as e:
            logging.error(f"Error parsing {index_url}: {e}")

    print(f"Found {len(urls)} unique Doxygen pages.")
    return list(urls)


def clean_doxygen_html(soup):
    """
    Aggressively clean Doxygen HTML to remove noise while preserving semantic content.
    Focuses on keeping: class/function signatures, descriptions, parameters, code blocks.
    """
    # 1. Remove all visual-only elements (diagrams, images, iframes)
    for tag in soup.find_all(["iframe", "img", "svg"]):
        tag.decompose()

    # 2. Remove Doxygen-specific UI elements
    for cls in [
        "dynheader",
        "dyncontent",
        "center",
        "permalink",
        "mlabels",
        "mlabels-left",
        "mlabels-right",
        "python_language",
        "memSeparator",
    ]:
        for tag in soup.find_all(class_=cls):
            tag.decompose()

    # 3. Remove separator rows (just whitespace)
    for tag in soup.find_all("tr", class_="separator"):
        tag.decompose()

    # 4. Remove empty documentation blocks
    for tag in soup.find_all("div", class_="memdoc"):
        if not tag.get_text(strip=True):
            tag.decompose()

    # 5. Remove "This browser is not able to show SVG" messages
    for p in soup.find_all("p"):
        text = p.get_text()
        if (
            "This browser is not able to show SVG" in text
            or "try Firefox, Chrome" in text
        ):
            p.decompose()

    # 6. Remove footer (everything after first <hr>)
    hr_tags = soup.find_all("hr")
    if hr_tags:
        first_hr = hr_tags[0]
        # Remove all siblings after the hr
        for sibling in list(first_hr.find_next_siblings()):
            sibling.decompose()
        first_hr.decompose()

    # 7. Clean up inheritance/collaboration diagram sections
    for tag in soup.find_all("div", class_="dynheader"):
        tag.decompose()

    # 8. Simplify member tables - remove layout-only columns
    for table in soup.find_all("table", class_="memberdecls"):
        # Remove groupheader rows with just section titles (we'll keep h2s instead)
        for tr in table.find_all("tr", class_="heading"):
            # Extract the h2 and preserve it, remove the tr
            h2 = tr.find("h2")
            if h2:
                table.insert_before(h2)
            tr.decompose()

    # 9. Simplify method documentation tables
    for table in soup.find_all("table", class_="memname"):
        # Extract just the text content, preserve structure but remove excess markup
        # Keep the table but this is already fairly clean
        pass

    # 10. Remove empty anchor tags
    for a in soup.find_all("a"):
        if not a.get_text(strip=True) and not a.find("img"):
            a.decompose()

    # 11. Remove pure navigation links (those with ../../ paths that won't work locally)
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if href.startswith("../../") or href.startswith("../"):
            # Replace link with just its text content
            a.replace_with(a.get_text())

    # 12. Clean up code includes at the top
    # Keep them but they're useful context

    # 13. Remove excessive whitespace-only paragraphs
    for p in soup.find_all("p"):
        if not p.get_text(strip=True):
            p.decompose()

    return soup


def url_to_filename(url, doc_root):
    """Clean filename generation."""
    # Remove the base URL
    rel_path = url.replace(doc_root, "").strip("/")
    # Replace slashes/dots with underscores
    clean_name = re.sub(r"[^a-zA-Z0-9]", "_", rel_path)
    # Ensure markdown extension
    return f"{clean_name}.md"


def process_url(
    url, config, output_dir, output_format="markdown", enable_cleanup=False, min_size=0
):
    """
    Download and convert a single URL to markdown or save as HTML, with optional
    cleanup and size filtering.
    """
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return False

        soup = BeautifulSoup(resp.content, "html.parser")

        # Cleanup: remove scripts, styles, nav, footer, sidebar
        for tag in soup(
            ["script", "style", "nav", "footer", "div.sphinxsidebar", "aside"]
        ):
            tag.decompose()

        # Extract Main Content
        selector = config.get("selector", "main")
        content = soup.select_one(selector)
        if not content:
            content = soup.find("article") or soup.find("body")

        if content:
            # Apply aggressive cleanup if requested (especially useful for Doxygen)
            if enable_cleanup:
                content = clean_doxygen_html(content)

            # Generate content based on output format
            if output_format == "html":
                final_content = f"<!-- Source: {url} -->\n{str(content)}"
            else:
                # Convert to Markdown (default)
                # The content is already cleaned if cleanup was enabled
                markdown = md(str(content), heading_style="ATX", code_language="python")
                final_content = f"# Source: {url}\n\n{markdown}"

            # Check minimum size threshold
            if min_size > 0 and len(final_content) < min_size:
                return "skipped"  # Return special value to track filtered files

            # Save the file
            filename = url_to_filename(url, config["doc_root"])
            if output_format == "html":
                filename = filename.replace(".md", ".html")
            save_path = os.path.join(output_dir, filename)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            return True

    except Exception as e:
        logging.error(f"Error {url}: {e}")
        return False


def scrape_library(
    library_name,
    config,
    max_workers=MAX_WORKERS,
    output_format="markdown",
    enable_cleanup=False,
    min_size=0,
):
    """Scrape documentation for a single library."""
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"{library_name}_{config['version']}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n{'=' * 60}")
    print(f"Scraping: {library_name} v{config['version']}")
    print(f"Doc Type: {config.get('type', 'sphinx')}")
    print(f"Output Format: {output_format}")
    print(f"Cleanup: {'enabled' if enable_cleanup else 'disabled'}")
    if min_size > 0:
        print(f"Min Size Filter: {min_size} characters")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # 1. Get the list of URLs based on documentation type
    doc_type = config.get(
        "type", "sphinx"
    )  # Changed from doc_type to type for consistency

    if doc_type == "doxygen":
        urls = fetch_doxygen_urls(config)
    elif doc_type == "sphinx":
        urls = fetch_inventory(config)
    else:
        logging.error(f"Unknown doc_type: {doc_type}. Supported: 'sphinx', 'doxygen'")
        return

    if not urls:
        print(f"⚠️  No URLs found for {library_name}")
        return

    # 2. Download
    print(f"Downloading {len(urls)} pages...")

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

    print(f"\n✅ Successfully downloaded {successful}/{len(urls)} pages")
    if skipped > 0:
        print(f"⏭️  Skipped {skipped} files (below {min_size} chars)")
    if failed > 0:
        print(f"❌ Failed {failed} files")
    print(f"{'=' * 60}\n")


def fetch_arxiv_paper(arxiv_id, output_dir, converter="pymupdf"):
    """
    Fetch and convert a single ArXiv paper.

    Args:
        arxiv_id: ArXiv paper ID (e.g., "1706.03762")
        output_dir: Directory to save markdown
        converter: 'pymupdf' or 'marker'

    Returns:
        True if successful, False otherwise
    """
    if not ARXIV_AVAILABLE:
        logging.error(
            "arxiv package not installed. Install with: pip install tensor-truth[docs]"
        )
        return False

    try:
        # Search ArXiv
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())

        # Sanitize title for filename
        safe_title = clean_filename(paper.title)
        pdf_filename = f"{arxiv_id.replace('.', '_')}_{safe_title}.pdf"
        md_filename = f"{arxiv_id.replace('.', '_')}_{safe_title}.md"

        pdf_path = os.path.join(output_dir, pdf_filename)
        md_path = os.path.join(output_dir, md_filename)

        # Check if already processed
        if os.path.exists(md_path):
            logging.info(f"✅ Already processed: {md_filename}")
            return True

        # Download PDF
        logging.info(f"Downloading: {paper.title}")
        paper.download_pdf(dirpath=output_dir, filename=pdf_filename)

        # Convert to markdown
        logging.info(f"Converting to markdown with {converter}...")
        md_text = convert_pdf_to_markdown(
            pdf_path, preserve_math=True, converter=converter
        )

        # Save markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {paper.title}\n\n")
            f.write(f"**ArXiv ID**: {arxiv_id}\n")
            f.write(f"**Authors**: {', '.join([a.name for a in paper.authors])}\n")
            f.write(f"**Published**: {paper.published.strftime('%Y-%m-%d')}\n\n")
            f.write(f"**Abstract**:\n{paper.summary}\n\n")
            f.write("---\n\n")
            f.write(md_text)

        logging.info(f"✅ Saved: {md_filename}")

        # Optionally remove PDF to save space
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        return True

    except Exception as e:
        logging.error(f"Failed to fetch ArXiv paper {arxiv_id}: {e}")
        return False


def fetch_paper_category(
    category_name, category_config, workers=1, converter="pymupdf"
):
    """
    Fetch all papers in a category.

    Args:
        category_name: Category name (e.g., "dl_foundations")
        category_config: Category configuration dict from sources.json
        workers: Number of parallel workers (not implemented yet, use 1)
        converter: PDF converter ('pymupdf' or 'marker')
    """
    output_dir = os.path.join(OUTPUT_BASE_DIR, category_name)
    os.makedirs(output_dir, exist_ok=True)

    items = category_config.get("items", [])
    if not items:
        logging.warning(f"No items found in category: {category_name}")
        return

    logging.info(f"Fetching {len(items)} papers in category: {category_name}")

    success_count = 0
    for item in tqdm(items, desc=f"Fetching {category_name}"):
        arxiv_id = item.get("arxiv_id")
        if not arxiv_id:
            logging.warning(
                f"Missing arxiv_id for item: {item.get('title', 'Unknown')}"
            )
            continue

        if fetch_arxiv_paper(arxiv_id, output_dir, converter=converter):
            success_count += 1

    logging.info(f"✅ Successfully fetched {success_count}/{len(items)} papers")


def list_sources(config):
    """List all available sources (libraries and papers)."""
    print("\n=== Available Libraries ===")
    if config.get("libraries"):
        for name, lib_config in sorted(config["libraries"].items()):
            doc_type = lib_config.get("type", "unknown")
            version = lib_config.get("version", "?")
            print(f"  • {name:<20} ({doc_type}, v{version})")
    else:
        print("  (none)")

    print("\n=== Available Paper Categories ===")
    if config.get("papers"):
        for name, cat_config in sorted(config["papers"].items()):
            cat_type = cat_config.get("type", "unknown")
            desc = cat_config.get("description", "")
            item_count = len(cat_config.get("items", []))
            print(f"  • {name:<30} ({cat_type}, {item_count} items)")
            if desc:
                print(f"    {desc[:80]}...")
    else:
        print("  (none)")


def main():
    """Main entry point for unified source fetching."""
    parser = argparse.ArgumentParser(
        description="Fetch documentation sources (libraries, papers, books)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available sources
  tensor-truth-docs --list

  # Fetch library documentation
  tensor-truth-docs --type library pytorch numpy

  # Fetch papers in a category
  tensor-truth-docs --type papers --category dl_foundations

  # Fetch specific papers
  tensor-truth-docs --type papers --category dl_foundations --ids 1706.03762 1810.04805

  # Use marker converter for better math
  tensor-truth-docs --type papers --category dl_foundations --converter marker

  # Positional library names (backward compatible)
  tensor-truth-docs pytorch numpy
        """,
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to sources config file (default: {DEFAULT_CONFIG})",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available sources and exit",
    )

    parser.add_argument(
        "--type",
        choices=["library", "papers"],
        help="Type of source to fetch",
    )

    parser.add_argument(
        "--category",
        help="Paper category name (for --type papers)",
    )

    parser.add_argument(
        "--ids",
        nargs="+",
        help="Specific ArXiv IDs to fetch (for --type papers)",
    )

    parser.add_argument(
        "libraries",
        nargs="*",
        help="Library names to scrape (for --type library, or positional)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    parser.add_argument(
        "--converter",
        choices=["pymupdf", "marker"],
        default="pymupdf",
        help="PDF converter: 'pymupdf' (fast) or 'marker' (better math)",
    )

    parser.add_argument(
        "--format",
        choices=["markdown", "html"],
        default="markdown",
        help="Output format for library docs (default: markdown)",
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

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # List mode
    if args.list:
        list_sources(config)
        return 0

    # Determine source type
    if args.type == "library" or (not args.type and args.libraries):
        # Library documentation scraping
        libraries_to_scrape = args.libraries
        if not libraries_to_scrape:
            logging.error(
                "No libraries specified. Use --list to see available libraries."
            )
            return 1

        for lib_name in libraries_to_scrape:
            if lib_name not in config["libraries"]:
                logging.error(
                    f"Library '{lib_name}' not found in config. "
                    "Use --list to see available libraries."
                )
                continue

            lib_config = config["libraries"][lib_name]
            logging.info(f"\n=== Scraping {lib_name} ===")
            scrape_library(
                lib_name,
                lib_config,
                max_workers=args.workers,
                output_format=args.format,
                enable_cleanup=args.cleanup,
                min_size=args.min_size,
            )

    elif args.type == "papers":
        # Paper fetching
        if not ARXIV_AVAILABLE:
            logging.error(
                "arxiv package not installed. Install with: pip install tensor-truth[docs]"
            )
            return 1

        if not args.category:
            logging.error("--category required for --type papers")
            return 1

        if args.category not in config["papers"]:
            logging.error(
                f"Paper category '{args.category}' not found. "
                "Use --list to see available categories."
            )
            return 1

        category_config = config["papers"][args.category]

        # If specific IDs provided, fetch only those
        if args.ids:
            output_dir = os.path.join(OUTPUT_BASE_DIR, args.category)
            os.makedirs(output_dir, exist_ok=True)
            for arxiv_id in args.ids:
                fetch_arxiv_paper(arxiv_id, output_dir, converter=args.converter)
        else:
            # Fetch entire category
            fetch_paper_category(
                args.category,
                category_config,
                workers=args.workers,
                converter=args.converter,
            )

    else:
        logging.error(
            "Must specify --type library or --type papers, or provide library names directly"
        )
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
