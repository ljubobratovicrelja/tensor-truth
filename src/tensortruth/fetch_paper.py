import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import arxiv
import pymupdf as fitz
import pymupdf4llm
import requests

# --- CONFIGURATION ---
ROOT_DIR = "./library_docs"
DEFAULT_CONFIG = "./config/papers.json"
DEFAULT_WORKERS = 4

# We use a global variable to cache the model so we don't reload it 25 times
MARKER_CONVERTER = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Thread-safe statistics tracking
class Statistics:
    def __init__(self):
        self.lock = Lock()
        self.total = 0
        self.processed = 0
        self.skipped = 0
        self.failed = 0

    def increment(self, stat_name):
        with self.lock:
            if stat_name == "total":
                self.total += 1
            elif stat_name == "processed":
                self.processed += 1
            elif stat_name == "skipped":
                self.skipped += 1
            elif stat_name == "failed":
                self.failed += 1

    def get_summary(self):
        with self.lock:
            return {
                "total": self.total,
                "processed": self.processed,
                "skipped": self.skipped,
                "failed": self.failed,
            }


def clean_filename(title):
    """Sanitize title for file system."""
    clean = re.sub(r"[^a-zA-Z0-9]", "_", title)
    return clean[:50]  # Truncate to avoid path length issues


def load_config(config_path):
    """Load JSON configuration file."""
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found: {config_path}")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return {}


def save_config(config, config_path):
    """Save JSON configuration file."""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info(f"✅ Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save config: {e}")


def detect_category_type(category_data):
    """
    Detect if a category contains papers or books.
    Returns 'papers' or 'books'.
    """
    if not category_data.get("items"):
        return "papers"  # default

    # Check first item for distinguishing features
    first_item = category_data["items"][0]

    # Books have 'source' field instead of 'arxiv_id'
    if "source" in first_item:
        return "books"
    elif "arxiv_id" in first_item:
        return "papers"

    return "papers"  # default


def download_pdf(url, output_path):
    """Download PDF from URL to output_path."""
    logging.info(f"Downloading PDF from {url}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"✅ Downloaded to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return False


def extract_toc(pdf_path):
    """
    Extract Table of Contents from PDF.
    Returns list of dicts: [{'title': str, 'page': int}, ...]
    """
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()  # Returns list of [level, title, page]
        doc.close()

        if not toc:
            logging.warning("No TOC found in PDF")
            return []

        # Convert to simpler format, filter to top-level chapters only (level 1)
        chapters = []
        for level, title, page in toc:
            if level == 1:  # Only top-level chapters
                chapters.append({"title": title.strip(), "page": page})

        return chapters
    except Exception as e:
        logging.error(f"Failed to extract TOC: {e}")
        return []


def split_pdf_by_pages(pdf_path, start_page, end_page, output_path):
    """
    Extract pages from PDF and save to new PDF.
    Pages are 1-indexed (as humans count them).
    """
    try:
        doc = fitz.open(pdf_path)

        # PyMuPDF uses 0-based indexing
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_page - 1, to_page=end_page - 1)
        new_doc.save(output_path)

        new_doc.close()
        doc.close()

        return True
    except Exception as e:
        logging.error(f"Failed to split PDF pages {start_page}-{end_page}: {e}")
        return False


def convert_pdf_to_markdown(pdf_path, preserve_math=True, converter="pymupdf"):
    """
    Convert PDF to markdown with optional better math preservation.

    Args:
        pdf_path: Path to PDF file
        preserve_math: If True, attempt to preserve mathematical formulas
        converter: 'pymupdf' (default, fast) or 'marker' (better math, slower)

    Returns:
        Markdown text
    """
    if converter == "marker":
        return convert_with_marker(pdf_path)

    # Default: pymupdf4llm
    try:
        # Use page_chunks for better handling of large documents
        md_text = pymupdf4llm.to_markdown(
            pdf_path, page_chunks=True, write_images=False
        )

        # If chunked, join the results
        if isinstance(md_text, list):
            md_text = "\n\n".join(
                [
                    chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                    for chunk in md_text
                ]
            )

        if md_text is None or not md_text.strip():
            logging.warning(f"PDF conversion returned empty content for {pdf_path}")
            md_text = (
                "\n\n[PDF content extraction failed. "
                "Please refer to the original PDF file.]\n"
            )

        # Post-process for better math rendering if requested
        if (
            preserve_math
            and md_text
            and "[PDF content extraction failed" not in md_text
        ):
            md_text = post_process_math(md_text)

        return md_text

    except Exception as e:
        logging.error(f"PDF conversion failed for {pdf_path}: {e}")
        return "\n\n[PDF content extraction failed. Please refer to the original PDF file.]\n"


def convert_with_marker(pdf_path):
    """
    Convert PDF using Marker with GPU acceleration.
    """
    global MARKER_CONVERTER

    try:
        import torch
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError:
        logging.error("Marker/Torch not installed.")
        return convert_pdf_to_markdown(pdf_path, converter="pymupdf")

    try:
        if MARKER_CONVERTER is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(
                f"Loading Marker models on {device.upper()} (One-time setup)..."
            )

            converter_config = {
                "batch_multiplier": 4,
                "languages": "English",
                "disable_image_extraction": True,  # <--- NEW: Don't extract images to disk
            }

            MARKER_CONVERTER = PdfConverter(
                artifact_dict=create_model_dict(device=device), config=converter_config
            )

        logging.info(f"Converting: {pdf_path}")

        # Convert
        rendered = MARKER_CONVERTER(str(pdf_path))
        full_text, _, _ = text_from_rendered(rendered)

        # <--- NEW: Regex to remove all image tags like ![](_page_1_Picture_2.jpeg)
        full_text = re.sub(r"!\[.*?\]\(.*?\)", "", full_text)

        return full_text

    except Exception as e:
        logging.error(f"Marker conversion failed: {e}")
        return convert_pdf_to_markdown(pdf_path, converter="pymupdf")


def post_process_math(md_text):
    """
    Post-process markdown to improve math rendering for ChromaDB/RAG.

    Converts Unicode math symbols to LaTeX equivalents and wraps in $ delimiters.
    """
    if not md_text:
        return md_text

    # Map of Unicode math symbols to LaTeX
    math_symbols = {
        "×": r"\times",
        "÷": r"\div",
        "≤": r"\leq",
        "≥": r"\geq",
        "≠": r"\neq",
        "≈": r"\approx",
        "∞": r"\infty",
        "∑": r"\sum",
        "∏": r"\prod",
        "∫": r"\int",
        "√": r"\sqrt",
        "∂": r"\partial",
        "∇": r"\nabla",
        "∈": r"\in",
        "∉": r"\notin",
        "⊂": r"\subset",
        "⊆": r"\subseteq",
        "∪": r"\cup",
        "∩": r"\cap",
        "→": r"\to",
        "⇒": r"\Rightarrow",
        "⇔": r"\Leftrightarrow",
        "∀": r"\forall",
        "∃": r"\exists",
        "α": r"\alpha",
        "β": r"\beta",
        "γ": r"\gamma",
        "δ": r"\delta",
        "ε": r"\epsilon",
        "θ": r"\theta",
        "λ": r"\lambda",
        "μ": r"\mu",
        "π": r"\pi",
        "σ": r"\sigma",
        "τ": r"\tau",
        "φ": r"\phi",
        "ω": r"\omega",
        "Δ": r"\Delta",
        "Σ": r"\Sigma",
        "Ω": r"\Omega",
        "±": r"\pm",
        "∓": r"\mp",
        "°": r"^\circ",
    }

    # Process line by line to detect math contexts
    lines = md_text.split("\n")
    processed_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip already processed lines (already have $ or $$)
        if stripped.startswith("$") or "$$" in line:
            processed_lines.append(line)
            continue

        # Check if line has math symbols
        has_math = any(sym in line for sym in math_symbols.keys())

        if has_math:
            # Replace Unicode symbols with LaTeX
            for unicode_sym, latex_sym in math_symbols.items():
                line = line.replace(unicode_sym, latex_sym)

        processed_lines.append(line)

    return "\n".join(processed_lines)


def get_pdf_page_count(pdf_path):
    """Get total number of pages in PDF."""
    try:
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception:
        return 0


# ============================================================================
# ARXIV PAPER HANDLING (Original functionality)
# ============================================================================


def paper_already_processed(category, arxiv_id, root_dir=ROOT_DIR):
    """Check if a paper has already been downloaded and converted."""
    output_dir = os.path.join(root_dir, category)

    if not os.path.exists(output_dir):
        return False

    # Check for PDF
    pdf_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
    if not os.path.exists(pdf_path):
        return False

    # Check for MD file with arxiv_id in header
    md_files = list(Path(output_dir).glob("*.md"))
    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read(500)
                if arxiv_id in content:
                    return True
        except Exception:
            continue

    return False


def fetch_and_convert_paper(
    category, arxiv_id, skip_if_exists=True, converter="pymupdf"
):
    """Fetch and convert an arXiv paper with optional Marker support."""
    if skip_if_exists and paper_already_processed(category, arxiv_id):
        logging.info(f"✓ Paper {arxiv_id} already processed. Skipping.")
        return None

    output_dir = os.path.join(ROOT_DIR, category)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Fetching metadata for arXiv:{arxiv_id}...")

    # Fetch Metadata via Client
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))

    safe_title = clean_filename(paper.title)
    pdf_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
    md_path = os.path.join(output_dir, f"{safe_title}.md")

    # Download PDF
    if not os.path.exists(pdf_path):
        logging.info(f"Downloading PDF: {paper.title}")
        paper.download_pdf(dirpath=output_dir, filename=f"{arxiv_id}.pdf")
    else:
        logging.info("PDF already exists. Skipping download.")

    # Convert to Markdown
    logging.info(f"Converting PDF layout to Markdown (converter: {converter})...")
    md_text = convert_pdf_to_markdown(pdf_path, converter=converter)

    # Prepend Metadata Header
    header = f"""# Title: {paper.title}
# Authors: {", ".join([a.name for a in paper.authors])}
# Year: {paper.published.year}
# ArXiv ID: {arxiv_id}
# Abstract: 
{paper.summary}

---
"""
    final_content = header + md_text

    # Save to Disk
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(final_content)

    if "extraction failed" in md_text:
        logging.info(f"✅ Saved with fallback: {md_path}")
        print(f"\n⚠️  Added with PDF extraction issue: {paper.title}")
    else:
        logging.info(f"✅ Saved: {md_path}")
        print(f"\nSuccessfully added: {paper.title}")

    return {
        "title": paper.title,
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


# ============================================================================
# BOOK/PDF HANDLING (New functionality)
# ============================================================================


def book_already_processed(category, book_title, root_dir=ROOT_DIR):
    """Check if a book has already been processed."""
    output_dir = os.path.join(root_dir, category)

    if not os.path.exists(output_dir):
        return False

    safe_title = clean_filename(book_title)

    # Check if any markdown files exist for this book
    md_pattern = f"{safe_title}*.md"
    md_files = list(Path(output_dir).glob(md_pattern))

    return len(md_files) > 0


def fetch_and_convert_book(
    category, book_item, skip_if_exists=True, converter="pymupdf"
):
    """
    Fetch and convert a book/PDF with various splitting options.

    book_item structure:
    {
        "title": "Book Title",
        "source": "http://example.com/book.pdf" or "/local/path/book.pdf",
        "split_method": "toc" | "manual" | "none" (default: "none"),
        "converter": "pymupdf" | "marker" (optional, default from category or 'pymupdf'),
        "chapters": [  # Only for split_method="manual"
            {"name": "Chapter 1", "pages": [1, 50]},
            {"name": "Chapter 2", "pages": [51, 100]}
        ]
    }
    """
    title = book_item["title"]
    source = book_item["source"]
    split_method = book_item.get("split_method", "none")

    # Get converter from book item, or use the passed default
    item_converter = book_item.get("converter", converter)

    if skip_if_exists and book_already_processed(category, title):
        logging.info(f"✓ Book '{title}' already processed. Skipping.")
        return None

    output_dir = os.path.join(ROOT_DIR, category)
    os.makedirs(output_dir, exist_ok=True)

    safe_title = clean_filename(title)

    # Download or copy PDF
    pdf_path = os.path.join(output_dir, f"{safe_title}.pdf")

    if source.startswith("http://") or source.startswith("https://"):
        if not os.path.exists(pdf_path):
            if not download_pdf(source, pdf_path):
                return None
        else:
            logging.info(f"PDF already exists: {pdf_path}")
    else:
        # Local file
        if not os.path.exists(source):
            logging.error(f"Local file not found: {source}")
            return None

        if not os.path.exists(pdf_path):
            import shutil

            shutil.copy2(source, pdf_path)
            logging.info(f"Copied local PDF to {pdf_path}")

    logging.info(f"Processing book: {title}")
    logging.info(f"Split method: {split_method}")
    logging.info(f"Converter: {item_converter}")

    # Process based on split method
    if split_method == "none":
        # No splitting - convert entire PDF
        logging.info("Converting entire PDF to markdown...")
        md_text = convert_pdf_to_markdown(pdf_path, converter=item_converter)

        header = f"""# Title: {title}
# Source: {source}

---
"""
        final_content = header + md_text

        md_path = os.path.join(output_dir, f"{safe_title}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        logging.info(f"✅ Saved: {md_path}")
        print(f"\nSuccessfully added book: {title}")

    elif split_method == "toc":
        # Split by Table of Contents
        logging.info("Extracting Table of Contents...")
        chapters = extract_toc(pdf_path)

        if not chapters:
            logging.warning("No TOC found, falling back to no splitting")
            return fetch_and_convert_book(
                category,
                {**book_item, "split_method": "none"},
                skip_if_exists=False,
                converter=item_converter,
            )

        logging.info(f"Found {len(chapters)} chapters")

        # Add end page for last chapter
        total_pages = get_pdf_page_count(pdf_path)

        for i, chapter in enumerate(chapters):
            start_page = chapter["page"]
            end_page = (
                chapters[i + 1]["page"] - 1 if i < len(chapters) - 1 else total_pages
            )

            logging.info(
                f"Processing: {chapter['title']} (pages {start_page}-{end_page})"
            )

            # Extract chapter pages to temp PDF
            temp_pdf = os.path.join(output_dir, f"temp_chapter_{i}.pdf")
            if not split_pdf_by_pages(pdf_path, start_page, end_page, temp_pdf):
                continue

            # Convert to markdown
            md_text = convert_pdf_to_markdown(temp_pdf, converter=item_converter)

            # Clean up temp PDF
            try:
                os.remove(temp_pdf)
            except Exception as e:
                logging.warning(f"Failed to remove temp PDF: {e}")

            # Save chapter markdown
            chapter_safe_name = clean_filename(chapter["title"])
            md_filename = f"{safe_title}_{i + 1:02d}_{chapter_safe_name}.md"
            md_path = os.path.join(output_dir, md_filename)

            header = f"""# Book: {title}
# Chapter: {chapter["title"]}
# Pages: {start_page}-{end_page}
# Source: {source}

---
"""
            final_content = header + md_text

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            logging.info(f"✅ Saved chapter: {md_path}")

        print(f"\nSuccessfully added book with {len(chapters)} chapters: {title}")

    elif split_method == "manual":
        # Split by manually defined chapters
        chapters = book_item.get("chapters", [])

        if not chapters:
            logging.error("Manual split method requires 'chapters' list")
            return None

        logging.info(f"Processing {len(chapters)} manually defined chapters")

        for i, chapter in enumerate(chapters):
            chapter_name = chapter["name"]
            start_page, end_page = chapter["pages"]

            logging.info(f"Processing: {chapter_name} (pages {start_page}-{end_page})")

            # Extract chapter pages to temp PDF
            temp_pdf = os.path.join(output_dir, f"temp_chapter_{i}.pdf")
            if not split_pdf_by_pages(pdf_path, start_page, end_page, temp_pdf):
                continue

            # Convert to markdown
            md_text = convert_pdf_to_markdown(temp_pdf, converter=item_converter)

            # Clean up temp PDF
            try:
                os.remove(temp_pdf)
            except Exception:
                pass

            # Save chapter markdown
            chapter_safe_name = clean_filename(chapter_name)
            md_filename = f"{safe_title}_{i + 1:02d}_{chapter_safe_name}.md"
            md_path = os.path.join(output_dir, md_filename)

            header = f"""# Book: {title}
# Chapter: {chapter_name}
# Pages: {start_page}-{end_page}
# Source: {source}

---
"""
            final_content = header + md_text

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            logging.info(f"✅ Saved chapter: {md_path}")

        print(f"\nSuccessfully added book with {len(chapters)} chapters: {title}")

    return {"title": title, "source": source, "split_method": split_method}
    """
    Fetch and convert a book/PDF with various splitting options.
    
    book_item structure:
    {
        "title": "Book Title",
        "source": "http://example.com/book.pdf" or "/local/path/book.pdf",
        "split_method": "toc" | "manual" | "none" (default: "none"),
        "chapters": [  # Only for split_method="manual"
            {"name": "Chapter 1", "pages": [1, 50]},
            {"name": "Chapter 2", "pages": [51, 100]}
        ]
    }
    """
    title = book_item["title"]
    source = book_item["source"]
    split_method = book_item.get("split_method", "none")

    if skip_if_exists and book_already_processed(category, title):
        logging.info(f"✓ Book '{title}' already processed. Skipping.")
        return None

    output_dir = os.path.join(ROOT_DIR, category)
    os.makedirs(output_dir, exist_ok=True)

    safe_title = clean_filename(title)

    # Download or copy PDF
    pdf_path = os.path.join(output_dir, f"{safe_title}.pdf")

    if source.startswith("http://") or source.startswith("https://"):
        if not os.path.exists(pdf_path):
            if not download_pdf(source, pdf_path):
                return None
        else:
            logging.info(f"PDF already exists: {pdf_path}")
    else:
        # Local file
        if not os.path.exists(source):
            logging.error(f"Local file not found: {source}")
            return None

        if not os.path.exists(pdf_path):
            import shutil

            shutil.copy2(source, pdf_path)
            logging.info(f"Copied local PDF to {pdf_path}")

    logging.info(f"Processing book: {title}")
    logging.info(f"Split method: {split_method}")

    # Process based on split method
    if split_method == "none":
        # No splitting - convert entire PDF
        logging.info("Converting entire PDF to markdown...")
        md_text = convert_pdf_to_markdown(pdf_path)

        header = f"""# Title: {title}
# Source: {source}

---
"""
        final_content = header + md_text

        md_path = os.path.join(output_dir, f"{safe_title}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        logging.info(f"✅ Saved: {md_path}")
        print(f"\nSuccessfully added book: {title}")

    elif split_method == "toc":
        # Split by Table of Contents
        logging.info("Extracting Table of Contents...")
        chapters = extract_toc(pdf_path)

        if not chapters:
            logging.warning("No TOC found, falling back to no splitting")
            return fetch_and_convert_book(
                category, {**book_item, "split_method": "none"}, skip_if_exists=False
            )

        logging.info(f"Found {len(chapters)} chapters")

        # Add end page for last chapter
        total_pages = get_pdf_page_count(pdf_path)

        for i, chapter in enumerate(chapters):
            start_page = chapter["page"]
            end_page = (
                chapters[i + 1]["page"] - 1 if i < len(chapters) - 1 else total_pages
            )

            logging.info(
                f"Processing: {chapter['title']} (pages {start_page}-{end_page})"
            )

            # Extract chapter pages to temp PDF
            temp_pdf = os.path.join(output_dir, f"temp_chapter_{i}.pdf")
            if not split_pdf_by_pages(pdf_path, start_page, end_page, temp_pdf):
                continue

            # Convert to markdown
            md_text = convert_pdf_to_markdown(temp_pdf)

            # Clean up temp PDF
            try:
                os.remove(temp_pdf)
            except Exception:
                pass

            # Save chapter markdown
            chapter_safe_name = clean_filename(chapter["title"])
            md_filename = f"{safe_title}_{i + 1:02d}_{chapter_safe_name}.md"
            md_path = os.path.join(output_dir, md_filename)

            header = f"""# Book: {title}
# Chapter: {chapter["title"]}
# Pages: {start_page}-{end_page}
# Source: {source}

---
"""
            final_content = header + md_text

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            logging.info(f"✅ Saved chapter: {md_path}")

        print(f"\nSuccessfully added book with {len(chapters)} chapters: {title}")

    elif split_method == "manual":
        # Split by manually defined chapters
        chapters = book_item.get("chapters", [])

        if not chapters:
            logging.error("Manual split method requires 'chapters' list")
            return None

        logging.info(f"Processing {len(chapters)} manually defined chapters")

        for i, chapter in enumerate(chapters):
            chapter_name = chapter["name"]
            start_page, end_page = chapter["pages"]

            logging.info(f"Processing: {chapter_name} (pages {start_page}-{end_page})")

            # Extract chapter pages to temp PDF
            temp_pdf = os.path.join(output_dir, f"temp_chapter_{i}.pdf")
            if not split_pdf_by_pages(pdf_path, start_page, end_page, temp_pdf):
                continue

            # Convert to markdown
            md_text = convert_pdf_to_markdown(temp_pdf)

            # Clean up temp PDF
            try:
                os.remove(temp_pdf)
            except Exception:
                pass

            # Save chapter markdown
            chapter_safe_name = clean_filename(chapter_name)
            md_filename = f"{safe_title}_{i + 1:02d}_{chapter_safe_name}.md"
            md_path = os.path.join(output_dir, md_filename)

            header = f"""# Book: {title}
# Chapter: {chapter_name}
# Pages: {start_page}-{end_page}
# Source: {source}

---
"""
            final_content = header + md_text

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(final_content)

            logging.info(f"✅ Saved chapter: {md_path}")

        print(f"\nSuccessfully added book with {len(chapters)} chapters: {title}")

    return {"title": title, "source": source, "split_method": split_method}


# ============================================================================
# CATEGORY PROCESSING
# ============================================================================


def rebuild_category(config, category, config_path, workers=1, converter="pymupdf"):
    """Rebuild/update items from a category in the config with optional converter."""
    if category not in config:
        logging.error(f"Category '{category}' not found in config")
        return

    category_type = detect_category_type(config[category])

    # Check for category-level converter override
    category_converter = config[category].get("converter", converter)

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Rebuilding category: {category}")
    logging.info(f"Type: {category_type}")
    logging.info(f"Description: {config[category]['description']}")
    logging.info(f"Workers: {workers}")
    logging.info(f"Converter: {category_converter}")
    logging.info(f"{'=' * 60}\n")

    items = config[category].get("items", [])
    stats = Statistics()
    stats.total = len(items)

    if category_type == "papers":

        def process_item(paper):
            arxiv_id = paper["arxiv_id"]
            title = paper["title"]

            logging.info(f"[{arxiv_id}] {title}")

            try:
                result = fetch_and_convert_paper(
                    category,
                    arxiv_id,
                    skip_if_exists=True,
                    converter=category_converter,
                )
                if result is None:
                    stats.increment("skipped")
                    return ("skipped", arxiv_id, title)
                else:
                    stats.increment("processed")
                    return ("processed", arxiv_id, title)
            except Exception as e:
                logging.error(f"✗ Failed to process {arxiv_id}: {e}")
                stats.increment("failed")
                return ("failed", arxiv_id, title)

    else:  # books

        def process_item(book):
            title = book["title"]

            logging.info(f"[BOOK] {title}")

            try:
                result = fetch_and_convert_book(
                    category, book, skip_if_exists=True, converter=category_converter
                )
                if result is None:
                    stats.increment("skipped")
                    return ("skipped", title, title)
                else:
                    stats.increment("processed")
                    return ("processed", title, title)
            except Exception as e:
                logging.error(f"✗ Failed to process {title}: {e}")
                stats.increment("failed")
                return ("failed", title, title)

    # Process items in parallel or sequentially
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_item, item) for item in items]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Unexpected error in worker: {e}")
    else:
        for item in items:
            process_item(item)

    # Print summary
    summary = stats.get_summary()
    logging.info(f"\n{'=' * 60}")
    logging.info(f"SUMMARY for {category}")
    logging.info(f"{'=' * 60}")
    logging.info(f"Total items: {summary['total']}")
    logging.info(f"Newly processed: {summary['processed']}")
    logging.info(f"Skipped (already exist): {summary['skipped']}")
    logging.info(f"Failed: {summary['failed']}")
    logging.info(f"{'=' * 60}\n")


def add_paper_to_config(config, category, arxiv_id, paper_metadata):
    """Add a paper to the configuration."""
    if category not in config:
        config[category] = {"description": f"Papers in {category}", "items": []}

    existing_ids = [item.get("arxiv_id") for item in config[category]["items"]]
    if arxiv_id in existing_ids:
        logging.info(f"Paper {arxiv_id} already in config for category '{category}'")
        return False

    paper_entry = {
        "title": paper_metadata["title"],
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }
    config[category]["items"].append(paper_entry)
    logging.info(f"✅ Added {arxiv_id} to config under '{category}'")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and index ArXiv papers and PDF books with JSON config management.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add new papers to a category
  tensor-truth-papers --config papers.json --category dl_foundations --ids 1234.56789 2345.67890
  
  # Rebuild all papers in a category from config
  tensor-truth-papers --config papers.json --rebuild dl_foundations
  
  # Rebuild with Marker converter (better math extraction)
  tensor-truth-papers --config papers.json --rebuild linear_algebra_books --converter marker
  
  # Rebuild multiple categories
  tensor-truth-papers --config papers.json --rebuild dl_foundations linear_algebra_books
  
  # Rebuild all categories
  tensor-truth-papers --config papers.json --rebuild-all
  
  # Rebuild with parallel workers (faster) and Marker
  tensor-truth-papers --config papers.json --rebuild-all --workers 8 --converter marker
  
  # List all categories in config
  tensor-truth-papers --config papers.json --list

Converters:
  pymupdf - Fast, good for text-heavy documents (default)
  marker  - Slower but much better math extraction (requires: pip install marker-pdf)
           First run downloads ~5GB of models
  
Books are defined in the JSON config with the following structure:
  {
    "category_name": {
      "description": "Category description",
      "converter": "marker",  // Optional: override converter for this category
      "items": [
        {
          "title": "Book Title",
          "source": "https://example.com/book.pdf" or "/local/path.pdf",
          "split_method": "none" | "toc" | "manual",
          "converter": "marker",  // Optional: override converter for this book
          "chapters": [  // Only for split_method="manual"
            {"name": "Chapter 1", "pages": [1, 50]},
            {"name": "Chapter 2", "pages": [51, 100]}
          ]
        }
      ]
    }
  }
        """,
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to JSON configuration file (default: {DEFAULT_CONFIG})",
    )

    parser.add_argument(
        "--category", help="Category ID to add papers to (used with --ids)"
    )

    parser.add_argument(
        "--ids",
        nargs="+",
        help="List of ArXiv IDs to add/fetch (e.g., 1706.03762 2308.04079)",
    )

    parser.add_argument(
        "--rebuild",
        nargs="+",
        metavar="CATEGORY",
        help="Rebuild items from config for specified category/categories",
    )

    parser.add_argument(
        "--rebuild-all", action="store_true", help="Rebuild all categories from config"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers for downloading (default: {DEFAULT_WORKERS})",
    )

    parser.add_argument(
        "--converter",
        choices=["pymupdf", "marker"],
        default="pymupdf",
        help=(
            "PDF converter: 'pymupdf' (fast, default) or 'marker' "
            "(better math, requires marker-pdf)"
        ),
    )

    parser.add_argument(
        "--list", action="store_true", help="List all categories in config"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # List categories mode
    if args.list:
        if not config:
            print("No categories found in config or config file doesn't exist.")
            return

        print("\nAvailable categories:")
        print("=" * 60)
        for cat_id, cat_data in config.items():
            cat_type = detect_category_type(cat_data)
            print(f"\n{cat_id} [{cat_type}]")
            print(f"  Description: {cat_data.get('description', 'N/A')}")
            print(f"  Items: {len(cat_data.get('items', []))}")
        print("=" * 60)
        return

    # Rebuild all mode
    if args.rebuild_all:
        if not config:
            logging.error("No categories found in config")
            return

        categories = list(config.keys())
        logging.info(
            f"Rebuilding all {len(categories)} categories with {args.workers} "
            f"workers (converter: {args.converter})"
        )

        for category in categories:
            rebuild_category(
                config,
                category,
                args.config,
                workers=args.workers,
                converter=args.converter,
            )
        return

    # Rebuild mode
    if args.rebuild:
        for category in args.rebuild:
            rebuild_category(
                config,
                category,
                args.config,
                workers=args.workers,
                converter=args.converter,
            )
        return

    # Add papers mode
    if args.category and args.ids:
        for arxiv_id in args.ids:
            try:
                paper_metadata = fetch_and_convert_paper(
                    args.category, arxiv_id, skip_if_exists=False
                )

                if paper_metadata:
                    add_paper_to_config(config, args.category, arxiv_id, paper_metadata)

            except Exception as e:
                logging.error(f"Failed to process {arxiv_id}: {e}")

        save_config(config, args.config)
        return

    # No valid arguments
    parser.print_help()


if __name__ == "__main__":
    main()
