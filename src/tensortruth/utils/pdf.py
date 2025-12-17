"""PDF processing utilities for Tensor-Truth."""

import logging
import re

import pymupdf as fitz
import pymupdf4llm
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global marker converter instance (lazy loaded)
MARKER_CONVERTER = None


def clean_filename(title):
    """Sanitize title for file system."""
    clean = re.sub(r"[^a-zA-Z0-9]", "_", title)
    return clean[:50]  # Truncate to avoid path length issues


def download_pdf(url, output_path):
    """Download PDF from URL to output_path."""
    logger.info(f"Downloading PDF from {url}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"✅ Downloaded to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
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
            logger.warning("No TOC found in PDF")
            return []

        # Convert to simpler format, filter to top-level chapters only (level 1)
        chapters = []
        for level, title, page in toc:
            if level == 1:  # Only top-level chapters
                chapters.append({"title": title.strip(), "page": page})

        return chapters
    except Exception as e:
        logger.error(f"Failed to extract TOC: {e}")
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
        logger.error(f"Failed to split PDF pages {start_page}-{end_page}: {e}")
        return False


def get_pdf_page_count(pdf_path):
    """Get total number of pages in PDF."""
    try:
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception:
        return 0


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
            logger.warning(f"PDF conversion returned empty content for {pdf_path}")
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
        logger.error(f"PDF conversion failed for {pdf_path}: {e}")
        return (
            "\n\n[PDF content extraction failed. "
            "Please refer to the original PDF file.]\n"
        )


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
        logger.error("Marker/Torch not installed.")
        return convert_pdf_to_markdown(pdf_path, converter="pymupdf")

    try:
        if MARKER_CONVERTER is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                f"Loading Marker models on {device.upper()} (One-time setup)..."
            )

            converter_config = {
                "batch_multiplier": 4,
                "languages": "English",
                "disable_image_extraction": True,  # Don't extract images to disk
            }

            MARKER_CONVERTER = PdfConverter(
                artifact_dict=create_model_dict(device=device), config=converter_config
            )

        logger.info(f"Converting: {pdf_path}")

        # Convert
        rendered = MARKER_CONVERTER(str(pdf_path))
        full_text, _, _ = text_from_rendered(rendered)

        # Remove all image tags like ![](_page_1_Picture_2.jpeg)
        full_text = re.sub(r"!\[.*?\]\(.*?\)", "", full_text)

        return full_text

    except Exception as e:
        logger.error(f"Marker conversion failed: {e}")
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
