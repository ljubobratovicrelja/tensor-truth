"""PDF processing utilities for Tensor-Truth."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

import pymupdf.layout  # isort: skip

pymupdf.layout.activate()
import pymupdf as fitz  # noqa: E402
import pymupdf4llm  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global marker converter instance (lazy loaded)
MARKER_CONVERTER = None


def clean_filename(title: str) -> str:
    """Sanitize title for file system.

    Args:
        title: Original title string

    Returns:
        Sanitized filename (max 50 characters)
    """
    clean = re.sub(r"[^a-zA-Z0-9]", "_", title)
    return clean[:50]  # Truncate to avoid path length issues


def download_pdf(url: str, output_path: Union[str, Path]) -> bool:
    """Download PDF from URL to output path.

    Args:
        url: PDF URL
        output_path: Destination file path

    Returns:
        True if successful, False otherwise
    """
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


def download_pdf_with_headers(url: str, output_path: Union[str, Path]) -> bool:
    """Download PDF from URL with browser-like headers to bypass bot detection.

    This alternative function adds User-Agent and other headers to mimic
    a real browser request, which helps bypass basic bot detection on
    some academic/research websites.

    Args:
        url: PDF URL
        output_path: Destination file path

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading PDF from {url} (with browser headers)")

    # Browser-like headers to bypass bot detection
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }

    try:
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"✅ Downloaded to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_toc(pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Extract table of contents from PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of dicts with 'title' and 'page' keys (top-level chapters only)
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


def split_pdf_by_pages(
    pdf_path: Union[str, Path],
    start_page: int,
    end_page: int,
    output_path: Union[str, Path],
) -> bool:
    """Extract pages from PDF and save to new PDF.

    Args:
        pdf_path: Source PDF path
        start_page: Start page number (1-indexed)
        end_page: End page number (1-indexed, inclusive)
        output_path: Destination PDF path

    Returns:
        True if successful, False otherwise
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


def get_pdf_page_count(pdf_path: Union[str, Path]) -> int:
    """Get total number of pages in PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Page count (0 on error or if not a PDF)
    """
    try:
        doc = fitz.open(pdf_path)

        # Check if it's actually a PDF (not HTML or other format)
        # PDF format strings can be "PDF", "PDF 1.4", "PDF 1.6", etc.
        doc_format = doc.metadata.get("format", "").upper()
        if doc_format and not doc_format.startswith("PDF"):
            logger.warning(
                f"File is not a PDF (format: {doc.metadata.get('format')}): {pdf_path}"
            )
            doc.close()
            return 0

        count = doc.page_count
        doc.close()
        return count
    except Exception:
        return 0


def pdf_has_extractable_text(
    pdf_path: Union[str, Path], sample_pages: int = 3, min_text_ratio: float = 0.1
) -> bool:
    """Check if PDF has extractable text or is mostly images/scanned.

    Args:
        pdf_path: Path to PDF file
        sample_pages: Number of pages to sample (from start)
        min_text_ratio: Minimum ratio of text chars to total chars

    Returns:
        True if PDF has extractable text, False if mostly images/scanned
    """
    try:
        doc = fitz.open(pdf_path)
        total_chars = 0
        total_text_chars = 0

        # Sample first N pages
        pages_to_check = min(sample_pages, doc.page_count)

        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text()

            # Count total characters (excluding whitespace)
            text_chars = len([c for c in text if not c.isspace()])
            total_text_chars += text_chars

            # Estimate total content (text + image area)
            # Images take up space, so we use page dimensions as proxy
            page_area = page.rect.width * page.rect.height
            total_chars += max(text_chars, page_area / 100)

        doc.close()

        # If we have very little text, it's likely a scanned PDF
        if total_chars == 0:
            return False

        text_ratio = total_text_chars / total_chars
        has_text = text_ratio >= min_text_ratio

        logger.info(
            f"PDF text detection: {total_text_chars} text chars, "
            f"ratio={text_ratio:.2f}, has_text={has_text}"
        )

        return has_text

    except Exception as e:
        logger.warning(f"Failed to detect PDF text content: {e}")
        # On error, assume it needs marker (safer fallback)
        return False


def convert_pdf_to_markdown(
    pdf_path: Union[str, Path], preserve_math: bool = True, converter: str = "pymupdf"
) -> str:
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


def convert_with_marker(pdf_path: Union[str, Path]) -> str:
    """Convert PDF using Marker with GPU acceleration.

    Falls back to pymupdf4llm if Marker is not available.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Markdown text
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


def post_process_math(md_text: Optional[str]) -> Optional[str]:
    """Post-process markdown to improve math rendering.

    Converts Unicode math symbols to LaTeX equivalents.

    Args:
        md_text: Markdown text

    Returns:
        Processed markdown with LaTeX math symbols
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
    return "\n".join(processed_lines)
