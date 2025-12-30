"""Book fetching and chapter splitting utilities."""

import logging
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from ..utils.pdf import (
    clean_filename,
    convert_pdf_to_markdown,
    download_pdf_with_headers,
    extract_toc,
    get_pdf_page_count,
    split_pdf_by_pages,
)

logger = logging.getLogger(__name__)


def fetch_book(
    book_name: str,
    book_config: Dict,
    output_base_dir: str,
    converter: str = "marker",
    pages_per_chunk: int = 15,
    max_pages_per_chapter: int = 0,
) -> bool:
    """
    Fetch a book PDF, split by TOC or page chunks, and convert to markdown.

    Workflow based on split_method:
    - "toc": Download PDF → Extract TOC → Split into chapter PDFs → Convert each
    - "manual"/"none": Download PDF → Split by page chunks → Convert each chunk

    Args:
        book_name: Book identifier (e.g., "linear_algebra_cherney")
        book_config: Book configuration from sources.json
        output_base_dir: Base output directory
        converter: "marker" or "pymupdf"
        pages_per_chunk: Pages per chunk for fallback splitting (default: 15)
        max_pages_per_chapter: If > 0, split TOC chapters larger than
            this into sub-chunks (default: 0 = no limit)

    Returns:
        True if successful, False otherwise
    """
    # Create directory with 'book_' prefix to match build_db expectations
    output_dir = Path(output_base_dir) / f"book_{book_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_url = book_config.get("source")
    split_method = book_config.get("split_method", "none")
    title = book_config.get("title", book_name)

    if not source_url:
        logger.error(f"No source URL for {book_name}")
        return False

    # Download PDF
    pdf_path = output_dir / f"{book_name}.pdf"

    if not pdf_path.exists():
        logger.info(f"Downloading: {title}")
        # Use version with browser headers to bypass bot detection
        if not download_pdf_with_headers(source_url, pdf_path):
            return False
    else:
        logger.info(f"PDF already exists: {pdf_path.name}")

    # Validate that we actually have a PDF
    page_count = get_pdf_page_count(pdf_path)
    if page_count == 0:
        logger.error(
            f"Downloaded file is not a valid PDF or is empty. "
            f"The source URL may be incorrect or blocked: {source_url}"
        )
        # Remove invalid file
        if pdf_path.exists():
            pdf_path.unlink()
        return False

    logger.info(f"PDF validated: {page_count} pages")

    # Process based on split_method
    if split_method == "toc":
        success = _process_with_toc_split(
            book_name,
            book_config,
            pdf_path,
            output_dir,
            converter,
            max_pages_per_chapter,
        )
        # If TOC split failed, fall back to page chunks
        if not success:
            logger.warning("TOC split failed, falling back to page chunks")
            return _process_with_page_chunks(
                book_name, book_config, pdf_path, output_dir, converter, pages_per_chunk
            )
        return success
    else:
        # "none" or "manual" - split by page chunks
        return _process_with_page_chunks(
            book_name, book_config, pdf_path, output_dir, converter, pages_per_chunk
        )


def _process_with_toc_split(
    book_name: str,
    book_config: Dict,
    pdf_path: Path,
    output_dir: Path,
    converter: str,
    max_pages_per_chapter: int = 0,
) -> bool:
    """Split book by TOC and convert chapters.

    Args:
        max_pages_per_chapter: If > 0, split chapters larger than this into sub-chunks
    """
    logger.info("Extracting table of contents...")
    chapters = extract_toc(pdf_path)

    if not chapters:
        logger.warning("No TOC found")
        return False

    logger.info(f"Found {len(chapters)} chapters")
    total_pages = get_pdf_page_count(pdf_path)

    # Update book_config items with chapter info
    if "items" not in book_config:
        book_config["items"] = {}

    # Convert each chapter
    success_count = 0
    for i, chapter in enumerate(tqdm(chapters, desc="Converting chapters")):
        chapter_num = i + 1
        chapter_title = chapter["title"]
        start_page = chapter["page"]

        # Determine end page (start of next chapter - 1, or end of book)
        end_page = chapters[i + 1]["page"] - 1 if i < len(chapters) - 1 else total_pages

        # Check if chapter needs to be split into sub-chunks
        chapter_page_count = end_page - start_page + 1
        if max_pages_per_chapter > 0 and chapter_page_count > max_pages_per_chapter:
            # Split large chapter into sub-chunks
            logger.info(
                f"Chapter {chapter_num} ({chapter_page_count} pages) exceeds "
                f"max ({max_pages_per_chapter}), splitting into sub-chunks"
            )
            num_subchunks = (
                chapter_page_count + max_pages_per_chapter - 1
            ) // max_pages_per_chapter

            for subchunk_idx in range(num_subchunks):
                subchunk_num = subchunk_idx + 1
                subchunk_start = start_page + (subchunk_idx * max_pages_per_chapter)
                subchunk_end = min(subchunk_start + max_pages_per_chapter - 1, end_page)

                # Clean chapter title for filename
                clean_title = clean_filename(chapter_title)
                subchunk_filename = (
                    f"chapter_{chapter_num:02d}_part{subchunk_num}_{clean_title}.md"
                )
                subchunk_path = output_dir / subchunk_filename

                # Check if already converted
                if subchunk_path.exists():
                    logger.debug(
                        f"Chapter {chapter_num} part {subchunk_num} already exists, skipping"
                    )
                    success_count += 1
                    continue

                # Split PDF sub-chunk
                subchunk_pdf = (
                    output_dir / f"temp_chapter_{chapter_num}_part{subchunk_num}.pdf"
                )
                if not split_pdf_by_pages(
                    pdf_path, subchunk_start, subchunk_end, subchunk_pdf
                ):
                    logger.warning(
                        f"Failed to split chapter {chapter_num} part {subchunk_num}, skipping"
                    )
                    continue

                # Convert to markdown
                try:
                    md_text = convert_pdf_to_markdown(
                        subchunk_pdf, preserve_math=True, converter=converter
                    )

                    # Add YAML header for metadata extraction
                    title = book_config.get("title")
                    authors = book_config.get("authors", [])
                    source_url = book_config.get("source")

                    with open(subchunk_path, "w", encoding="utf-8") as f:
                        f.write(
                            f"# Title: {chapter_title} (Part {subchunk_num}/{num_subchunks})\n"
                        )
                        f.write(f"# Book: {title}\n")
                        f.write(f"# Authors: {', '.join(authors)}\n")
                        f.write(f"# Source: {source_url}\n")
                        f.write(f"# Chapter: {chapter_num}\n")
                        f.write(f"# Part: {subchunk_num}/{num_subchunks}\n")
                        f.write("\n---\n\n")
                        f.write(md_text)

                    # Update items dict
                    subchunk_id = f"chapter_{chapter_num:02d}_part{subchunk_num}"
                    book_config["items"][subchunk_id] = {
                        "title": f"{chapter_title} (Part {subchunk_num}/{num_subchunks})",
                        "page_start": subchunk_start,
                        "page_end": subchunk_end,
                        "filename": subchunk_filename,
                    }

                    success_count += 1

                    # Clean up temp PDF
                    if subchunk_pdf.exists():
                        subchunk_pdf.unlink()

                except Exception as e:
                    logger.error(
                        f"Failed to convert chapter {chapter_num} part {subchunk_num}: {e}"
                    )
                    if subchunk_pdf.exists():
                        subchunk_pdf.unlink()
                    continue
        else:
            # Regular single chapter conversion
            # Clean chapter title for filename
            clean_title = clean_filename(chapter_title)
            chapter_filename = f"chapter_{chapter_num:02d}_{clean_title}.md"
            chapter_path = output_dir / chapter_filename

            # Check if already converted
            if chapter_path.exists():
                logger.debug(f"Chapter {chapter_num} already exists, skipping")
                success_count += 1
                continue

            # Split PDF chapter
            chapter_pdf = output_dir / f"temp_chapter_{chapter_num}.pdf"
            if not split_pdf_by_pages(pdf_path, start_page, end_page, chapter_pdf):
                logger.warning(f"Failed to split chapter {chapter_num}, skipping")
                continue

            # Convert to markdown
            try:
                md_text = convert_pdf_to_markdown(
                    chapter_pdf, preserve_math=True, converter=converter
                )

                # Add YAML header for metadata extraction
                title = book_config.get("title")
                authors = book_config.get("authors", [])
                source_url = book_config.get("source")

                with open(chapter_path, "w", encoding="utf-8") as f:
                    f.write(f"# Title: {chapter_title}\n")
                    f.write(f"# Book: {title}\n")
                    f.write(f"# Authors: {', '.join(authors)}\n")
                    f.write(f"# Source: {source_url}\n")
                    f.write(f"# Chapter: {chapter_num}\n")
                    f.write("\n---\n\n")
                    f.write(md_text)

                # Update items dict
                chapter_id = f"chapter_{chapter_num:02d}"
                book_config["items"][chapter_id] = {
                    "title": chapter_title,
                    "page_start": start_page,
                    "page_end": end_page,
                    "filename": chapter_filename,
                }

                success_count += 1

                # Clean up temp PDF
                if chapter_pdf.exists():
                    chapter_pdf.unlink()

            except Exception as e:
                logger.error(f"Failed to convert chapter {chapter_num}: {e}")
                if chapter_pdf.exists():
                    chapter_pdf.unlink()
                continue

    logger.info(f"✅ Successfully processed {success_count}/{len(chapters)} chapters")
    return success_count > 0


def _process_with_page_chunks(
    book_name: str,
    book_config: Dict,
    pdf_path: Path,
    output_dir: Path,
    converter: str,
    pages_per_chunk: int,
) -> bool:
    """Split book by fixed page chunks and convert."""
    logger.info(
        f"Splitting book into chunks of {pages_per_chunk} pages "
        "(no TOC or split_method='none'/'manual')"
    )

    total_pages = get_pdf_page_count(pdf_path)
    if total_pages == 0:
        logger.error("Could not determine page count")
        return False

    # Calculate chunks
    num_chunks = (total_pages + pages_per_chunk - 1) // pages_per_chunk
    logger.info(f"Book has {total_pages} pages, splitting into {num_chunks} chunks")

    # Update book_config items
    if "items" not in book_config:
        book_config["items"] = {}

    # Convert each chunk
    success_count = 0
    for chunk_idx in tqdm(range(num_chunks), desc="Converting chunks"):
        chunk_num = chunk_idx + 1
        start_page = chunk_idx * pages_per_chunk + 1
        end_page = min((chunk_idx + 1) * pages_per_chunk, total_pages)

        chunk_filename = f"section_{chunk_num:02d}_pages_{start_page}-{end_page}.md"
        chunk_path = output_dir / chunk_filename

        # Check if already converted
        if chunk_path.exists():
            logger.debug(f"Chunk {chunk_num} already exists, skipping")
            success_count += 1
            continue

        # Split PDF chunk
        chunk_pdf = output_dir / f"temp_chunk_{chunk_num}.pdf"
        if not split_pdf_by_pages(pdf_path, start_page, end_page, chunk_pdf):
            logger.warning(f"Failed to split chunk {chunk_num}, skipping")
            continue

        # Convert to markdown
        try:
            md_text = convert_pdf_to_markdown(
                chunk_pdf, preserve_math=True, converter=converter
            )

            # Add YAML header for metadata extraction
            title = book_config.get("title")
            authors = book_config.get("authors", [])
            source_url = book_config.get("source")

            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(f"# Title: {title} (Pages {start_page}-{end_page})\n")
                f.write(f"# Book: {title}\n")
                f.write(f"# Authors: {', '.join(authors)}\n")
                f.write(f"# Source: {source_url}\n")
                f.write(f"# Section: {chunk_num}\n")
                f.write("\n---\n\n")
                f.write(md_text)

            # Update items dict
            chunk_id = f"section_{chunk_num:02d}"
            book_config["items"][chunk_id] = {
                "title": f"Pages {start_page}-{end_page}",
                "page_start": start_page,
                "page_end": end_page,
                "filename": chunk_filename,
            }

            success_count += 1

            # Clean up temp PDF
            if chunk_pdf.exists():
                chunk_pdf.unlink()

        except Exception as e:
            logger.error(f"Failed to convert chunk {chunk_num}: {e}")
            if chunk_pdf.exists():
                chunk_pdf.unlink()
            continue

    logger.info(f"✅ Successfully processed {success_count}/{num_chunks} chunks")
    return success_count > 0


def fetch_book_category(
    category_name: str,
    config: Dict,
    output_base_dir: str,
    converter: str = "marker",
    pages_per_chunk: int = 15,
    max_pages_per_chapter: int = 0,
) -> None:
    """
    Fetch all books in a category.

    Args:
        category_name: Category name (e.g., "linear_algebra")
        config: Full sources.json config
        output_base_dir: Base output directory
        converter: "marker" or "pymupdf"
        pages_per_chunk: Pages per chunk for fallback splitting
        max_pages_per_chapter: Max pages per TOC chapter before sub-splitting
    """
    # Find all books with matching category
    books = {}
    for name, item_config in config.get("papers", {}).items():
        if (
            item_config.get("type") == "pdf_book"
            and item_config.get("category") == category_name
        ):
            books[name] = item_config

    if not books:
        logger.warning(f"No books found in category: {category_name}")
        return

    logger.info(f"Fetching {len(books)} books in category: {category_name}")

    success_count = 0
    for book_name, book_config in books.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Fetching: {book_config.get('title')}")
        logger.info(f"{'=' * 60}\n")

        try:
            if fetch_book(
                book_name,
                book_config,
                output_base_dir,
                converter,
                pages_per_chunk,
                max_pages_per_chapter,
            ):
                success_count += 1
        except Exception as e:
            logger.error(
                f"Failed to fetch book {book_name} ({book_config.get('title')}): {e}. "
            )
            continue

    logger.info(f"\n✅ Successfully fetched {success_count}/{len(books)} books")
