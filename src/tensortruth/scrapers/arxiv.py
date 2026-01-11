"""ArXiv paper fetching utilities."""

import logging
import os

import arxiv
from tqdm import tqdm

from ..utils.pdf import convert_pdf_to_markdown

logger = logging.getLogger(__name__)


def fetch_arxiv_paper(arxiv_id, output_dir, output_format="pdf", converter="marker"):
    """
    Fetch and optionally convert a single ArXiv paper.

    Args:
        arxiv_id: ArXiv paper ID (e.g., "1706.03762")
        output_dir: Directory to save output
        output_format: Output format - 'pdf' (keep as PDF) or 'markdown' (convert to MD)
        converter: Markdown converter if output_format='markdown' ('marker' or 'pymupdf')

    Returns:
        True if successful, False otherwise
    """

    try:
        # Search ArXiv
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())

        # Use ArXiv ID as filename (simple bidirectional mapping)
        pdf_filename = f"{arxiv_id}.pdf"
        md_filename = f"{arxiv_id}.md"

        pdf_path = os.path.join(output_dir, pdf_filename)
        md_path = os.path.join(output_dir, md_filename)

        # Check if already processed (based on output format)
        target_path = md_path if output_format == "markdown" else pdf_path
        target_filename = md_filename if output_format == "markdown" else pdf_filename

        if os.path.exists(target_path):
            logger.info(f"✅ Already exists: {target_filename}")
            return True

        # Download PDF
        logger.info(f"Downloading: {paper.title}")
        paper.download_pdf(dirpath=output_dir, filename=pdf_filename)

        if output_format == "markdown":
            # Convert to markdown
            logger.info(f"Converting to markdown with {converter}...")
            md_text = convert_pdf_to_markdown(
                pdf_path, preserve_math=True, converter=converter
            )

            # Save markdown with metadata
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {paper.title}\n\n")
                f.write(f"**ArXiv ID**: {arxiv_id}\n")
                f.write(f"**Authors**: {', '.join([a.name for a in paper.authors])}\n")
                f.write(f"**Published**: {paper.published.strftime('%Y-%m-%d')}\n\n")
                f.write(f"**Abstract**:\n{paper.summary}\n\n")
                f.write("---\n\n")
                f.write(md_text)

            logger.info(f"✅ Saved markdown: {md_filename}")

            # Remove PDF after conversion
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        else:
            # Keep as PDF
            logger.info(f"✅ Saved PDF: {pdf_filename}")

        return True

    except Exception as e:
        logger.error(f"Failed to fetch ArXiv paper {arxiv_id}: {e}")
        return False


def fetch_paper_category(
    category_name,
    category_config,
    output_base_dir,
    output_format="pdf",
    converter="marker",
):
    """
    Fetch all papers in a category.

    Args:
        category_name: Category name (e.g., "dl_foundations")
        category_config: Category configuration dict from sources.json
        output_base_dir: Base directory for output
        workers: Number of parallel workers (not implemented yet, use 1)
        output_format: Output format - 'pdf' or 'markdown'
        converter: Markdown converter if output_format='markdown' ('marker' or 'pymupdf')
    """
    output_dir = os.path.join(output_base_dir, f"papers_{category_name}")
    os.makedirs(output_dir, exist_ok=True)

    items = category_config.get("items", {})
    if not items:
        logger.warning(f"No items found in category: {category_name}")
        return

    logger.info(f"Fetching {len(items)} papers in category: {category_name}")

    success_count = 0
    for item_key, item in tqdm(items.items(), desc=f"Fetching {category_name}"):
        arxiv_id = item.get("arxiv_id")
        if not arxiv_id:
            logger.warning(
                f"Missing arxiv_id for {item_key}: {item.get('title', 'Unknown')}"
            )
            continue

        if fetch_arxiv_paper(
            arxiv_id, output_dir, output_format=output_format, converter=converter
        ):
            success_count += 1

    logger.info(f"✅ Successfully fetched {success_count}/{len(items)} papers")
