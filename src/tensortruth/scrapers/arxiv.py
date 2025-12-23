"""ArXiv paper fetching utilities."""

import logging
import os

from tqdm import tqdm

# Optional dependencies for paper fetching
try:
    import arxiv

    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

from ..utils.pdf import clean_filename, convert_pdf_to_markdown

logger = logging.getLogger(__name__)


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
        logger.error(
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
            logger.info(f"✅ Already processed: {md_filename}")
            return True

        # Download PDF
        logger.info(f"Downloading: {paper.title}")
        paper.download_pdf(dirpath=output_dir, filename=pdf_filename)

        # Convert to markdown
        logger.info(f"Converting to markdown with {converter}...")
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

        logger.info(f"✅ Saved: {md_filename}")

        # Optionally remove PDF to save space
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        return True

    except Exception as e:
        logger.error(f"Failed to fetch ArXiv paper {arxiv_id}: {e}")
        return False


def fetch_paper_category(
    category_name, category_config, output_base_dir, workers=1, converter="pymupdf"
):
    """
    Fetch all papers in a category.

    Args:
        category_name: Category name (e.g., "dl_foundations")
        category_config: Category configuration dict from sources.json
        output_base_dir: Base directory for output
        workers: Number of parallel workers (not implemented yet, use 1)
        converter: PDF converter ('pymupdf' or 'marker')
    """
    output_dir = os.path.join(output_base_dir, category_name)
    os.makedirs(output_dir, exist_ok=True)

    items = category_config.get("items", [])
    if not items:
        logger.warning(f"No items found in category: {category_name}")
        return

    logger.info(f"Fetching {len(items)} papers in category: {category_name}")

    success_count = 0
    for item in tqdm(items, desc=f"Fetching {category_name}"):
        arxiv_id = item.get("arxiv_id")
        if not arxiv_id:
            logger.warning(f"Missing arxiv_id for item: {item.get('title', 'Unknown')}")
            continue

        if fetch_arxiv_paper(arxiv_id, output_dir, converter=converter):
            success_count += 1

    logger.info(f"✅ Successfully fetched {success_count}/{len(items)} papers")
