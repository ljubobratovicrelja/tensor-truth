"""Library documentation scraping functionality.

Handles scraping of library documentation (Sphinx and Doxygen formats).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

from tqdm import tqdm

from .common import process_url
from .doxygen import fetch_doxygen_urls
from .sphinx import fetch_inventory

# Default number of parallel workers (safe for most systems)
DEFAULT_MAX_WORKERS = 20

logger = logging.getLogger(__name__)


def scrape_library(
    library_name: str,
    config: Dict,
    output_base_dir: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    output_format: str = "markdown",
    enable_cleanup: bool = False,
    min_size: int = 0,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> bool:
    """Scrape documentation for a single library.

    Args:
        library_name: Name of the library (e.g., "pytorch_2.9")
        config: Library configuration dictionary containing:
            - version: Library version string
            - type: Documentation type ('sphinx' or 'doxygen', defaults to 'sphinx')
            - Additional scraper-specific config (urls, inventory_url, etc.)
        output_base_dir: Base directory for output (e.g., ~/.tensortruth/library_docs)
        max_workers: Number of parallel workers for downloading (default: 20)
        output_format: Output format - 'markdown' or 'html' (default: 'markdown')
        enable_cleanup: Enable aggressive HTML cleanup (default: False, recommended for Doxygen)
        min_size: Minimum file size in characters to keep (default: 0, no filtering)
        progress_callback: Optional callback function called with (successful, skipped, failed)
            counts after scraping completes

    Returns:
        True if scraping succeeded, False if it failed or found no URLs

    Example:
        >>> config = {
        ...     "version": "2.9",
        ...     "type": "sphinx",
        ...     "inventory_url": "https://pytorch.org/docs/stable/objects.inv"
        ... }
        >>> scrape_library("pytorch_2.9", config, "~/.tensortruth/library_docs")
        True
    """
    # Create directory with 'library_' prefix to match build_db expectations
    # library_name already includes version (e.g., "pytorch_2.9")
    output_dir = os.path.join(output_base_dir, f"library_{library_name}")

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
    doc_type = config.get("type", "sphinx")

    if doc_type == "doxygen":
        urls = fetch_doxygen_urls(config)
    elif doc_type == "sphinx":
        urls = fetch_inventory(config)
    else:
        logger.error(f"Unknown doc_type: {doc_type}. Supported: 'sphinx', 'doxygen'")
        return False

    if not urls:
        logger.warning(f"No URLs found for {library_name}")
        return False

    # 2. Download with progress bar
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

    # 3. Calculate statistics
    successful = sum(1 for r in results if r is True)
    skipped = sum(1 for r in results if r == "skipped")
    failed = len(results) - successful - skipped

    logger.info(f"\n✅ Successfully downloaded {successful}/{len(urls)} pages")
    if skipped > 0:
        logger.info(f"⏭️  Skipped {skipped} files (below {min_size} chars)")
    if failed > 0:
        logger.warning(f"❌ Failed {failed} files")
    logger.info(f"{'=' * 60}\n")

    # 4. Call progress callback if provided
    if progress_callback:
        progress_callback(successful, skipped, failed)

    return True
