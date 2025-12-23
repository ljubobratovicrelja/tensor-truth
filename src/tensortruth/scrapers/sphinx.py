"""Sphinx documentation scraping utilities."""

import logging
import os

import sphobjinv as soi

logger = logging.getLogger(__name__)


def fetch_inventory(config):
    """Download and decode Sphinx objects.inv file.

    Args:
        config: Library configuration dictionary

    Returns:
        List of unique API page URLs
    """
    logger.info(f"Fetching inventory from {config['inventory_url']}...")
    try:
        inv = soi.Inventory(url=config["inventory_url"])
    except Exception as e:
        logger.error(f"Failed to fetch inventory: {e}")
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

    logger.info(f"Found {len(urls)} unique API pages.")
    return list(urls)
