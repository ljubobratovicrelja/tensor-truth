"""Auto-detection utilities for documentation sources.

Provides functions to automatically detect documentation types, inventory URLs,
and CSS selectors for library documentation.
"""

import logging

logger = logging.getLogger(__name__)


def detect_doc_type(doc_root: str) -> str | None:
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


def detect_objects_inv(doc_root: str) -> str | None:
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


def detect_css_selector(doc_root: str) -> str | None:
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
