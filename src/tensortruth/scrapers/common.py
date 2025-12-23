"""Common utilities shared across documentation scrapers."""

import logging
import os
import re

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from .doxygen import clean_doxygen_html

logger = logging.getLogger(__name__)


def url_to_filename(url, doc_root):
    """Generate clean filename from URL.

    Args:
        url: Source URL
        doc_root: Base documentation URL

    Returns:
        Sanitized filename with .md extension
    """
    # Remove the base URL
    rel_path = url.replace(doc_root, "").strip("/")
    # Replace slashes/dots with underscores
    clean_name = re.sub(r"[^a-zA-Z0-9]", "_", rel_path)
    # Ensure markdown extension
    return f"{clean_name}.md"


def process_url(
    url, config, output_dir, output_format="markdown", enable_cleanup=False, min_size=0
):
    """Download and convert single URL to markdown or HTML.

    Args:
        url: URL to process
        config: Library configuration dictionary
        output_dir: Output directory path
        output_format: Output format ('markdown' or 'html')
        enable_cleanup: Enable aggressive HTML cleanup
        min_size: Minimum file size in characters (skip smaller files)

    Returns:
        True if successful, 'skipped' if filtered, False on error
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
        logger.error(f"Error {url}: {e}")
        return False
