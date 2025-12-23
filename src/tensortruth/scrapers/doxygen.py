"""Doxygen documentation scraping utilities."""

import logging
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def fetch_doxygen_urls(config):
    """Extract documentation URLs from Doxygen index pages.

    Args:
        config: Library configuration dictionary

    Returns:
        List of unique Doxygen documentation page URLs
    """
    doc_root = config["doc_root"]
    index_pages = config.get("index_pages", ["annotated.html", "modules.html"])

    logger.info(f"Fetching Doxygen URLs from {doc_root}...")
    urls = set()

    for index_page in index_pages:
        index_url = urljoin(doc_root, index_page)
        logger.info(f"  Parsing {index_page}...")

        try:
            resp = requests.get(index_url, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch {index_url}: {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.content, "html.parser")

            # Doxygen typically has links in tables or div.contents
            # We look for links to .html files (classes, structs, functions, modules)
            for link in soup.find_all("a", href=True):
                href = link["href"]

                # Skip external links, anchors, and non-HTML
                if href.startswith(("http://", "https://", "#", "javascript:")):
                    continue
                if not href.endswith(".html"):
                    continue

                # Skip index pages themselves and common navigation pages
                if href in [
                    "index.html",
                    "pages.html",
                    "annotated.html",
                    "classes.html",
                    "modules.html",
                    "namespaces.html",
                    "files.html",
                    "examples.html",
                ]:
                    continue

                # Build full URL
                full_url = urljoin(doc_root, href)
                urls.add(full_url)

        except Exception as e:
            logger.error(f"Error parsing {index_url}: {e}")

    logger.info(f"Found {len(urls)} unique Doxygen pages.")
    return list(urls)


def clean_doxygen_html(soup):
    """Aggressively clean Doxygen HTML to remove noise.

    Focuses on keeping class/function signatures, descriptions, parameters,
    and code blocks while removing diagrams, navigation, and visual elements.

    Args:
        soup: BeautifulSoup object

    Returns:
        Cleaned BeautifulSoup object
    """
    # 1. Remove all visual-only elements (diagrams, images, iframes)
    for tag in soup.find_all(["iframe", "img", "svg"]):
        tag.decompose()

    # 2. Remove Doxygen-specific UI elements
    for cls in [
        "dynheader",
        "dyncontent",
        "center",
        "permalink",
        "mlabels",
        "mlabels-left",
        "mlabels-right",
        "python_language",
        "memSeparator",
    ]:
        for tag in soup.find_all(class_=cls):
            tag.decompose()

    # 3. Remove separator rows (just whitespace)
    for tag in soup.find_all("tr", class_="separator"):
        tag.decompose()

    # 4. Remove empty documentation blocks
    for tag in soup.find_all("div", class_="memdoc"):
        if not tag.get_text(strip=True):
            tag.decompose()

    # 5. Remove "This browser is not able to show SVG" messages
    for p in soup.find_all("p"):
        text = p.get_text()
        if (
            "This browser is not able to show SVG" in text
            or "try Firefox, Chrome" in text
        ):
            p.decompose()

    # 6. Remove footer (everything after first <hr>)
    hr_tags = soup.find_all("hr")
    if hr_tags:
        first_hr = hr_tags[0]
        # Remove all siblings after the hr
        for sibling in list(first_hr.find_next_siblings()):
            sibling.decompose()
        first_hr.decompose()

    # 7. Clean up inheritance/collaboration diagram sections
    for tag in soup.find_all("div", class_="dynheader"):
        tag.decompose()

    # 8. Simplify member tables - remove layout-only columns
    for table in soup.find_all("table", class_="memberdecls"):
        # Remove groupheader rows with just section titles (we'll keep h2s instead)
        for tr in table.find_all("tr", class_="heading"):
            # Extract the h2 and preserve it, remove the tr
            h2 = tr.find("h2")
            if h2:
                table.insert_before(h2)
            tr.decompose()

    # 9. Simplify method documentation tables
    for table in soup.find_all("table", class_="memname"):
        # Extract just the text content, preserve structure but remove excess markup
        # Keep the table but this is already fairly clean
        pass

    # 10. Remove empty anchor tags
    for a in soup.find_all("a"):
        if not a.get_text(strip=True) and not a.find("img"):
            a.decompose()

    # 11. Remove pure navigation links (those with ../../ paths that won't work locally)
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if href.startswith("../../") or href.startswith("../"):
            # Replace link with just its text content
            a.replace_with(a.get_text())

    # 12. Clean up code includes at the top
    # Keep them but they're useful context

    # 13. Remove excessive whitespace-only paragraphs
    for p in soup.find_all("p"):
        if not p.get_text(strip=True):
            p.decompose()

    return soup
