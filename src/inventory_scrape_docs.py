import os
import logging
import requests
import sphobjinv as soi
from concurrent.futures import ThreadPoolExecutor
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

# --- CONFIGURATION ---
LIBRARY_CONFIG = {
    "pytorch": {
        "version": "2.5",
        "doc_root": "https://pytorch.org/docs/stable/",
        "inventory_url": "https://pytorch.org/docs/stable/objects.inv",
        "selector": "div[role='main']"
    },
    "numpy": {
        "version": "1.26",
        "doc_root": "https://numpy.org/doc/stable/reference/",
        "inventory_url": "https://numpy.org/doc/stable/reference/objects.inv",
        "selector": "section#reference"
    }
}

TARGET_LIB = "pytorch"
OUTPUT_DIR = f"./library_docs/{TARGET_LIB}_{LIBRARY_CONFIG[TARGET_LIB]['version']}"
MAX_WORKERS = 20  # Safe number for parallel downloads

logging.basicConfig(level=logging.INFO)

def fetch_inventory(config):
    """Downloads and decodes the Sphinx objects.inv file."""
    print(f"Fetching inventory from {config['inventory_url']}...")
    inv = soi.Inventory(url=config['inventory_url'])
    
    urls = set()
    # Iterate through all objects (functions, classes, methods)
    for obj in inv.objects:
        # We only want Python API docs, not generic labels or C++ docs
        if obj.domain == 'py' and obj.role in ['function', 'class', 'method', 'module']:
            # Resolve relative URL to absolute
            full_url = os.path.join(config['doc_root'], obj.uri)
            # Remove anchors (#) to avoid duplicates like 'func.html#func'
            clean_url = full_url.split('#')[0]
            urls.add(clean_url)
    
    print(f"Found {len(urls)} unique API pages.")
    return list(urls)

def url_to_filename(url, doc_root):
    """Clean filename generation."""
    # Remove the base URL
    rel_path = url.replace(doc_root, "").strip("/")
    # Replace slashes/dots with underscores
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', rel_path)
    # Ensure markdown extension
    return f"{clean_name}.md"

def process_url(url, config):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return False

        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # Cleanup
        for tag in soup(["script", "style", "nav", "footer", "div.sphinxsidebar"]):
            tag.decompose()

        # Extract Main Content
        selector = config.get("selector", "main")
        content = soup.select_one(selector)
        if not content:
            content = soup.find("article") or soup.find("body")

        if content:
            markdown = md(str(content), heading_style="ATX", code_language="python")
            
            # Save
            filename = url_to_filename(url, config['doc_root'])
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"# Source: {url}\n\n" + markdown)
            return True
            
    except Exception as e:
        logging.error(f"Error {url}: {e}")
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    config = LIBRARY_CONFIG[TARGET_LIB]
    
    # 1. Get the list (The Map)
    urls = fetch_inventory(config)
    
    # 2. Download (The Mow)
    print(f"Downloading {len(urls)} pages to {OUTPUT_DIR}...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use tqdm for a progress bar
        list(tqdm(executor.map(lambda u: process_url(u, config), urls), total=len(urls)))

if __name__ == "__main__":
    main()