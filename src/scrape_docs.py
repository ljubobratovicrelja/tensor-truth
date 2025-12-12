import os
import requests
import logging
import re
import threading
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from concurrent.futures import ThreadPoolExecutor, wait
from collections import deque
from tqdm import tqdm

# --- CONFIGURATION ---
LIBRARY_CONFIG = {
    "pytorch": {
        "version": "2.5",
        "doc_root": "https://pytorch.org/docs/stable/", 
        "entry_point": "https://pytorch.org/docs/stable/genindex.html", 
        "selector": "div[role='main']"
    },
    "numpy": {
        "version": "1.26",
        "doc_root": "https://numpy.org/doc/stable/reference/",
        "entry_point": "https://numpy.org/doc/stable/genindex.html",
        "selector": "section#reference"
    }
}

TARGET_LIB = "pytorch" 
OUTPUT_BASE = "./library_docs"
MAX_WORKERS = min(32, os.cpu_count() * 4) # Cap threads to avoid OS limits

# --- FILTERS ---
# Skip these irrelevant sections to save time and space
EXCLUDE_PATTERNS = [
    "/_modules/",    # Source code view (we want API docs, not raw code)
    "/_static/",     # Assets
    "/_images/",     # Images
    "/_downloads/",  # Downloads
    "search.html",   # Search page
    "genindex.html", # Index page (we use it to start, but don't want to save it)
    "py-modindex.html"
]

# Only allow these extensions (or no extension)
ALLOWED_EXTENSIONS = {".html", ".htm", ""} 

logging.basicConfig(filename='scraper_strict.log', level=logging.INFO, filemode='w')

class StrictScraper:
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.join(OUTPUT_BASE, f"{TARGET_LIB}_{config['version']}")
        
        self.visited = set()
        self.visited_lock = threading.Lock()
        self.queue = deque([config["entry_point"]])
        self.queue_lock = threading.Lock()
        
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
        self.session.mount('https://', adapter)
        self.session.headers.update({"User-Agent": "TensorTruth-Bot/1.0"})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def normalize_url(self, url):
        """
        STRICT normalization:
        1. Strip anchors (#)
        2. Strip query parameters (?) -> This fixes the infinite loop
        3. Remove trailing slashes
        """
        url = url.split('#')[0] # Remove anchor
        url = url.split('?')[0] # Remove query params (CRITICAL FIX)
        return url.rstrip('/')

    def should_scrape(self, url):
        """Pre-flight check before even adding to queue."""
        # 1. Must match domain scope
        if not url.startswith(self.config["doc_root"]):
            return False
            
        # 2. Check exclusions
        for pattern in EXCLUDE_PATTERNS:
            if pattern in url:
                return False

        # 3. Check extension (don't scrape .png, .zip, .txt)
        parsed = urlparse(url)
        path = parsed.path
        ext = os.path.splitext(path)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False

        return True

    def url_to_filename(self, url):
        path = url.replace(self.config["doc_root"], "")
        path = re.sub(r'[^a-zA-Z0-9_\-]', '_', path)
        if not path or path == "_": path = "index"
        return f"{path}.md"

    def process_url(self, url):
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.content, 'html.parser')

            # 1. Harvest Links
            found_links = []
            for a_tag in soup.find_all('a', href=True):
                next_url = urljoin(url, a_tag['href'])
                next_url = self.normalize_url(next_url)
                
                if self.should_scrape(next_url):
                    found_links.append(next_url)

            # 2. Save Content (Skip if it's the index page itself)
            if "genindex" in url:
                return found_links

            selector = self.config.get("selector", "main")
            content = soup.select_one(selector)
            if not content:
                content = soup.find("article") or soup.find("body")

            if content:
                # Cleanup
                for tag in content(["script", "style", "nav", "footer", "a.headerlink"]):
                    tag.decompose()
                
                markdown = md(str(content), heading_style="ATX", code_language="python")
                
                # Metadata Header
                header = f"# Source: {url}\n\n"
                filename = self.url_to_filename(url)
                save_path = os.path.join(self.output_dir, filename)
                
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(header + markdown)
            
            return found_links

        except Exception as e:
            logging.error(f"Failed {url}: {e}")
            return []

    def run(self):
        print(f"--- Starting Strict Scrape: {TARGET_LIB} ---")
        print(f"Scope: {self.config['doc_root']}")
        
        # We start with a progress bar of 1 (the index)
        # It will grow as we find REAL links
        pbar = tqdm(total=1, unit="pg")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            
            while True:
                with self.queue_lock:
                    while self.queue and len(futures) < MAX_WORKERS * 2:
                        url = self.queue.popleft()
                        
                        with self.visited_lock:
                            if url in self.visited:
                                continue
                            self.visited.add(url)
                        
                        fut = executor.submit(self.process_url, url)
                        futures[fut] = url
                        
                        # Update progress bar total
                        pbar.total = len(self.visited) + len(self.queue)
                        pbar.refresh()

                if not futures:
                    break

                done, _ = wait(futures.keys(), return_when='FIRST_COMPLETED')
                
                for fut in done:
                    original_url = futures.pop(fut)
                    pbar.update(1)
                    
                    try:
                        new_links = fut.result()
                        with self.queue_lock:
                            for link in new_links:
                                # Pre-check visited to keep queue clean
                                with self.visited_lock:
                                    if link not in self.visited:
                                        self.queue.append(link)
                    except Exception:
                        pass

        pbar.close()
        print(f"\n--- Complete. {len(self.visited)} pages saved. ---")

if __name__ == "__main__":
    scraper = StrictScraper(LIBRARY_CONFIG[TARGET_LIB])
    scraper.run()