import os
import json
import logging
import requests
import sphobjinv as soi
from concurrent.futures import ThreadPoolExecutor
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin
import re
import argparse

# --- CONFIGURATION ---
DEFAULT_CONFIG = "./library_docs.json"
OUTPUT_BASE_DIR = "./library_docs"
MAX_WORKERS = 20  # Safe number for parallel downloads

logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    """Load library configuration from JSON file."""
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)

def fetch_inventory(config):
    """Downloads and decodes the Sphinx objects.inv file."""
    print(f"Fetching inventory from {config['inventory_url']}...")
    try:
        inv = soi.Inventory(url=config['inventory_url'])
    except Exception as e:
        logging.error(f"Failed to fetch inventory: {e}")
        return []
    
    urls = set()
    # Iterate through all objects (functions, classes, methods)
    for obj in inv.objects:
        # We only want Python API docs, not generic labels or C++ docs
        if obj.domain == 'py' and obj.role in ['function', 'class', 'method', 'module', 'data']:
            # Resolve relative URL to absolute
            full_url = os.path.join(config['doc_root'], obj.uri)
            # Remove anchors (#) to avoid duplicates
            clean_url = full_url.split('#')[0]
            urls.add(clean_url)
    
    print(f"Found {len(urls)} unique API pages.")
    return list(urls)

def fetch_doxygen_urls(config):
    """Extracts documentation URLs from Doxygen index pages."""
    doc_root = config['doc_root']
    index_pages = config.get('index_pages', ['annotated.html', 'modules.html'])
    
    print(f"Fetching Doxygen URLs from {doc_root}...")
    urls = set()
    
    for index_page in index_pages:
        index_url = urljoin(doc_root, index_page)
        print(f"  Parsing {index_page}...")
        
        try:
            resp = requests.get(index_url, timeout=10)
            if resp.status_code != 200:
                logging.warning(f"Failed to fetch {index_url}: {resp.status_code}")
                continue
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Doxygen typically has links in tables or div.contents
            # We look for links to .html files (classes, structs, functions, modules)
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Skip external links, anchors, and non-HTML
                if href.startswith(('http://', 'https://', '#', 'javascript:')):
                    continue
                if not href.endswith('.html'):
                    continue
                
                # Skip index pages themselves and common navigation pages
                if href in ['index.html', 'pages.html', 'annotated.html', 
                           'classes.html', 'modules.html', 'namespaces.html',
                           'files.html', 'examples.html']:
                    continue
                
                # Build full URL
                full_url = urljoin(doc_root, href)
                urls.add(full_url)
        
        except Exception as e:
            logging.error(f"Error parsing {index_url}: {e}")
    
    print(f"Found {len(urls)} unique Doxygen pages.")
    return list(urls)

def clean_doxygen_html(soup):
    """
    Aggressively clean Doxygen HTML to remove noise while preserving semantic content.
    Focuses on keeping: class/function signatures, descriptions, parameters, code blocks.
    """
    # 1. Remove all visual-only elements (diagrams, images, iframes)
    for tag in soup.find_all(['iframe', 'img', 'svg']):
        tag.decompose()
    
    # 2. Remove Doxygen-specific UI elements
    for cls in ['dynheader', 'dyncontent', 'center', 'permalink', 
                'mlabels', 'mlabels-left', 'mlabels-right', 
                'python_language', 'memSeparator']:
        for tag in soup.find_all(class_=cls):
            tag.decompose()
    
    # 3. Remove separator rows (just whitespace)
    for tag in soup.find_all('tr', class_='separator'):
        tag.decompose()
    
    # 4. Remove empty documentation blocks
    for tag in soup.find_all('div', class_='memdoc'):
        if not tag.get_text(strip=True):
            tag.decompose()
    
    # 5. Remove "This browser is not able to show SVG" messages
    for p in soup.find_all('p'):
        text = p.get_text()
        if 'This browser is not able to show SVG' in text or 'try Firefox, Chrome' in text:
            p.decompose()
    
    # 6. Remove footer (everything after first <hr>)
    hr_tags = soup.find_all('hr')
    if hr_tags:
        first_hr = hr_tags[0]
        # Remove all siblings after the hr
        for sibling in list(first_hr.find_next_siblings()):
            sibling.decompose()
        first_hr.decompose()
    
    # 7. Clean up inheritance/collaboration diagram sections
    for tag in soup.find_all('div', class_='dynheader'):
        tag.decompose()
    
    # 8. Simplify member tables - remove layout-only columns
    for table in soup.find_all('table', class_='memberdecls'):
        # Remove groupheader rows with just section titles (we'll keep h2s instead)
        for tr in table.find_all('tr', class_='heading'):
            # Extract the h2 and preserve it, remove the tr
            h2 = tr.find('h2')
            if h2:
                table.insert_before(h2)
            tr.decompose()
    
    # 9. Simplify method documentation tables
    for table in soup.find_all('table', class_='memname'):
        # Extract just the text content, preserve structure but remove excess markup
        # Keep the table but this is already fairly clean
        pass
    
    # 10. Remove empty anchor tags
    for a in soup.find_all('a'):
        if not a.get_text(strip=True) and not a.find('img'):
            a.decompose()
    
    # 11. Remove pure navigation links (those with ../../ paths that won't work locally)
    for a in soup.find_all('a', href=True):
        href = a.get('href')
        if href.startswith('../../') or href.startswith('../'):
            # Replace link with just its text content
            a.replace_with(a.get_text())
    
    # 12. Clean up code includes at the top
    # Keep them but they're useful context
    
    # 13. Remove excessive whitespace-only paragraphs
    for p in soup.find_all('p'):
        if not p.get_text(strip=True):
            p.decompose()
    
    return soup

def url_to_filename(url, doc_root):
    """Clean filename generation."""
    # Remove the base URL
    rel_path = url.replace(doc_root, "").strip("/")
    # Replace slashes/dots with underscores
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', rel_path)
    # Ensure markdown extension
    return f"{clean_name}.md"

def process_url(url, config, output_dir, output_format='markdown', enable_cleanup=False, min_size=0):
    """Download and convert a single URL to markdown or save as HTML, with optional cleanup and size filtering."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return False

        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # Cleanup: remove scripts, styles, nav, footer, sidebar
        for tag in soup(["script", "style", "nav", "footer", "div.sphinxsidebar", "aside"]):
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
            if output_format == 'html':
                final_content = f"<!-- Source: {url} -->\n{str(content)}"
            else:
                # Convert to Markdown (default)
                # The content is already cleaned if cleanup was enabled
                markdown = md(str(content), heading_style="ATX", code_language="python")
                final_content = f"# Source: {url}\n\n{markdown}"
            
            # Check minimum size threshold
            if min_size > 0 and len(final_content) < min_size:
                return 'skipped'  # Return special value to track filtered files
            
            # Save the file
            filename = url_to_filename(url, config['doc_root'])
            if output_format == 'html':
                filename = filename.replace('.md', '.html')
            save_path = os.path.join(output_dir, filename)
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(final_content)
            
            return True
            
    except Exception as e:
        logging.error(f"Error {url}: {e}")
        return False

def scrape_library(library_name, config, max_workers=MAX_WORKERS, output_format='markdown', enable_cleanup=False, min_size=0):
    """Scrape documentation for a single library."""
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"{library_name}_{config['version']}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Scraping: {library_name} v{config['version']}")
    print(f"Doc Type: {config.get('doc_type', 'sphinx')}")
    print(f"Output Format: {output_format}")
    print(f"Cleanup: {'enabled' if enable_cleanup else 'disabled'}")
    if min_size > 0:
        print(f"Min Size Filter: {min_size} characters")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # 1. Get the list of URLs based on documentation type
    doc_type = config.get('doc_type', 'sphinx')  # Default to sphinx for backward compatibility
    
    if doc_type == 'doxygen':
        urls = fetch_doxygen_urls(config)
    elif doc_type == 'sphinx':
        urls = fetch_inventory(config)
    else:
        logging.error(f"Unknown doc_type: {doc_type}. Supported: 'sphinx', 'doxygen'")
        return
    
    if not urls:
        print(f"⚠️  No URLs found for {library_name}")
        return
    
    # 2. Download
    print(f"Downloading {len(urls)} pages...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm for progress bar
        results = list(tqdm(
            executor.map(lambda u: process_url(u, config, output_dir, output_format, enable_cleanup, min_size), urls),
            total=len(urls),
            desc=library_name
        ))
    
    successful = sum(1 for r in results if r is True)
    skipped = sum(1 for r in results if r == 'skipped')
    failed = len(results) - successful - skipped
    
    print(f"\n✅ Successfully downloaded {successful}/{len(urls)} pages")
    if skipped > 0:
        print(f"⏭️  Skipped {skipped} files (below {min_size} chars)")
    if failed > 0:
        print(f"❌ Failed {failed} files")
    print(f"{'='*60}\n")

def list_libraries(config):
    """List all available libraries in config."""
    print("\nAvailable libraries:")
    print("="*60)
    for lib_name, lib_config in config.items():
        print(f"\n{lib_name}")
        print(f"  Version: {lib_config['version']}")
        print(f"  Doc type: {lib_config.get('doc_type', 'sphinx')}")
        print(f"  Doc root: {lib_config['doc_root']}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Scrape Sphinx documentation for Python libraries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape a single library (markdown format by default)
  python scrape_docs.py pytorch
  
  # Scrape in HTML format instead of markdown
  python scrape_docs.py opencv --format html
  
  # Scrape with aggressive cleanup (removes diagrams, navigation, etc.)
  python scrape_docs.py opencv --cleanup
  
  # Filter out nearly empty files (< 128 characters)
  python scrape_docs.py opencv --cleanup --min-size 128
  
  # Scrape to cleaned HTML (best for Doxygen docs like OpenCV)
  python scrape_docs.py opencv --format html --cleanup
  
  # Scrape to cleaned Markdown with size filter
  python scrape_docs.py opencv --format markdown --cleanup --min-size 128
  
  # Scrape multiple libraries
  python scrape_docs.py numpy scipy matplotlib
  
  # Scrape all libraries with cleanup and size filter
  python scrape_docs.py --all --cleanup --min-size 128
  
  # List available libraries
  python scrape_docs.py --list
  
  # Use custom config file
  python scrape_docs.py --config my_libs.json pytorch
  
  # Adjust parallel workers
  python scrape_docs.py --workers 10 numpy
        """
    )
    
    parser.add_argument(
        "libraries",
        nargs='*',
        help="Library names to scrape (e.g., pytorch numpy scipy)"
    )
    
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to JSON config file (default: {DEFAULT_CONFIG})"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape all libraries in config"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available libraries in config"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})"
    )
    
    parser.add_argument(
        "--format",
        choices=["markdown", "html"],
        default="markdown",
        help="Output format for scraped documentation (default: markdown)"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Enable aggressive HTML cleanup to remove navigation, diagrams, and other noise (recommended for Doxygen docs)"
    )
    
    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        metavar="CHARS",
        help="Minimum file size in characters. Files smaller than this will be skipped (e.g., 128 to filter nearly empty files)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if not config:
        print("❌ Failed to load config or config is empty")
        return
    
    # List libraries mode
    if args.list:
        list_libraries(config)
        return
    
    # Determine which libraries to scrape
    if args.all:
        libraries_to_scrape = list(config.keys())
    elif args.libraries:
        libraries_to_scrape = args.libraries
    else:
        parser.print_help()
        return
    
    # Validate library names
    invalid_libs = [lib for lib in libraries_to_scrape if lib not in config]
    if invalid_libs:
        print(f"❌ Unknown libraries: {', '.join(invalid_libs)}")
        print(f"Available: {', '.join(config.keys())}")
        print(f"Use --list to see all available libraries")
        return
    
    # Scrape each library
    print(f"\n{'='*60}")
    print(f"Starting scrape for {len(libraries_to_scrape)} libraries")
    print(f"Workers: {args.workers}")
    print(f"Output Format: {args.format}")
    print(f"Cleanup: {'enabled' if args.cleanup else 'disabled'}")
    if args.min_size > 0:
        print(f"Min Size Filter: {args.min_size} characters")
    print(f"{'='*60}")
    
    for library_name in libraries_to_scrape:
        lib_config = config[library_name]
        scrape_library(library_name, lib_config, max_workers=args.workers, 
                      output_format=args.format, enable_cleanup=args.cleanup,
                      min_size=args.min_size)
    
    print(f"\n{'='*60}")
    print(f"✅ Completed scraping {len(libraries_to_scrape)} libraries")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()