import os
import argparse
import arxiv
import pymupdf4llm
import logging
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- CONFIGURATION ---
ROOT_DIR = "./library_docs"
DEFAULT_CONFIG = "./config/papers.json"
DEFAULT_WORKERS = 4

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Thread-safe statistics tracking
class Statistics:
    def __init__(self):
        self.lock = Lock()
        self.total = 0
        self.processed = 0
        self.skipped = 0
        self.failed = 0
    
    def increment(self, stat_name):
        with self.lock:
            if stat_name == 'total':
                self.total += 1
            elif stat_name == 'processed':
                self.processed += 1
            elif stat_name == 'skipped':
                self.skipped += 1
            elif stat_name == 'failed':
                self.failed += 1
    
    def get_summary(self):
        with self.lock:
            return {
                'total': self.total,
                'processed': self.processed,
                'skipped': self.skipped,
                'failed': self.failed
            }

def clean_filename(title):
    """Sanitize title for file system."""
    clean = re.sub(r'[^a-zA-Z0-9]', '_', title)
    return clean[:50]  # Truncate to avoid path length issues

def load_config(config_path):
    """Load JSON configuration file."""
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return {}

def save_config(config, config_path):
    """Save JSON configuration file."""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info(f"✅ Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save config: {e}")

def paper_already_processed(category, arxiv_id, root_dir=ROOT_DIR):
    """
    Check if a paper has already been downloaded and converted.
    Returns True if both PDF and corresponding MD file exist.
    """
    output_dir = os.path.join(root_dir, category)
    
    if not os.path.exists(output_dir):
        return False
    
    # Check for PDF
    pdf_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
    if not os.path.exists(pdf_path):
        return False
    
    # Check for MD file with arxiv_id in header
    md_files = list(Path(output_dir).glob("*.md"))
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read(500)  # Read first 500 chars (header)
                if arxiv_id in content:
                    return True
        except Exception:
            continue
    
    return False

def add_paper_to_config(config, category, arxiv_id, paper_metadata):
    """Add a paper to the configuration."""
    # Ensure category exists
    if category not in config:
        config[category] = {
            "description": f"Papers in {category}",
            "items": []
        }
    
    # Check if paper already in config
    existing_ids = [item['arxiv_id'] for item in config[category]['items']]
    if arxiv_id in existing_ids:
        logging.info(f"Paper {arxiv_id} already in config for category '{category}'")
        return False
    
    # Add paper
    paper_entry = {
        "title": paper_metadata['title'],
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}"
    }
    config[category]['items'].append(paper_entry)
    logging.info(f"✅ Added {arxiv_id} to config under '{category}'")
    return True

def fetch_and_convert(category, arxiv_id, skip_if_exists=True):
    """
    Fetch and convert a paper. Returns paper metadata dict.
    """
    # Check if already processed
    if skip_if_exists and paper_already_processed(category, arxiv_id):
        logging.info(f"✓ Paper {arxiv_id} already processed. Skipping.")
        return None

    output_dir = os.path.join(ROOT_DIR, category)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info(f"Fetching metadata for arXiv:{arxiv_id}...")
    
    # 1. Fetch Metadata via Client
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))
    
    safe_title = clean_filename(paper.title)
    pdf_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
    md_path = os.path.join(output_dir, f"{safe_title}.md")

    # 2. Download PDF
    if not os.path.exists(pdf_path):
        logging.info(f"Downloading PDF: {paper.title}")
        paper.download_pdf(dirpath=output_dir, filename=f"{arxiv_id}.pdf")
    else:
        logging.info("PDF already exists. Skipping download.")

    # 3. Convert to Markdown (The Magic Step)
    logging.info("Converting PDF layout to Markdown...")
    try:
        # pymupdf4llm handles 2-column layouts and tables reasonably well
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Handle case where conversion returns None or empty string
        if md_text is None:
            logging.warning(f"PDF conversion returned None for {arxiv_id}, using fallback message")
            md_text = "\n\n[PDF content could not be extracted. Please refer to the original PDF file.]\n"
        elif not md_text.strip():
            logging.warning(f"PDF conversion returned empty content for {arxiv_id}")
            md_text = "\n\n[PDF content extraction returned empty. Please refer to the original PDF file.]\n"
            
    except Exception as e:
        logging.warning(f"PDF conversion failed for {arxiv_id}: {e}. Using fallback.")
        md_text = "\n\n[PDF content extraction failed. Please refer to the original PDF file.]\n"

    # 4. Prepend Metadata Header
    # This acts as a "Summary Node" for the RAG system
    header = f"""# Title: {paper.title}
# Authors: {', '.join([a.name for a in paper.authors])}
# Year: {paper.published.year}
# ArXiv ID: {arxiv_id}
# Abstract: 
{paper.summary}

---
"""
    final_content = header + md_text

    # 5. Save to Disk
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(final_content)
    
    # Check if we had to use fallback
    if "could not be extracted" in md_text or "extraction failed" in md_text:
        logging.info(f"✅ Saved with fallback: {md_path}")
        print(f"\n⚠️  Added with PDF extraction issue: {paper.title}")
        print(f"    (Metadata and PDF saved, but content extraction had issues)")
    else:
        logging.info(f"✅ Saved: {md_path}")
        print(f"\nSuccessfully added: {paper.title}")
    
    # Return metadata for config
    return {
        'title': paper.title,
        'arxiv_id': arxiv_id,
        'url': f"https://arxiv.org/abs/{arxiv_id}"
    }

def rebuild_category(config, category, config_path, workers=1):
    """Rebuild/update papers from a category in the config."""
    if category not in config:
        logging.error(f"Category '{category}' not found in config")
        return
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Rebuilding category: {category}")
    logging.info(f"Description: {config[category]['description']}")
    logging.info(f"Workers: {workers}")
    logging.info(f"{'='*60}\n")
    
    items = config[category].get('items', [])
    stats = Statistics()
    stats.total = len(items)
    
    def process_paper(paper):
        """Process a single paper - used for parallel execution."""
        arxiv_id = paper['arxiv_id']
        title = paper['title']
        
        logging.info(f"[{arxiv_id}] {title}")
        
        try:
            result = fetch_and_convert(category, arxiv_id, skip_if_exists=True)
            if result is None:
                stats.increment('skipped')
                return ('skipped', arxiv_id, title)
            else:
                stats.increment('processed')
                return ('processed', arxiv_id, title)
        except Exception as e:
            logging.error(f"✗ Failed to process {arxiv_id}: {e}")
            stats.increment('failed')
            return ('failed', arxiv_id, title)
    
    # Process papers in parallel or sequentially
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_paper, paper) for paper in items]
            
            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Unexpected error in worker: {e}")
    else:
        # Sequential processing
        for paper in items:
            process_paper(paper)
    
    # Print summary
    summary = stats.get_summary()
    logging.info(f"\n{'='*60}")
    logging.info(f"SUMMARY for {category}")
    logging.info(f"{'='*60}")
    logging.info(f"Total papers: {summary['total']}")
    logging.info(f"Newly processed: {summary['processed']}")
    logging.info(f"Skipped (already exist): {summary['skipped']}")
    logging.info(f"Failed: {summary['failed']}")
    logging.info(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Fetch and index ArXiv papers with JSON config management.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add new papers to a category (writes to config, then downloads)
  python fetch_paper.py --config papers_config.json --category dl_foundations --ids 1234.56789 2345.67890
  
  # Rebuild all papers in a category from config
  python fetch_paper.py --config papers_config.json --rebuild dl_foundations
  
  # Rebuild multiple categories
  python fetch_paper.py --config papers_config.json --rebuild dl_foundations vision_2d_generative
  
  # Rebuild all categories
  python fetch_paper.py --config papers_config.json --rebuild-all
  
  # Rebuild with parallel workers (faster)
  python fetch_paper.py --config papers_config.json --rebuild-all --workers 8
  
  # List all categories in config
  python fetch_paper.py --config papers_config.json --list
        """
    )
    
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to JSON configuration file (default: {DEFAULT_CONFIG})"
    )
    
    parser.add_argument(
        "--category",
        help="Category ID to add papers to (used with --ids)"
    )
    
    parser.add_argument(
        "--ids",
        nargs="+",
        help="List of ArXiv IDs to add/fetch (e.g., 1706.03762 2308.04079)"
    )
    
    parser.add_argument(
        "--rebuild",
        nargs="+",
        metavar="CATEGORY",
        help="Rebuild papers from config for specified category/categories"
    )
    
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Rebuild all categories from config"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers for downloading (default: {DEFAULT_WORKERS})"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all categories in config"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # List categories mode
    if args.list:
        if not config:
            print("No categories found in config or config file doesn't exist.")
            return
        
        print("\nAvailable categories:")
        print("=" * 60)
        for cat_id, cat_data in config.items():
            print(f"\n{cat_id}")
            print(f"  Description: {cat_data.get('description', 'N/A')}")
            print(f"  Papers: {len(cat_data.get('items', []))}")
        print("=" * 60)
        return
    
    # Rebuild all mode
    if args.rebuild_all:
        if not config:
            logging.error("No categories found in config")
            return
        
        categories = list(config.keys())
        logging.info(f"Rebuilding all {len(categories)} categories with {args.workers} workers")
        
        for category in categories:
            rebuild_category(config, category, args.config, workers=args.workers)
        return
    
    # Rebuild mode
    if args.rebuild:
        for category in args.rebuild:
            rebuild_category(config, category, args.config, workers=args.workers)
        return
    
    # Add papers mode (original functionality + write to config)
    if args.category and args.ids:
        for arxiv_id in args.ids:
            try:
                # Fetch and convert
                paper_metadata = fetch_and_convert(args.category, arxiv_id, skip_if_exists=False)
                
                # Add to config (if not already there)
                if paper_metadata:
                    add_paper_to_config(config, args.category, arxiv_id, paper_metadata)
                
            except Exception as e:
                logging.error(f"Failed to process {arxiv_id}: {e}")
        
        # Save updated config
        save_config(config, args.config)
        return
    
    # No valid arguments
    parser.print_help()

if __name__ == "__main__":
    main()