import os
import argparse
import arxiv
import pymupdf4llm
import logging
import re

# --- CONFIGURATION ---
OUTPUT_DIR = "./library_docs/papers"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_filename(title):
    """Sanitize title for file system."""
    clean = re.sub(r'[^a-zA-Z0-9]', '_', title)
    return clean[:50]  # Truncate to avoid path length issues

def fetch_and_convert(arxiv_id):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logging.info(f"Fetching metadata for arXiv:{arxiv_id}...")
    
    # 1. Fetch Metadata via Client
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))
    
    safe_title = clean_filename(paper.title)
    pdf_path = os.path.join(OUTPUT_DIR, f"{arxiv_id}.pdf")
    md_path = os.path.join(OUTPUT_DIR, f"{safe_title}.md")

    # 2. Download PDF
    if not os.path.exists(pdf_path):
        logging.info(f"Downloading PDF: {paper.title}")
        paper.download_pdf(dirpath=OUTPUT_DIR, filename=f"{arxiv_id}.pdf")
    else:
        logging.info("PDF already exists. Skipping download.")

    # 3. Convert to Markdown (The Magic Step)
    logging.info("Converting PDF layout to Markdown...")
    # pymupdf4llm handles 2-column layouts and tables reasonably well
    md_text = pymupdf4llm.to_markdown(pdf_path)

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
    
    # Cleanup: Optionally delete the PDF to save space
    # os.remove(pdf_path) 
    
    logging.info(f"✅ Saved: {md_path}")
    print(f"\nSuccessfully added: {paper.title}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and index ArXiv papers.")
    parser.add_argument("ids", nargs="+", help="List of ArXiv IDs (e.g., 1706.03762)")
    args = parser.parse_args()

    for pid in args.ids:
        try:
            fetch_and_convert(pid)
        except Exception as e:
            logging.error(f"Failed to process {pid}: {e}")