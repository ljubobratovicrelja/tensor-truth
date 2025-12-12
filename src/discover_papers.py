import argparse
import logging
import sys
from semanticscholar import SemanticScholar
from fetch_paper import fetch_and_convert

# --- CONFIGURATION ---
TOPIC_MAP = {
    "cs.CV": "Computer Vision Deep Learning", 
    "cs.AI": "Artificial Intelligence Deep Learning",
    "cs.LG": "Machine Learning Architecture",
    "cs.CL": "Natural Language Processing Transformers",
}

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_influential_arxiv_ids(category, limit=10):
    # Initialize with a timeout to prevent infinite hanging
    sch = SemanticScholar(timeout=10)
    query_term = TOPIC_MAP.get(category, category)
    
    print(f"--- Searching Semantic Scholar (Gentle Mode) ---")
    print(f"Query: '{query_term}'")
    
    # LIMITATION: Without an API Key, we cannot fetch 100+ results to sort.
    # We fetch a safe batch (30) by 'Relevance' and pick the best from there.
    search_limit = 30 
    
    try:
        results = sch.search_paper(
            query=query_term, 
            fields=['title', 'externalIds', 'citationCount', 'year'],
            limit=search_limit
            # Removed 'sort' param to avoid warnings/errors
        )
    except Exception as e:
        print(f"\n❌ API Error: {e}")
        print("Tip: You might be rate-limited. Wait 2 minutes and try again.")
        return []

    found_papers = []
    seen_ids = set()

    for item in results:
        if len(found_papers) >= limit:
            break
            
        # Filter for ArXiv availability
        if item.externalIds and 'ArXiv' in item.externalIds:
            arxiv_id = item.externalIds['ArXiv']
            
            if arxiv_id not in seen_ids:
                found_papers.append({
                    "id": arxiv_id,
                    "title": item.title,
                    "citations": item.citationCount if item.citationCount else 0,
                    "year": item.year
                })
                seen_ids.add(arxiv_id)
    
    # Local Sort: We sort the 30 relevant papers we found by citations
    found_papers.sort(key=lambda x: x['citations'], reverse=True)
    
    return found_papers[:limit]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-discover and index influential papers.")
    parser.add_argument("query", help="Search term or Category (e.g., 'Object Detection', cs.CV)")
    parser.add_argument("--n", type=int, default=5, help="Number of papers to download")
    
    args = parser.parse_args()

    # 1. Discovery Phase
    top_papers = get_influential_arxiv_ids(args.query, args.n)
    
    if not top_papers:
        print("\nNo papers found or API blocked.")
        sys.exit(1)

    print(f"\nFound {len(top_papers)} influential papers:")
    for i, p in enumerate(top_papers):
        print(f"{i+1}. [{p['citations']} cites] ({p['year']}) {p['title']} (ID: {p['id']})")

    # 2. Ingestion Phase
    confirm = input("\nProceed to download and index these? [y/N]: ")
    if confirm.lower() == 'y':
        print("\n--- Starting Ingestion ---")
        for p in top_papers:
            try:
                print(f"Processing: {p['title']}...")
                fetch_and_convert(p['id'])
            except Exception as e:
                logging.error(f"Failed {p['id']}: {e}")
        
        print("\n✅ Done. Remember to run: python src/build_db.py papers")
    else:
        print("Operation cancelled.")