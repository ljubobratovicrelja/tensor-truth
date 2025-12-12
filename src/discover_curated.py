import argparse
import logging
import sys
import time
from semanticscholar import SemanticScholar
from fetch_paper import fetch_and_convert
from tqdm import tqdm

# --- THE "HALL OF FAME" SEARCH TERMS ---
SEARCH_TOPICS = [
    "Neural Networks",
    "Convolutional",
    "Transformer",
    "Transformers",
    "Attention",
    "Generative Adversarial Networks",
    "Deep Reinforcement Learning",
    "Reinforcement Learning",
    "Tracking",
    "Detection",
    "Semantic Segmentation",
    "Visual Transformers",
    "Large Language Models"
]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Silence the noisy HTTP logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("semanticscholar").setLevel(logging.WARNING)

def get_hall_of_fame_papers(limit=10, year_range="2012-2025"):
    sch = SemanticScholar(timeout=10)
    
    candidates = {} # Map ID -> Paper Data (for deduping)
    
    print(f"--- Running Ensemble Search ({len(SEARCH_TOPICS)} topics) ---")
    
    # We use a progress bar because we are making multiple API calls
    for topic in tqdm(SEARCH_TOPICS, unit="topic"):
        try:
            # Be polite to the API to avoid 429 errors
            time.sleep(3.0) 
            
            results = sch.search_paper(
                query=topic, 
                year=year_range, 
                fields_of_study=['Computer Science'],
                fields=['title', 'externalIds', 'citationCount', 'year'],
                limit=limit # This often acts as page size, not total limit
            )
            
            count = 0
            for item in results:
                # MANUAL BRAKE: Stop processing this topic after 'limit' items
                if count >= limit:
                    break
                
                # ArXiv Filter
                if item.externalIds and 'ArXiv' in item.externalIds:
                    arxiv_id = item.externalIds['ArXiv']
                    
                    # Deduplication Logic:
                    # If we haven't seen this ID, add it.
                    # If we HAVE seen it, we simply ignore it (idempotent).
                    if arxiv_id not in candidates:
                        candidates[arxiv_id] = {
                            "id": arxiv_id,
                            "title": item.title,
                            "citations": item.citationCount if item.citationCount else 0,
                            "year": item.year,
                            "topic": topic
                        }
                    count += 1
                    
        except Exception as e:
            # Don't crash the whole search if one topic fails
            pass

    # Convert to list and GLOBAL SORT by citations
    final_list = list(candidates.values())
    final_list.sort(key=lambda x: x['citations'], reverse=True)
    
    return final_list[:limit]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover Hall of Fame ML papers.")
    parser.add_argument("--n", type=int, default=15, help="Number of papers to download")
    args = parser.parse_args()

    # 1. Discovery Phase
    top_papers = get_hall_of_fame_papers(args.n)
    
    if not top_papers:
        print("\nNo papers found.")
        sys.exit(1)

    print(f"\n--- 🏆 ML Hall of Fame (Top {len(top_papers)}) ---")
    for i, p in enumerate(top_papers):
        print(f"{i+1}. [{p['citations']} cites] ({p['year']}) {p['title']}")
        print(f"    ID: {p['id']} | Found via: {p['topic']}")

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