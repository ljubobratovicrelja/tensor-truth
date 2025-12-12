import re
import requests
from fetch_paper import fetch_and_convert

from build_db import build_module


OLLAMA_API_BASE = "http://localhost:11434/api"


def get_running_models():
    """
    Equivalent to `ollama ps`. Returns list of active models with VRAM usage.
    """
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/ps", timeout=2)
        if response.status_code == 200:
            data = response.json()
            # simplify data for UI
            active = []
            for m in data.get("models", []):
                active.append({
                    "name": m["name"],
                    "size_vram": f"{m.get('size_vram', 0) / 1024**3:.1f} GB",
                    "expires": m.get("expires_at", "Unknown")
                })
            return active
    except Exception:
        return [] # Server likely down
    return []

def stop_model(model_name):
    """
    Forces a model to unload immediately by setting keep_alive to 0.
    """
    try:
        # We send a dummy request with keep_alive=0 to trigger unload
        payload = {
            "model": model_name,
            "keep_alive": 0
        }
        # We use /api/chat as the generic endpoint
        requests.post(f"{OLLAMA_API_BASE}/chat", json=payload, timeout=2)
        return True
    except Exception as e:
        print(f"Failed to stop {model_name}: {e}")
        return False

def parse_thinking_response(raw_text):
    """
    Splits the raw response into (Thought, Answer).
    Handles standard tags <thought>...</thought> and common malformations.
    """
    if not raw_text:
        return None, ""

    # 1. Standard Case
    think_pattern = r"<thought>(.*?)</thought>"
    match = re.search(think_pattern, raw_text, re.DOTALL)
    
    if match:
        thought = match.group(1).strip()
        answer = re.sub(think_pattern, "", raw_text, flags=re.DOTALL).strip()
        return thought, answer

    # 2. Edge Case: Unclosed Tag (Model was cut off or forgot to close)
    if "<thought>" in raw_text and "</thought>" not in raw_text:
        # Treat everything after <thought > as thought, assume answer is empty/cut off
        parts = raw_text.split("<thought>", 1)
        return parts[1].strip(), "..."

    # 3. No Thinking detected
    return None, raw_text

def run_ingestion(arxiv_id):
    """
    Orchestrates the Fetch -> Build pipeline.
    """
    status_log = []
    
    try:
        status_log.append(f"📥 Fetching ArXiv ID: {arxiv_id}...")
        fetch_and_convert(arxiv_id)
        
        status_log.append("📚 Updating Vector Index (this takes a moment)...")
        build_module("papers")
        
        status_log.append(f"✅ Success! {arxiv_id} is now in your library.")
        return True, status_log
    except Exception as e:
        return False, [f"❌ Error: {str(e)}"]

def convert_chat_to_markdown(session):
    """
    Converts session JSON to clean Markdown.
    """
    title = session.get('title', 'Untitled')
    date = session.get('created_at', 'Unknown Date')
    
    md = f"# {title}\n"
    md += f"**Date:** {date}\n\n"
    md += "---\n\n"
    
    for msg in session["messages"]:
        role = msg['role'].upper()
        content = msg['content']
        
        # Clean the markdown export so thoughts don't clutter it (optional)
        # or keep them if you want a full record. Here we separate them.
        thought, clean_content = parse_thinking_response(content)
        
        md += f"### {role}\n\n"
        if thought:
            md += f"> **Thought Process:**\n> {thought.replace('\n', '\n> ')}\n\n"
        
        md += f"{clean_content}\n\n"
        
        if "sources" in msg and msg["sources"]:
            md += "> **Sources:**\n"
            for src in msg["sources"]:
                md += f"> * {src['file']} ({src['score']:.2f})\n"
            md += "\n"
            
    return md