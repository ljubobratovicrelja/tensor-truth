"""Core utility functions for Tensor-Truth."""

import re

from tensortruth.build_db import build_module
from tensortruth.fetch_paper import fetch_and_convert_paper, paper_already_processed


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


def run_ingestion(category, arxiv_id):
    """
    Orchestrates the Fetch -> Build pipeline.
    """
    status_log = []

    try:
        status_log.append(f"📥 Fetching ArXiv ID: {arxiv_id}...")
        if paper_already_processed(category, arxiv_id):
            status_log.append("⚠️ Paper already processed. Skipping fetch.")
        else:
            fetch_and_convert_paper(category, arxiv_id)
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
    title = session.get("title", "Untitled")
    date = session.get("created_at", "Unknown Date")

    md = f"# {title}\n"
    md += f"**Date:** {date}\n\n"
    md += "---\n\n"

    for msg in session["messages"]:
        role = msg["role"].upper()
        content = msg["content"]

        # Clean the markdown export so thoughts don't clutter it (optional)
        # or keep them if you want a full record. Here we separate them.
        thought, clean_content = parse_thinking_response(content)

        md += f"### {role}\n\n"
        if thought:
            formatted_thought = thought.replace("\n", "\n> ")
            md += f"> **Thought Process:**\n> {formatted_thought}\n\n"

        md += f"{clean_content}\n\n"

        if "sources" in msg and msg["sources"]:
            md += "> **Sources:**\n"
            for src in msg["sources"]:
                md += f"> * {src['file']} ({src['score']:.2f})\n"
            md += "\n"

    return md
