"""Chat-related utility functions for Tensor-Truth."""

import re
from typing import Dict, Optional, Tuple


def parse_thinking_response(raw_text: Optional[str]) -> Tuple[Optional[str], str]:
    """Split response into thought and answer sections.

    Handles standard <thought>...</thought> tags and common malformations.

    Args:
        raw_text: Raw model response text

    Returns:
        Tuple of (thought, answer) where thought may be None
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


def convert_latex_delimiters(text: Optional[str]) -> Optional[str]:
    r"""
    Converts LaTeX math delimiters from standard LaTeX format to Streamlit format.

    Converts:
    - \(...\) to $...$ (inline math)
    - \[...\] to $$...$$ (display math)

    Args:
        text: String containing LaTeX expressions with standard delimiters

    Returns:
        String with Streamlit-compatible LaTeX delimiters
    """
    if not text:
        return text

    # Convert display math \[...\] to $$...$$
    # Use DOTALL flag to match across newlines
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # Convert inline math \(...\) to $...$
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", text, flags=re.DOTALL)

    return text


def convert_chat_to_markdown(session: Dict) -> str:
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
