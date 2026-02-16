"""Title generation utilities using the session's LLM.

Supports two strategies:
- **Tool mode** (`/api/chat` with a `set_title` tool) — preferred when the
  model advertises tool-calling capability.  Gives structured output.
- **Prompt mode** (`/api/generate`) — fallback for models without tool support.

Both paths strip markdown artefacts and enforce a word limit before returning.
If the LLM is unreachable or returns garbage, a cleaned-up truncation of the
raw assistant text is used as the ultimate fallback.
"""

import asyncio
import json
import re
from typing import Optional

import aiohttp

from .logging_config import logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_markdown(text: str) -> str:
    """Strip common markdown syntax to plain text."""
    text = text[:200]
    # Remove code fences
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Remove inline code
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove link syntax [text](url) → text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Remove heading prefixes
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove blockquotes
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Remove list markers (- , * , 1. )
    text = re.sub(r"^[\-\*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"[*_]{1,3}", "", text)
    # Remove quotes and newlines → spaces
    text = text.replace('"', "").replace("'", "").replace("\n", " ").replace("\r", " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_fallback_title(text: str, max_len: int = 60) -> str:
    """Create a clean fallback title from raw text."""
    title = _strip_markdown(text)
    if not title:
        return "New Chat"
    if len(title) <= max_len:
        return title
    # Truncate on word boundary
    truncated = title[:max_len]
    space_idx = truncated.rfind(" ")
    if space_idx > 0:
        truncated = truncated[:space_idx]
    return truncated + ".."


def _clean_title(raw: str, max_words: int = 5) -> Optional[str]:
    """Strip markdown, enforce word limit, return *None* if empty."""
    title = _strip_markdown(raw)
    words = title.split()
    if len(words) > max_words:
        title = " ".join(words[:max_words])
    return title or None


# ---------------------------------------------------------------------------
# Capability check
# ---------------------------------------------------------------------------


async def _check_tool_support_async(model_name: str, base_url: str) -> bool:
    """Check if a model supports tool calling via Ollama /api/show."""
    try:
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{base_url}/api/show",
                json={"model": model_name},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return "tools" in data.get("capabilities", [])
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Strategy: tool mode  (/api/chat  with structured tool call)
# ---------------------------------------------------------------------------

_TITLE_TOOL = {
    "type": "function",
    "function": {
        "name": "set_title",
        "description": "Set the conversation title",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "A concise 2-5 word title summarizing the conversation topic. "
                        "No markdown, no punctuation."
                    ),
                }
            },
            "required": ["title"],
        },
    },
}


async def _generate_title_tool_mode(
    text: str, model_name: str, base_url: str
) -> Optional[str]:
    """Generate a title by asking the model to call the ``set_title`` tool."""
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You must call the set_title tool with a concise 2-5 word title "
                    "for this conversation. No markdown, no punctuation, no quotes."
                ),
            },
            {"role": "user", "content": text[:1000]},
        ],
        "tools": [_TITLE_TOOL],
        "stream": False,
    }

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base_url}/api/chat", json=payload) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

    msg = data.get("message", {})

    # 1. Prefer structured tool-call response
    for tc in msg.get("tool_calls", []):
        args = tc.get("function", {}).get("arguments", {})
        # Some models return arguments as a JSON string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        title = _clean_title(args.get("title", ""))
        if title:
            return title

    # 2. Fall back to plain content (model may ignore the tool)
    return _clean_title(msg.get("content", ""))


# ---------------------------------------------------------------------------
# Strategy: prompt mode  (/api/generate)
# ---------------------------------------------------------------------------


async def _generate_title_prompt_mode(
    text: str, model_name: str, base_url: str
) -> Optional[str]:
    """Generate a title via a plain prompt on ``/api/generate``."""
    prompt = (
        "Generate a 2-4 word title for this text. "
        "MAXIMUM 4 WORDS. No sentences. No punctuation. Just a short label. "
        "Examples: 'RAG Overview', 'Python Basics', 'Database Design'. "
        f"Text: {text[:1000]}"
    )
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 512, "temperature": 0.8},
    }

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base_url}/api/generate", json=payload) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

    return _clean_title(data.get("response", ""))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_smart_title_async(
    text: str,
    model_name: Optional[str] = None,
) -> str:
    """Generate a concise title using the session's model.

    Uses tool-calling mode when the model supports it, otherwise falls back
    to a plain prompt, and finally to a cleaned-up text truncation.

    Args:
        text: The assistant response to derive a title from.
        model_name: Ollama model name (from the session).  When *None* only
            the text-truncation fallback is used.
    """
    if not model_name:
        return _make_fallback_title(text)

    from tensortruth.core.ollama import get_ollama_url

    base_url = get_ollama_url()

    try:
        supports_tools = await _check_tool_support_async(model_name, base_url)

        if supports_tools:
            logger.debug("Using tool mode for title generation with %s", model_name)
            title = await _generate_title_tool_mode(text, model_name, base_url)
        else:
            logger.debug("Using prompt mode for title generation with %s", model_name)
            title = await _generate_title_prompt_mode(text, model_name, base_url)

        if title:
            logger.debug("Title generation success: '%s'", title)
            return title
        else:
            logger.warning("Title generation returned empty result")
    except asyncio.TimeoutError:
        logger.warning("Title generation timeout")
    except aiohttp.ClientError as e:
        logger.error("Connection error during title generation: %s", e)
    except Exception as e:
        logger.error("Title generation error: %s: %s", type(e).__name__, str(e))

    # Ultimate fallback
    logger.info("Using fallback title: '%s..'", text[:30])
    return _make_fallback_title(text)
