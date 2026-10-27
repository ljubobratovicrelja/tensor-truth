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


def _strip_markdown(text: Optional[str]) -> str:
    """Strip common markdown syntax to plain text.

    Tolerates ``None``: OpenAI-compatible servers return ``content: null``
    when a reasoning model spends its whole token budget on ``reasoning``
    (observed with deepseek-v4.1-flash at max_tokens=30), and that must not
    raise — the title path is best-effort.
    """
    if not text:
        return ""
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


def build_title_source(
    user_text: str, assistant_text: str = "", min_user_words: int = 3
) -> str:
    """Return the text a title should be derived from.

    A title describes what the *user asked*, never what the assistant narrated
    about its own tooling ("I'll search arXiv first.Found both..."). The user's
    first message is therefore the primary source; the assistant reply is only
    appended when the user typed almost nothing useful ("what is this?").
    """
    user = _strip_markdown(user_text or "")
    if not user:
        return _strip_markdown(assistant_text or "")
    if len(user.split()) >= min_user_words:
        return user
    assistant = _strip_markdown(assistant_text or "")
    return f"{user} {assistant}".strip() if assistant else user


def make_fallback_title(text: str, max_len: int = 60) -> str:
    """Create a clean fallback title from raw text (public alias)."""
    return _make_fallback_title(text, max_len=max_len)


def _make_fallback_title(text: str, max_len: int = 60) -> str:
    """Create a clean fallback title from raw text."""
    title = _strip_markdown(text)
    # Drop a leading slash-command token ("/arxiv 1234.5678" -> "1234.5678")
    title = re.sub(r"^/\w+\s*", "", title)
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


def _clean_title(raw: Optional[str], max_words: int = 5) -> Optional[str]:
    """Strip markdown, enforce word limit, return *None* if empty."""
    title = _strip_markdown(raw).rstrip(".,;:")
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
    text: str, model_name: str, base_url: str, context_window: int
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
        "options": {"num_ctx": context_window, "num_predict": 256},
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
    text: str, model_name: str, base_url: str, context_window: int
) -> Optional[str]:
    """Generate a title via a plain prompt on ``/api/generate``."""
    prompt = (
        "Generate a 2-4 word title for what the user is asking about below. "
        "MAXIMUM 4 WORDS. No sentences. No punctuation. Just a short label. "
        "Examples: 'RAG Overview', 'Python Basics', 'Database Design'. "
        f"User message: {text[:1000]}"
    )
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": context_window, "temperature": 0.8, "num_predict": 256},
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


#: Token budget for the title call. Reasoning models spend this budget on
#: hidden ``reasoning`` first, so 30 tokens (the old value) produced
#: ``content: null`` every time and the caller silently fell back to a
#: truncated prompt. 512 clears the observed 164-token reasoning preamble.
TITLE_MAX_TOKENS = 512
TITLE_RETRY_MAX_TOKENS = 2048


def _choice_text(choices) -> tuple[Optional[str], Optional[str]]:
    """Return (content, finish_reason) from an OpenAI-style ``choices`` list.

    ``content`` is ``None`` for a reasoning-only completion; that is a
    normal response, not an error.
    """
    if not choices or not isinstance(choices, list):
        return None, None
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content")
    return (content if isinstance(content, str) else None), choice.get("finish_reason")


async def _generate_title_openai_mode(
    text: str,
    model_name: str,
    base_url: str,
    api_key: str,
    context_window: int,
) -> Optional[str]:
    """Generate a title via OpenAI-compatible ``/chat/completions``.

    Retries once with a larger budget when the model burned the whole
    allowance on reasoning and returned no content.
    """
    headers: dict = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/chat/completions"
    timeout = aiohttp.ClientTimeout(total=30)

    for max_tokens in (TITLE_MAX_TOKENS, TITLE_RETRY_MAX_TOKENS):
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 2-5 word title for this conversation. "
                        "No markdown, no punctuation, no quotes. Just a short label."
                    ),
                },
                {"role": "user", "content": text[:1000]},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    return None
                try:
                    data = await resp.json()
                except Exception:
                    return None

        if not isinstance(data, dict):
            return None
        content, finish_reason = _choice_text(data.get("choices"))
        title = _clean_title(content)
        if title:
            return title
        if content is not None or finish_reason != "length":
            # Empty/garbage answer that was not a token-budget cut: retrying
            # with more room cannot help.
            return None
        logger.debug(
            "Title completion returned no content (finish=length) at "
            "max_tokens=%s; retrying with %s",
            max_tokens,
            TITLE_RETRY_MAX_TOKENS,
        )

    return None


async def generate_smart_title_async(
    text: str,
    model_name: Optional[str] = None,
    context_window: int = 16384,
    provider_id: Optional[str] = None,
) -> str:
    """Generate a concise title using the session's model.

    Uses tool-calling mode when the model supports it, otherwise falls back
    to a plain prompt, and finally to a cleaned-up text truncation.

    Args:
        text: The user's first message (see :func:`build_title_source`). The
            assistant reply is only a last-resort adjunct — titles derived from
            the assistant's own narration describe its tooling, not the topic.
        model_name: Model name (from the session).  When *None* only
            the text-truncation fallback is used.
        context_window: Context window size — passed as ``num_ctx`` to Ollama
            so the model is not reloaded with a different context size.
        provider_id: Provider ID. Defaults to Ollama.
    """
    if not model_name:
        return _make_fallback_title(text)

    # Resolve provider to determine which path to use
    from tensortruth.core.providers import ProviderRegistry, resolve_model_from_params

    model_ref = resolve_model_from_params({"provider_id": provider_id}, model_name)

    # ``resolve_model`` invents an Ollama reference when it cannot match the
    # model anywhere. If no such provider is actually configured, every call
    # would burn its timeout on a refused connection — fall back immediately.
    try:
        registry = ProviderRegistry.get_instance()
        configured = registry.get_provider(model_ref.provider_id) is not None
    except Exception:
        configured = True
    if not configured:
        logger.warning(
            "Title generation skipped: model '%s' resolved to unconfigured "
            "provider '%s' (base_url=%s)",
            model_name,
            model_ref.provider_id,
            model_ref.base_url,
        )
        return _make_fallback_title(text)

    try:
        if model_ref.provider_type in ("openai_compatible", "llama_cpp"):
            # OpenAI-compatible path
            logger.debug(
                "Using OpenAI-compatible mode for title generation with %s",
                model_name,
            )
            title = await _generate_title_openai_mode(
                text,
                model_ref.model_name,
                model_ref.base_url,
                model_ref.api_key,
                context_window,
            )
        else:
            # Ollama path (existing logic)
            base_url = model_ref.base_url
            supports_tools = await _check_tool_support_async(model_name, base_url)

            if supports_tools:
                logger.debug("Using tool mode for title generation with %s", model_name)
                title = await _generate_title_tool_mode(
                    text, model_name, base_url, context_window
                )
            else:
                logger.debug(
                    "Using prompt mode for title generation with %s", model_name
                )
                title = await _generate_title_prompt_mode(
                    text, model_name, base_url, context_window
                )

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
