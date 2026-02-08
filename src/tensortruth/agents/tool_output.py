"""Shared helpers for extracting clean text from tool results.

Handles LlamaIndex ``ToolOutput`` objects whose ``.content`` may be
an MCP ``CallToolResult`` containing ``TextContent`` items, as well
as plain strings and other types.
"""

import functools
from typing import Any, Callable


def wrap_mcp_tool_fn(fn: Callable) -> Callable:
    """Wrap an async MCP tool function to return clean text instead of CallToolResult.

    LlamaIndex's ``FunctionTool._parse_tool_output`` doesn't know about MCP's
    ``CallToolResult``, so it falls through to ``str(raw_output)`` which produces
    ugly Python repr like ``meta=None content=[TextContent(...)]``.  This repr
    ends up in ``ToolOutput.blocks`` and is what the LLM sees — confusing it.

    By wrapping the underlying function we ensure it returns a plain string,
    which ``_parse_tool_output`` handles correctly via ``TextBlock(text=...)``.
    """

    @functools.wraps(fn)
    async def _wrapped(**kwargs: Any) -> str:
        result = await fn(**kwargs)
        return extract_tool_text(result)

    return _wrapped


def _truncate(text: str, max_len: int = 60) -> str:
    """Truncate with ellipsis."""
    return text if len(text) <= max_len else text[: max_len - 1] + "\u2026"


def _extract_domain(url: str) -> str:
    """Extract domain from URL, stripping www. prefix."""
    from urllib.parse import urlparse

    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else (netloc or url[:40])
    except Exception:
        return url[:40]


# Key patterns checked against lowercased kwarg key names.
_URL_KEYS = {"url", "page_url", "target_url", "link"}
_URL_LIST_KEYS = {"urls", "pages", "links"}
_QUERY_KEYS = {"query", "queries", "q", "search_query", "search_term"}
_TOPIC_KEYS = {"topic", "subject", "question"}
_NAME_KEYS = {"libraryname", "library", "name", "package"}


def describe_tool_call(tool_name: str, kwargs: dict) -> str:
    """Human-readable one-line description of a tool call.

    Inspects kwarg keys (not tool names) to choose a descriptive verb.
    Priority order, first match wins.
    """
    first_short_string: str | None = None

    for key, value in kwargs.items():
        low = key.lower()

        # Priority 1: single URL
        if low in _URL_KEYS and isinstance(value, str):
            return f"Fetching {_extract_domain(value)}..."

        # Priority 2: list of URLs
        if low in _URL_LIST_KEYS and isinstance(value, list) and value:
            domains = [_extract_domain(u) for u in value[:3] if isinstance(u, str)]
            summary = ", ".join(domains)
            if len(value) > 3:
                summary += "\u2026"
            return f"Fetching {len(value)} pages ({summary})..."

        # Priority 3: query / search term
        if low in _QUERY_KEYS:
            if isinstance(value, str):
                return f"Searching: {_truncate(value)}"
            if isinstance(value, list) and value:
                first = _truncate(str(value[0]), 40)
                if len(value) == 1:
                    return f"Searching: {first}"
                return f"Searching {len(value)} queries: {first}\u2026"

        # Priority 4: topic / subject / question
        if low in _TOPIC_KEYS and isinstance(value, str):
            return f"Looking up: {_truncate(value)}"

        # Priority 5: library / name / package
        if low in _NAME_KEYS and isinstance(value, str):
            return f"Resolving {_truncate(value)}..."

        # Track first short string for priority 6 fallback
        if first_short_string is None and isinstance(value, str) and len(value) <= 80:
            first_short_string = value

    # Priority 6: first short string value
    if first_short_string is not None:
        return f"Calling {tool_name}: {first_short_string}"

    # Priority 7: bare fallback
    return f"Calling {tool_name}..."


def extract_tool_text(raw_data: Any) -> str:
    """Get a clean string from a tool result.

    Traverses through ``ToolOutput → raw_output → CallToolResult →
    [TextContent] → .text`` to extract human-readable text.  Falls back
    to ``str()`` when no structured content is found.
    """
    text = _dig_for_text(raw_data)
    if text is not None:
        return text
    return str(raw_data)


def _dig_for_text(obj: Any, depth: int = 0) -> str | None:
    """Recursively walk through ToolOutput / CallToolResult / TextContent."""
    if depth > 5:
        return None
    if isinstance(obj, str):
        return obj
    # List of content blocks (e.g. [TextContent(...), ...])
    if isinstance(obj, list):
        texts = [
            item.text
            for item in obj
            if hasattr(item, "text") and isinstance(item.text, str)
        ]
        if texts:
            return "\n".join(texts)
    # ToolOutput.raw_output often has the actual MCP result object
    # (while .content may already be str(result) — a lossy conversion)
    if hasattr(obj, "raw_output"):
        result = _dig_for_text(obj.raw_output, depth + 1)
        if result is not None:
            return result
    # Object with .content (ToolOutput, CallToolResult)
    if hasattr(obj, "content"):
        return _dig_for_text(obj.content, depth + 1)
    # Single TextContent-like object with .text
    if hasattr(obj, "text") and isinstance(obj.text, str):
        return obj.text
    return None
