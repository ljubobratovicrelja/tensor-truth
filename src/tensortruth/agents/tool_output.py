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
