"""Chat history cleaning for LLM context efficiency.

Preprocesses chat history to reduce token usage without losing semantic meaning.
All settings are configurable via config.yaml.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

from llama_index.core.llms import ChatMessage

# Emoji pattern covering common Unicode ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001fa6f"  # chess, extended-A
    "\U0001fa70-\U0001faff"  # symbols extended-B
    "\U00002600-\U000026ff"  # misc symbols
    "]+",
    flags=re.UNICODE,
)

# Default filler phrases (regex patterns, case-insensitive)
DEFAULT_FILLER_PHRASES = [
    r"(?i)^(great|good|excellent)\s+(question|point)[!.]*\s*",
    r"(?i)^i['']?d be happy to help[!.]*\s*",
    r"(?i)^let me (think|see|help)[^.]*[.!]*\s*",
    r"(?i)^(sure|certainly|absolutely)[!.,]*\s*",
    r"(?i)if you have any (more |other )?questions[^.]*[.!]*\s*$",
    r"(?i)feel free to ask[^.]*[.!]*\s*$",
    r"(?i)hope this helps[!.]*\s*$",
]


@dataclass
class HistoryCleanerConfig:
    """Configuration for history cleaning operations."""

    enabled: bool = True
    remove_emojis: bool = True
    remove_filler_phrases: bool = True
    normalize_whitespace: bool = True
    collapse_newlines: bool = True
    filler_phrases: List[str] = field(
        default_factory=lambda: list(DEFAULT_FILLER_PHRASES)
    )


def _config_to_cache_key(config: HistoryCleanerConfig) -> tuple:
    """Convert config to hashable tuple for cache key."""
    return (
        config.enabled,
        config.remove_emojis,
        config.remove_filler_phrases,
        config.normalize_whitespace,
        config.collapse_newlines,
        tuple(config.filler_phrases) if config.filler_phrases else (),
    )


@lru_cache(maxsize=256)
def _clean_cached(content: str, config_key: tuple) -> str:
    """LRU-cached cleaning for repeated messages.

    Args:
        content: Raw message content
        config_key: Hashable config representation

    Returns:
        Cleaned content string
    """
    # Reconstruct config from cache key
    (
        enabled,
        remove_emojis,
        remove_filler_phrases,
        normalize_whitespace,
        collapse_newlines,
        filler_phrases_tuple,
    ) = config_key

    if not enabled:
        return content

    result = content

    # Remove emojis
    if remove_emojis:
        result = EMOJI_PATTERN.sub("", result)

    # Remove filler phrases
    if remove_filler_phrases and filler_phrases_tuple:
        for pattern in filler_phrases_tuple:
            try:
                result = re.sub(pattern, "", result, flags=re.MULTILINE)
            except re.error:
                # Skip invalid patterns
                pass

    # Normalize whitespace (multiple inline spaces to single space)
    # Uses lookbehind to preserve leading whitespace (indentation)
    if normalize_whitespace:
        result = re.sub(r"(?<=\S) {2,}", " ", result)

    # Collapse excessive newlines (3+ to 2)
    if collapse_newlines:
        result = re.sub(r"\n{3,}", "\n\n", result)

    # Strip leading/trailing whitespace
    result = result.strip()

    return result


def clean_history_content(
    content: Optional[str],
    config: Optional[HistoryCleanerConfig] = None,
) -> Optional[str]:
    """Clean message content for LLM context efficiency.

    Args:
        content: Raw message content
        config: Cleaning configuration (uses defaults if None)

    Returns:
        Cleaned content string
    """
    if content is None:
        return None
    if not content:
        return content

    if config is None:
        config = HistoryCleanerConfig()

    config_key = _config_to_cache_key(config)
    return _clean_cached(content, config_key)


def clean_chat_history(
    messages: List[ChatMessage],
    config: Optional[HistoryCleanerConfig] = None,
) -> List[ChatMessage]:
    """Apply cleaning to chat history before prompt formatting.

    Creates new ChatMessage objects with cleaned content.
    Original messages are not modified.

    Args:
        messages: List of chat messages
        config: Cleaning configuration

    Returns:
        List of new ChatMessage objects with cleaned content
    """
    if not messages:
        return []

    return [
        ChatMessage(
            role=m.role,
            content=clean_history_content(m.content, config) or "",
        )
        for m in messages
    ]


def clear_cache() -> None:
    """Clear the LRU cache. Useful for testing or memory management."""
    _clean_cached.cache_clear()
