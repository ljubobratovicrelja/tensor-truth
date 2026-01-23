"""Chat history and message conversion utilities.

DEPRECATED: These functions are deprecated in favor of ChatHistoryService.
Use tensortruth.services.ChatHistoryService.build_history() instead.
"""

import warnings
from typing import List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole


def build_chat_history(
    session_messages: List[dict], max_messages: Optional[int] = None
) -> List[ChatMessage]:
    """Convert session messages to LlamaIndex ChatMessage format.

    DEPRECATED: Use ChatHistoryService.build_history() instead.

    Args:
        session_messages: List of message dicts from session history
        max_messages: Optional limit on number of messages to include

    Returns:
        List of ChatMessage objects (user and assistant only, no commands)
    """
    warnings.warn(
        "build_chat_history is deprecated. Use ChatHistoryService.build_history() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    chat_messages = []

    for msg in session_messages:
        if msg["role"] == "user":
            chat_messages.append(
                ChatMessage(content=msg["content"], role=MessageRole.USER)
            )
        elif msg["role"] == "assistant":
            chat_messages.append(
                ChatMessage(content=msg["content"], role=MessageRole.ASSISTANT)
            )
        # Skip command messages

    # Apply max_messages limit if specified
    if max_messages is not None and len(chat_messages) > max_messages:
        return chat_messages[-max_messages:]

    return chat_messages if chat_messages else []


def preserve_chat_history(
    session_messages: List[dict], max_messages: int = 4
) -> Optional[List[ChatMessage]]:
    """Extract and preserve recent chat history for engine loading.

    DEPRECATED: Chat history is now passed directly to query methods.
    Use ChatHistoryService.build_history() instead.

    This preserves only the last N messages (default 4 = 2 conversation turns)
    to maintain immediate context without causing hallucinations.

    Args:
        session_messages: List of message dicts from session history
        max_messages: Maximum number of recent messages to preserve (default 4)

    Returns:
        List of ChatMessage objects or None if no valid messages
    """
    warnings.warn(
        "preserve_chat_history is deprecated. Chat history is now passed directly "
        "to query methods. Use ChatHistoryService.build_history() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not session_messages:
        return None

    try:
        # Use the deprecated function internally for backward compatibility
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            chat_messages = build_chat_history(session_messages)

        if not chat_messages:
            return None

        # Preserve only the last N messages
        if len(chat_messages) > max_messages:
            return chat_messages[-max_messages:]
        else:
            return chat_messages

    except Exception as e:
        print(f"Error preserving chat history: {e}")
        return None
