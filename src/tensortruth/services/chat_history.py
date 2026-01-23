"""Chat history service for unified history management.

This module provides immutable value objects and a service for chat history
operations. It serves as the single source of truth for history management,
replacing scattered conversion logic across the codebase.

The session JSON file remains the authoritative storage for chat history.
This service handles conversion, limiting, and cleaning operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from tensortruth.app_utils.config_schema import TensorTruthConfig
from tensortruth.app_utils.history_cleaner import (
    HistoryCleanerConfig,
    clean_history_content,
)


@dataclass(frozen=True)
class ChatHistoryMessage:
    """Single immutable chat message.

    Attributes:
        role: Message role - "user", "assistant", or "system"
        content: Message text content
        timestamp: Optional creation timestamp

    Example:
        msg = ChatHistoryMessage(role="user", content="Hello")
    """

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None


@dataclass(frozen=True)
class ChatHistory:
    """Immutable chat history for LLM consumption.

    Uses tuple for hashability (enables caching, required for frozen dataclass).
    Created from session messages via ChatHistoryService, never from engine memory.

    Attributes:
        messages: Tuple of ChatHistoryMessage objects

    Example:
        history = chat_history_service.build_history(session_messages)
        if not history.is_empty:
            llama_msgs = history.to_llama_messages()
            prompt_str = history.to_prompt_string()
    """

    messages: Tuple[ChatHistoryMessage, ...]

    @property
    def is_empty(self) -> bool:
        """True if no messages in history."""
        return len(self.messages) == 0

    def truncated(self, max_messages: int) -> "ChatHistory":
        """Return new ChatHistory with only last N messages.

        Preserves immutability by returning new instance.

        Args:
            max_messages: Maximum number of messages to keep

        Returns:
            New ChatHistory with at most max_messages
        """
        if len(self.messages) <= max_messages:
            return ChatHistory(messages=self.messages)
        return ChatHistory(messages=self.messages[-max_messages:])

    def to_prompt_string(self) -> str:
        """Format for inclusion in prompt templates.

        Returns:
            String like "user: Hello\nassistant: Hi there\n..."
        """
        if not self.messages:
            return ""
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)

    def to_llama_messages(self) -> List[ChatMessage]:
        """Convert to LlamaIndex ChatMessage list.

        Returns mutable list for LlamaIndex API compatibility.

        Returns:
            List of ChatMessage objects
        """
        if not self.messages:
            return []

        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
        }

        return [
            ChatMessage(role=role_map[m.role], content=m.content) for m in self.messages
        ]


class ChatHistoryService:
    """Service for chat history operations.

    This is the single source of truth for history management.
    Always use this service instead of direct conversions.

    All methods are stateless (accept input, return new output).

    Constants:
        MAX_HISTORY_MESSAGES: Hard safety limit (100) to prevent memory issues

    Example:
        service = get_chat_history_service()
        history = service.build_history(session["messages"], max_messages=5)
        cleaned = service.build_cleaned(history)
        llama_msgs = cleaned.to_llama_messages()
    """

    # Hard safety limit to prevent memory issues
    MAX_HISTORY_MESSAGES = 100

    # Valid roles for chat messages
    VALID_ROLES = {"user", "assistant", "system"}

    def __init__(self, config: TensorTruthConfig):
        """Initialize the service with configuration.

        Args:
            config: TensorTruth configuration object
        """
        self.config = config

    def build_history(
        self,
        session_messages: Optional[List[Dict[str, Any]]],
        max_messages: Optional[int] = None,
        apply_cleaning: Optional[bool] = None,
    ) -> ChatHistory:
        """Build ChatHistory from session messages.

        Args:
            session_messages: Raw message dicts from session storage
                Expected format: [{"role": "user", "content": "..."}, ...]
            max_messages: Override max messages limit
                None = use config.rag.max_history_messages
                0 = return empty history (disabled)
                N = keep last N messages
            apply_cleaning: Override history cleaning
                None = use config.history_cleaning.enabled

        Returns:
            Immutable ChatHistory object

        Note:
            - Validates messages, skips malformed (doesn't raise)
            - Enforces MAX_HISTORY_MESSAGES (100) as hard safety limit
            - Filters to user/assistant/system messages only
        """
        # Handle None or empty input
        if not session_messages:
            return ChatHistory(messages=())

        # Check for explicit disable (0 means no history)
        if max_messages == 0:
            return ChatHistory(messages=())

        # Convert valid messages
        valid_messages: List[ChatHistoryMessage] = []

        for msg in session_messages:
            # Skip non-dict entries
            if not isinstance(msg, dict):
                continue

            # Get role and content
            role = msg.get("role")
            content = msg.get("content")

            # Skip if missing required fields
            if role is None or content is None:
                continue

            # Skip non-valid roles (e.g., "command")
            if role not in self.VALID_ROLES:
                continue

            # Convert content to string if needed
            content_str = str(content) if not isinstance(content, str) else content

            valid_messages.append(ChatHistoryMessage(role=role, content=content_str))

        # Apply limits
        # First, apply config/override limit
        effective_limit = max_messages
        if effective_limit is None:
            effective_limit = self.config.rag.max_history_messages

        if effective_limit and len(valid_messages) > effective_limit:
            valid_messages = valid_messages[-effective_limit:]

        # Then apply hard safety limit
        if len(valid_messages) > self.MAX_HISTORY_MESSAGES:
            valid_messages = valid_messages[-self.MAX_HISTORY_MESSAGES :]

        # Create immutable history
        history = ChatHistory(messages=tuple(valid_messages))

        # Apply cleaning if enabled
        should_clean = apply_cleaning
        if should_clean is None:
            should_clean = self.config.history_cleaning.enabled

        if should_clean:
            history = self.build_cleaned(history)

        return history

    def build_cleaned(self, history: ChatHistory) -> ChatHistory:
        """Apply history cleaning transformations.

        Removes emojis, filler phrases, normalizes whitespace per config.
        Returns new ChatHistory instance (preserves immutability).
        Filters out messages that become empty after cleaning.

        Args:
            history: ChatHistory to clean

        Returns:
            New ChatHistory with cleaned content
        """
        if history.is_empty:
            return ChatHistory(messages=())

        # Check if cleaning is enabled
        if not self.config.history_cleaning.enabled:
            return ChatHistory(messages=history.messages)

        # Build cleaner config from app config
        cleaner_config = HistoryCleanerConfig(
            enabled=True,
            remove_emojis=self.config.history_cleaning.remove_emojis,
            remove_filler_phrases=self.config.history_cleaning.remove_filler_phrases,
            normalize_whitespace=self.config.history_cleaning.normalize_whitespace,
            collapse_newlines=self.config.history_cleaning.collapse_newlines,
            filler_phrases=self.config.history_cleaning.filler_phrases,
        )

        # Clean each message
        cleaned_messages: List[ChatHistoryMessage] = []

        for msg in history.messages:
            cleaned_content = clean_history_content(msg.content, cleaner_config)

            # Skip messages that become empty after cleaning
            if not cleaned_content or not cleaned_content.strip():
                continue

            cleaned_messages.append(
                ChatHistoryMessage(
                    role=msg.role,
                    content=cleaned_content,
                    timestamp=msg.timestamp,
                )
            )

        return ChatHistory(messages=tuple(cleaned_messages))
