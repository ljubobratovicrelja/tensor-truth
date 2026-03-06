"""Unit tests for ChatHistoryService.

Tests follow TDD approach - written before implementation.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest
from llama_index.core.base.llms.types import MessageRole

# =============================================================================
# ChatHistoryMessage Tests
# =============================================================================


@pytest.mark.unit
def test_chat_history_message_is_immutable():
    """ChatHistoryMessage should be frozen (immutable)."""
    from tensortruth.services.chat_history import ChatHistoryMessage

    msg = ChatHistoryMessage(role="user", content="Hello")

    with pytest.raises(Exception):  # FrozenInstanceError
        msg.content = "Changed"


@pytest.mark.unit
def test_chat_history_message_with_timestamp():
    """ChatHistoryMessage should accept optional timestamp."""
    from tensortruth.services.chat_history import ChatHistoryMessage

    ts = datetime.now()
    msg = ChatHistoryMessage(role="assistant", content="Hi there", timestamp=ts)

    assert msg.role == "assistant"
    assert msg.content == "Hi there"
    assert msg.timestamp == ts


@pytest.mark.unit
def test_chat_history_message_default_timestamp_is_none():
    """ChatHistoryMessage timestamp should default to None."""
    from tensortruth.services.chat_history import ChatHistoryMessage

    msg = ChatHistoryMessage(role="user", content="Hello")

    assert msg.timestamp is None


# =============================================================================
# ChatHistory Tests
# =============================================================================


@pytest.mark.unit
def test_chat_history_is_immutable():
    """ChatHistory should be frozen (immutable)."""
    from tensortruth.services.chat_history import ChatHistory, ChatHistoryMessage

    msg = ChatHistoryMessage(role="user", content="Hello")
    history = ChatHistory(messages=(msg,))

    with pytest.raises(Exception):  # FrozenInstanceError
        history.messages = ()


@pytest.mark.unit
def test_chat_history_is_hashable():
    """ChatHistory should be hashable (uses tuple)."""
    from tensortruth.services.chat_history import ChatHistory, ChatHistoryMessage

    msg = ChatHistoryMessage(role="user", content="Hello")
    history = ChatHistory(messages=(msg,))

    # Should not raise - hashable objects can be used in sets
    history_set = {history}
    assert len(history_set) == 1


@pytest.mark.unit
def test_chat_history_is_empty_true():
    """is_empty should return True for empty history."""
    from tensortruth.services.chat_history import ChatHistory

    history = ChatHistory(messages=())

    assert history.is_empty is True


@pytest.mark.unit
def test_chat_history_is_empty_false():
    """is_empty should return False for non-empty history."""
    from tensortruth.services.chat_history import ChatHistory, ChatHistoryMessage

    msg = ChatHistoryMessage(role="user", content="Hello")
    history = ChatHistory(messages=(msg,))

    assert history.is_empty is False


@pytest.mark.unit
def test_chat_history_truncated_returns_new_instance():
    """truncated() should return new ChatHistory, not mutate."""
    from tensortruth.services.chat_history import ChatHistory, ChatHistoryMessage

    msgs = tuple(
        ChatHistoryMessage(role="user", content=f"Message {i}") for i in range(5)
    )
    history = ChatHistory(messages=msgs)

    truncated = history.truncated(3)

    # Should be different object
    assert truncated is not history
    # Original should be unchanged
    assert len(history.messages) == 5


@pytest.mark.unit
def test_chat_history_truncated_keeps_last_n():
    """truncated(n) should keep only last n messages."""
    from tensortruth.services.chat_history import ChatHistory, ChatHistoryMessage

    msgs = tuple(
        ChatHistoryMessage(role="user", content=f"Message {i}") for i in range(5)
    )
    history = ChatHistory(messages=msgs)

    truncated = history.truncated(3)

    assert len(truncated.messages) == 3
    # Should keep messages 2, 3, 4 (the last 3)
    assert truncated.messages[0].content == "Message 2"
    assert truncated.messages[1].content == "Message 3"
    assert truncated.messages[2].content == "Message 4"


@pytest.mark.unit
def test_chat_history_truncated_less_than_n():
    """truncated(n) with less than n messages should return all."""
    from tensortruth.services.chat_history import ChatHistory, ChatHistoryMessage

    msgs = tuple(
        ChatHistoryMessage(role="user", content=f"Message {i}") for i in range(2)
    )
    history = ChatHistory(messages=msgs)

    truncated = history.truncated(5)

    assert len(truncated.messages) == 2


@pytest.mark.unit
def test_chat_history_to_prompt_string():
    """to_prompt_string should format as 'role: content' lines."""
    from tensortruth.services.chat_history import ChatHistory, ChatHistoryMessage

    msgs = (
        ChatHistoryMessage(role="user", content="Hello"),
        ChatHistoryMessage(role="assistant", content="Hi there"),
        ChatHistoryMessage(role="user", content="How are you?"),
    )
    history = ChatHistory(messages=msgs)

    result = history.to_prompt_string()

    assert result == "user: Hello\nassistant: Hi there\nuser: How are you?"


@pytest.mark.unit
def test_chat_history_to_prompt_string_empty():
    """to_prompt_string on empty history should return empty string."""
    from tensortruth.services.chat_history import ChatHistory

    history = ChatHistory(messages=())

    assert history.to_prompt_string() == ""


@pytest.mark.unit
def test_chat_history_to_llama_messages():
    """to_llama_messages should return List[ChatMessage]."""
    from tensortruth.services.chat_history import ChatHistory, ChatHistoryMessage

    msgs = (
        ChatHistoryMessage(role="user", content="Hello"),
        ChatHistoryMessage(role="assistant", content="Hi there"),
    )
    history = ChatHistory(messages=msgs)

    llama_msgs = history.to_llama_messages()

    assert len(llama_msgs) == 2
    assert llama_msgs[0].role == MessageRole.USER
    assert llama_msgs[0].content == "Hello"
    assert llama_msgs[1].role == MessageRole.ASSISTANT
    assert llama_msgs[1].content == "Hi there"


@pytest.mark.unit
def test_chat_history_to_llama_messages_empty():
    """to_llama_messages on empty history should return empty list."""
    from tensortruth.services.chat_history import ChatHistory

    history = ChatHistory(messages=())

    assert history.to_llama_messages() == []


@pytest.mark.unit
def test_chat_history_to_llama_messages_system_role():
    """to_llama_messages should handle system role."""
    from tensortruth.services.chat_history import ChatHistory, ChatHistoryMessage

    msgs = (ChatHistoryMessage(role="system", content="You are helpful"),)
    history = ChatHistory(messages=msgs)

    llama_msgs = history.to_llama_messages()

    assert len(llama_msgs) == 1
    assert llama_msgs[0].role == MessageRole.SYSTEM
    assert llama_msgs[0].content == "You are helpful"


# =============================================================================
# ChatHistoryService.build_history Tests
# =============================================================================


def _create_mock_config(max_history_turns=3, cleaning_enabled=False):
    """Create a mock TensorTruthConfig for testing."""
    config = Mock()
    config.conversation.max_history_turns = max_history_turns
    config.history_cleaning.enabled = cleaning_enabled
    config.history_cleaning.remove_emojis = True
    config.history_cleaning.remove_filler_phrases = True
    config.history_cleaning.normalize_whitespace = True
    config.history_cleaning.collapse_newlines = True
    config.history_cleaning.filler_phrases = []
    return config


@pytest.mark.unit
def test_build_history_converts_session_messages():
    """Should convert user/assistant dicts to ChatHistoryMessage."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config()
    service = ChatHistoryService(config)

    session_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    history = service.build_history(session_messages)

    assert len(history.messages) == 2
    assert history.messages[0].role == "user"
    assert history.messages[0].content == "Hello"
    assert history.messages[1].role == "assistant"
    assert history.messages[1].content == "Hi there"


@pytest.mark.unit
def test_build_history_skips_command_messages():
    """Should skip messages with role not in (user, assistant)."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config()
    service = ChatHistoryService(config)

    session_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "command", "content": "/help"},  # Should be skipped
        {"role": "assistant", "content": "Hi there"},
    ]

    history = service.build_history(session_messages)

    assert len(history.messages) == 2


@pytest.mark.unit
def test_build_history_skips_malformed_messages():
    """Should skip messages missing role or content (no exception)."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config()
    service = ChatHistoryService(config)

    session_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant"},  # Missing content - skip
        {"content": "No role"},  # Missing role - skip
        {},  # Empty - skip
        {"role": "user", "content": "Valid"},
    ]

    history = service.build_history(session_messages)

    assert len(history.messages) == 2
    assert history.messages[0].content == "Hello"
    assert history.messages[1].content == "Valid"


@pytest.mark.unit
def test_build_history_handles_non_string_content():
    """Should convert non-string content to string."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config()
    service = ChatHistoryService(config)

    session_messages = [
        {"role": "user", "content": 123},  # Integer
        {"role": "assistant", "content": ["list", "item"]},  # List
    ]

    history = service.build_history(session_messages)

    assert len(history.messages) == 2
    assert history.messages[0].content == "123"
    assert history.messages[1].content == "['list', 'item']"


@pytest.mark.unit
def test_build_history_applies_hard_limit():
    """Should enforce MAX_HISTORY_TURNS (50) as safety net."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config(max_history_turns=1000)  # Config wants more
    service = ChatHistoryService(config)

    # Create 120 messages (60 turns)
    session_messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
        for i in range(120)
    ]

    history = service.build_history(session_messages)

    # Should be capped at 50 turns = 100 messages
    assert len(history.messages) == 100
    # Should keep the LAST 50 turns (messages 20-119)
    assert history.messages[0].content == "Message 20"


@pytest.mark.unit
def test_build_history_applies_max_turns_limit():
    """Should keep only last N turns when limit specified."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config(max_history_turns=10)
    service = ChatHistoryService(config)

    # Create 10 messages = 5 turns
    session_messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
        for i in range(10)
    ]

    # Request 2 turns = 4 messages
    history = service.build_history(session_messages, max_turns=2)

    assert len(history.messages) == 4
    # Should keep last 2 turns: messages 6, 7, 8, 9
    assert history.messages[0].content == "Message 6"
    assert history.messages[-1].content == "Message 9"


@pytest.mark.unit
def test_build_history_respects_zero_max_turns():
    """max_turns=0 should return empty history."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config()
    service = ChatHistoryService(config)

    session_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    history = service.build_history(session_messages, max_turns=0)

    assert history.is_empty
    assert len(history.messages) == 0


@pytest.mark.unit
def test_build_history_uses_config_defaults():
    """Should use config.rag.max_history_turns when not specified."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config(max_history_turns=1)  # Config says 1 turn
    service = ChatHistoryService(config)

    # 4 messages = 2 turns
    session_messages = [
        {"role": "user", "content": "Message 0"},
        {"role": "assistant", "content": "Message 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "assistant", "content": "Message 3"},
    ]

    history = service.build_history(session_messages)  # No max_turns arg

    # Should use config default of 1 turn = 2 messages
    assert len(history.messages) == 2
    assert history.messages[0].content == "Message 2"
    assert history.messages[1].content == "Message 3"


@pytest.mark.unit
def test_build_history_empty_input():
    """Should handle empty input gracefully."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config()
    service = ChatHistoryService(config)

    history = service.build_history([])

    assert history.is_empty


@pytest.mark.unit
def test_build_history_none_input():
    """Should handle None input gracefully."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config()
    service = ChatHistoryService(config)

    history = service.build_history(None)

    assert history.is_empty


@pytest.mark.unit
def test_build_history_preserves_system_messages():
    """Should preserve system role messages."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config()
    service = ChatHistoryService(config)

    session_messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    history = service.build_history(session_messages)

    assert len(history.messages) == 2
    assert history.messages[0].role == "system"


# =============================================================================
# ChatHistoryService.build_cleaned Tests
# =============================================================================


@pytest.mark.unit
def test_build_cleaned_removes_emojis_when_enabled():
    """Should remove emojis when config.history_cleaning.remove_emojis."""
    from tensortruth.services.chat_history import (
        ChatHistory,
        ChatHistoryMessage,
        ChatHistoryService,
    )

    config = _create_mock_config(cleaning_enabled=True)
    config.history_cleaning.remove_emojis = True
    service = ChatHistoryService(config)

    msgs = (ChatHistoryMessage(role="user", content="Hello! 😀 How are you?"),)
    history = ChatHistory(messages=msgs)

    cleaned = service.build_cleaned(history)

    assert "😀" not in cleaned.messages[0].content
    assert "Hello!" in cleaned.messages[0].content


@pytest.mark.unit
def test_build_cleaned_returns_new_instance():
    """Should return new ChatHistory, preserving immutability."""
    from tensortruth.services.chat_history import (
        ChatHistory,
        ChatHistoryMessage,
        ChatHistoryService,
    )

    config = _create_mock_config(cleaning_enabled=True)
    service = ChatHistoryService(config)

    msgs = (ChatHistoryMessage(role="user", content="Hello 😀"),)
    history = ChatHistory(messages=msgs)

    cleaned = service.build_cleaned(history)

    # Should be different object
    assert cleaned is not history
    # Original should be unchanged
    assert "😀" in history.messages[0].content


@pytest.mark.unit
def test_build_cleaned_filters_empty_content():
    """Should remove messages with empty content after cleaning."""
    from tensortruth.services.chat_history import (
        ChatHistory,
        ChatHistoryMessage,
        ChatHistoryService,
    )

    config = _create_mock_config(cleaning_enabled=True)
    config.history_cleaning.remove_emojis = True
    service = ChatHistoryService(config)

    msgs = (
        ChatHistoryMessage(role="user", content="Hello"),
        ChatHistoryMessage(role="assistant", content="😀😀😀"),  # Only emojis
        ChatHistoryMessage(role="user", content="Thanks"),
    )
    history = ChatHistory(messages=msgs)

    cleaned = service.build_cleaned(history)

    # The emoji-only message should be filtered out
    assert len(cleaned.messages) == 2
    assert cleaned.messages[0].content == "Hello"
    assert cleaned.messages[1].content == "Thanks"


@pytest.mark.unit
def test_build_cleaned_disabled():
    """Should return unchanged history when cleaning disabled."""
    from tensortruth.services.chat_history import (
        ChatHistory,
        ChatHistoryMessage,
        ChatHistoryService,
    )

    config = _create_mock_config(cleaning_enabled=False)
    service = ChatHistoryService(config)

    msgs = (ChatHistoryMessage(role="user", content="Hello 😀"),)
    history = ChatHistory(messages=msgs)

    cleaned = service.build_cleaned(history)

    # Should keep the emoji since cleaning is disabled
    assert "😀" in cleaned.messages[0].content


@pytest.mark.unit
def test_build_cleaned_normalizes_whitespace():
    """Should normalize multiple spaces when enabled."""
    from tensortruth.services.chat_history import (
        ChatHistory,
        ChatHistoryMessage,
        ChatHistoryService,
    )

    config = _create_mock_config(cleaning_enabled=True)
    config.history_cleaning.normalize_whitespace = True
    service = ChatHistoryService(config)

    msgs = (ChatHistoryMessage(role="user", content="Hello    world"),)
    history = ChatHistory(messages=msgs)

    cleaned = service.build_cleaned(history)

    # Multiple spaces should be normalized to single
    assert cleaned.messages[0].content == "Hello world"


@pytest.mark.unit
def test_build_cleaned_empty_history():
    """Should handle empty history gracefully."""
    from tensortruth.services.chat_history import ChatHistory, ChatHistoryService

    config = _create_mock_config(cleaning_enabled=True)
    service = ChatHistoryService(config)

    history = ChatHistory(messages=())

    cleaned = service.build_cleaned(history)

    assert cleaned.is_empty


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_build_history_with_cleaning_enabled():
    """Should apply cleaning when apply_cleaning=True."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config(cleaning_enabled=True)
    config.history_cleaning.remove_emojis = True
    service = ChatHistoryService(config)

    session_messages = [
        {"role": "user", "content": "Hello 😀"},
        {"role": "assistant", "content": "Hi there! 🎉"},
    ]

    history = service.build_history(session_messages, apply_cleaning=True)

    # Emojis should be removed
    assert "😀" not in history.messages[0].content
    assert "🎉" not in history.messages[1].content


@pytest.mark.unit
def test_build_history_cleaning_override():
    """apply_cleaning parameter should override config."""
    from tensortruth.services.chat_history import ChatHistoryService

    config = _create_mock_config(cleaning_enabled=True)
    service = ChatHistoryService(config)

    session_messages = [{"role": "user", "content": "Hello 😀"}]

    # Override to disable cleaning
    history = service.build_history(session_messages, apply_cleaning=False)

    # Emoji should be preserved
    assert "😀" in history.messages[0].content
