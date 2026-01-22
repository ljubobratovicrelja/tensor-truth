"""Tests for history cleaner module.

Tests for regex-based preprocessing that cleans chat history before passing to LLM prompts.
Follows TDD approach - tests written before implementation.
"""

from llama_index.core.llms import ChatMessage, MessageRole

from tensortruth.app_utils.history_cleaner import (
    HistoryCleanerConfig,
    clean_chat_history,
    clean_history_content,
    clear_cache,
)


class TestCleanHistoryContent:
    """Tests for content cleaning function."""

    def test_removes_common_emojis(self):
        assert "Hello World" in clean_history_content("Hello 👋 World 😊")
        assert "👋" not in clean_history_content("Hello 👋 World")

    def test_removes_filler_great_question(self):
        result = clean_history_content("Great question! Here's the answer.")
        assert "Great question" not in result
        assert "Here's the answer." in result

    def test_removes_filler_happy_to_help(self):
        result = clean_history_content("I'd be happy to help! Let me explain.")
        assert "happy to help" not in result

    def test_removes_trailing_hope_this_helps(self):
        result = clean_history_content("The answer is 42. Hope this helps!")
        assert "Hope this helps" not in result
        assert "The answer is 42." in result

    def test_normalizes_multiple_spaces(self):
        assert clean_history_content("Hello    World") == "Hello World"

    def test_collapses_excessive_newlines(self):
        result = clean_history_content("Line1\n\n\n\nLine2")
        assert result == "Line1\n\nLine2"

    def test_preserves_code_blocks(self):
        code = "```python\ndef foo():\n    pass\n```"
        assert clean_history_content(code) == code

    def test_empty_and_none_input(self):
        assert clean_history_content("") == ""
        assert clean_history_content(None) is None

    def test_respects_config_disable_emoji(self):
        config = HistoryCleanerConfig(remove_emojis=False)
        result = clean_history_content("Hello 👋", config=config)
        assert "👋" in result

    def test_respects_config_disable_filler(self):
        config = HistoryCleanerConfig(remove_filler_phrases=False)
        result = clean_history_content("Great question! Answer.", config=config)
        assert "Great question" in result

    def test_custom_filler_phrases(self):
        config = HistoryCleanerConfig(filler_phrases=[r"(?i)^custom phrase[!.]*\s*"])
        result = clean_history_content("Custom phrase! Real content.", config=config)
        assert "Custom phrase" not in result
        assert "Real content." in result

    def test_disabled_config_returns_original(self):
        config = HistoryCleanerConfig(enabled=False)
        original = "Hello 👋 World! Great question!"
        result = clean_history_content(original, config=config)
        assert result == original


class TestCleanChatHistory:
    """Tests for chat history list cleaning."""

    def test_cleans_list_of_messages(self):
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello 👋"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Great question! Answer."),
        ]
        cleaned = clean_chat_history(messages)

        assert len(cleaned) == 2
        assert "👋" not in cleaned[0].content
        assert "Great question" not in cleaned[1].content
        assert "Answer." in cleaned[1].content

    def test_preserves_message_roles(self):
        messages = [
            ChatMessage(role=MessageRole.USER, content="Test"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response"),
        ]
        cleaned = clean_chat_history(messages)

        assert cleaned[0].role == MessageRole.USER
        assert cleaned[1].role == MessageRole.ASSISTANT

    def test_returns_empty_for_empty_input(self):
        assert clean_chat_history([]) == []

    def test_does_not_modify_original(self):
        original_content = "Hello 👋"
        messages = [ChatMessage(role=MessageRole.USER, content=original_content)]
        clean_chat_history(messages)
        assert messages[0].content == original_content


class TestCleanerCache:
    """Tests for LRU cache behavior."""

    def test_cache_returns_consistent_results(self):
        """Same input with same config returns same output."""
        config = HistoryCleanerConfig()
        content = "Hello 👋 World"

        result1 = clean_history_content(content, config)
        result2 = clean_history_content(content, config)

        assert result1 == result2

    def test_cache_invalidates_on_config_change(self):
        """Different config produces different results."""
        content = "Hello 👋 World"

        config_with_emoji = HistoryCleanerConfig(remove_emojis=True)
        config_without_emoji = HistoryCleanerConfig(remove_emojis=False)

        result_cleaned = clean_history_content(content, config_with_emoji)
        result_preserved = clean_history_content(content, config_without_emoji)

        assert "👋" not in result_cleaned
        assert "👋" in result_preserved

    def test_cache_invalidates_on_filler_phrase_change(self):
        """Custom filler phrases are respected."""
        content = "CustomPhrase! Real content."

        default_config = HistoryCleanerConfig()
        custom_config = HistoryCleanerConfig(
            filler_phrases=[r"(?i)^CustomPhrase[!.]*\s*"]
        )

        result_default = clean_history_content(content, default_config)
        result_custom = clean_history_content(content, custom_config)

        # Default should keep CustomPhrase (not in default list)
        assert "CustomPhrase" in result_default
        # Custom should remove it
        assert "CustomPhrase" not in result_custom

    def test_cache_is_ephemeral(self):
        """Cache doesn't affect original content or persist."""
        content = "Test 👋 content"
        config = HistoryCleanerConfig()

        # Clean content
        clean_history_content(content, config)

        # Clear cache
        clear_cache()

        # Should still work after cache clear
        result = clean_history_content(content, config)
        assert "👋" not in result


class TestHistoryCleanerConfig:
    """Tests for configuration dataclass."""

    def test_default_config_has_all_cleaning_enabled(self):
        config = HistoryCleanerConfig()
        assert config.enabled is True
        assert config.remove_emojis is True
        assert config.remove_filler_phrases is True
        assert config.normalize_whitespace is True
        assert config.collapse_newlines is True

    def test_default_filler_phrases_not_empty(self):
        config = HistoryCleanerConfig()
        assert len(config.filler_phrases) > 0

    def test_config_is_hashable_for_caching(self):
        """Config should produce consistent hash for LRU cache."""
        config1 = HistoryCleanerConfig()
        config2 = HistoryCleanerConfig()
        # Both should work with the cache (not necessarily equal hash,
        # but should work as cache keys)
        content = "Test"
        result1 = clean_history_content(content, config1)
        result2 = clean_history_content(content, config2)
        assert result1 == result2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_only_emojis(self):
        result = clean_history_content("👋😊🎉")
        assert result == ""

    def test_only_filler_phrase(self):
        result = clean_history_content("Great question!")
        assert result == ""

    def test_mixed_whitespace_types(self):
        result = clean_history_content("Hello\t\t  World")
        # Tabs and spaces normalized
        assert "Hello" in result and "World" in result

    def test_preserves_single_newlines(self):
        result = clean_history_content("Line1\nLine2")
        assert result == "Line1\nLine2"

    def test_preserves_double_newlines(self):
        result = clean_history_content("Line1\n\nLine2")
        assert result == "Line1\n\nLine2"

    def test_unicode_text_preserved(self):
        result = clean_history_content("こんにちは World")
        assert "こんにちは" in result

    def test_long_content_performance(self):
        """Ensure cleaning works on longer content without issues."""
        long_content = "Hello 👋 World. " * 1000
        result = clean_history_content(long_content)
        assert "👋" not in result
        assert "Hello" in result

    def test_filler_phrase_case_insensitive(self):
        """Filler phrases should match regardless of case."""
        result = clean_history_content("GREAT QUESTION! Here's the answer.")
        assert "GREAT QUESTION" not in result
        assert "Here's the answer." in result

    def test_filler_at_different_positions(self):
        """Filler phrases only match at expected positions."""
        # "Great question" at start should be removed
        result1 = clean_history_content("Great question! Answer here.")
        assert "Great question" not in result1

        # "Hope this helps" at end should be removed
        result2 = clean_history_content("The answer is here. Hope this helps!")
        assert "Hope this helps" not in result2
