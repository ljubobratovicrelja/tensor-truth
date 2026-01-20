"""Tests for natural language intent classification and query enhancement."""

from unittest.mock import Mock

from tensortruth.app_utils.intent_classifier import (
    classify_intent,
    detect_and_classify,
    enhance_query_with_context,
    has_agent_triggers,
)


class TestTriggerDetection:
    """Test fast pre-filter for trigger words."""

    def test_browse_trigger_detected(self):
        """Browse trigger words should be detected."""
        assert has_agent_triggers("browse the latest AI news")
        assert has_agent_triggers("Research transformers")
        assert has_agent_triggers("find out more about this")
        assert has_agent_triggers("look up Python features")

    def test_search_trigger_detected(self):
        """Search trigger words should be detected."""
        assert has_agent_triggers("search for quantum computing")
        assert has_agent_triggers("google the latest updates")
        assert has_agent_triggers("web search for tutorials")
        assert has_agent_triggers("look online for documentation")

    def test_no_triggers(self):
        """Normal chat messages should not have triggers."""
        assert not has_agent_triggers("What does the docs say?")
        assert not has_agent_triggers("Tell me more about that")
        assert not has_agent_triggers("Can you explain this concept?")
        assert not has_agent_triggers("Show me the documentation")

    def test_case_insensitive(self):
        """Trigger detection should be case insensitive."""
        assert has_agent_triggers("BROWSE the news")
        assert has_agent_triggers("Search FOR updates")
        assert has_agent_triggers("ReSeArCh transformers")


class TestIntentClassification:
    """Test LLM-based intent classification."""

    def test_explicit_browse_intent(self):
        """Explicit browse command should classify as browse."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock(
            __str__=lambda _: (
                '{"intent": "browse", "query": "latest AI news", '
                '"reason": "explicit"}'
            )
        )

        result = classify_intent("Browse the latest AI news", [], mock_llm)

        assert result.intent == "browse"
        assert result.query == "latest AI news"

    def test_contextual_browse_with_followup(self):
        """Browse command with contextual reference should still classify as browse."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock(
            __str__=lambda _: (
                '{"intent": "browse", "query": "this concept", '
                '"reason": "explicit_browse"}'
            )
        )

        recent_messages = [
            {"role": "user", "content": "Tell me about backpropagation"},
            {"role": "assistant", "content": "Backpropagation is a method..."},
        ]

        result = classify_intent(
            "browse more about this concept online", recent_messages, mock_llm
        )

        # Should classify as browse even though it's a follow-up
        assert result.intent == "browse"

    def test_search_intent(self):
        """Search command should classify as search."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock(
            __str__=lambda _: (
                '{"intent": "search", "query": "Python 3.12 features", '
                '"reason": "explicit_search"}'
            )
        )

        result = classify_intent("Search for Python 3.12 features", [], mock_llm)

        assert result.intent == "search"
        assert result.query == "Python 3.12 features"

    def test_chat_intent(self):
        """Normal questions should classify as chat."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock(
            __str__=lambda _: '{"intent": "chat", "query": null, "reason": "no_web_intent"}'
        )

        result = classify_intent(
            "What does the documentation say about tensors?", [], mock_llm
        )

        assert result.intent == "chat"
        assert result.query is None


class TestQueryEnhancement:
    """Test contextual query enhancement."""

    def test_enhance_contextual_reference(self):
        """Query with 'this' should be enhanced with context."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock(__str__=lambda _: "backpropagation")

        recent_messages = [
            {"role": "user", "content": "Tell me about backpropagation"},
            {
                "role": "assistant",
                "content": "Backpropagation is a method for training neural networks...",
            },
        ]

        enhanced = enhance_query_with_context("this concept", recent_messages, mock_llm)

        # Should extract topic from context
        assert enhanced == "backpropagation"
        mock_llm.complete.assert_called_once()

    def test_no_enhancement_for_explicit_query(self):
        """Explicit queries should not be enhanced."""
        mock_llm = Mock()

        enhanced = enhance_query_with_context(
            "quantum computing",
            [{"role": "user", "content": "something unrelated"}],
            mock_llm,
        )

        # Should return original query without calling LLM
        assert enhanced == "quantum computing"
        mock_llm.complete.assert_not_called()

    def test_enhance_more_about_pattern(self):
        """'more about' pattern should trigger enhancement."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock(
            __str__=lambda _: "transformers architecture"
        )

        recent_messages = [
            {"role": "assistant", "content": "Transformers use self-attention..."}
        ]

        enhanced = enhance_query_with_context(
            "more about it", recent_messages, mock_llm
        )

        # Should call LLM for enhancement
        mock_llm.complete.assert_called_once()
        assert enhanced == "transformers architecture"


class TestEndToEndIntentDetection:
    """Test complete intent detection pipeline."""

    def test_no_triggers_fast_path(self):
        """Messages without triggers should skip LLM classification."""
        result = detect_and_classify(
            "What is backpropagation?", [], llm=None  # LLM not needed
        )

        assert result.intent == "chat"
        assert result.reason == "no_triggers"

    def test_browse_with_triggers_uses_llm(self):
        """Messages with triggers should use LLM classification."""
        mock_llm = Mock()
        mock_llm.complete.side_effect = [
            # First call: classify_intent
            Mock(
                __str__=lambda _: '{"intent": "browse", "query": "AI news", "reason": "explicit"}'
            ),
            # Second call: enhance_query_with_context (not called for non-contextual queries)
        ]

        result = detect_and_classify("browse the latest AI news", [], llm=mock_llm)

        assert result.intent == "browse"
        assert result.query == "AI news"
        # At least one LLM call for classification
        assert mock_llm.complete.call_count >= 1

    def test_contextual_browse_full_pipeline(self):
        """Test full pipeline: classification -> enhancement."""
        mock_llm = Mock()
        mock_llm.complete.side_effect = [
            # First call: classify_intent
            Mock(
                __str__=lambda _: (
                    '{"intent": "browse", "query": "this concept", '
                    '"reason": "explicit"}'
                )
            ),
            # Second call: enhance_query_with_context
            Mock(__str__=lambda _: "backpropagation"),
        ]

        recent_messages = [
            {"role": "user", "content": "Tell me about backpropagation"},
            {"role": "assistant", "content": "Backpropagation is..."},
        ]

        result = detect_and_classify(
            "browse more about this concept online", recent_messages, llm=mock_llm
        )

        assert result.intent == "browse"
        assert result.query == "backpropagation"  # Enhanced from "this concept"
        assert mock_llm.complete.call_count == 2  # Classification + enhancement

    def test_reuse_provided_llm(self):
        """Should reuse provided LLM instead of creating new one."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock(
            __str__=lambda _: '{"intent": "browse", "query": "test", "reason": "explicit"}'
        )

        detect_and_classify("browse something", [], llm=mock_llm)  # Provide LLM

        # Should use provided LLM, not create new one
        mock_llm.complete.assert_called_once()

    def test_fallback_llm_creation(self):
        """Should create LLM if none provided."""
        # This is harder to test without mocking Ollama constructor
        # but we can verify it doesn't crash
        result = detect_and_classify(
            "What is this?", [], llm=None  # No triggers, won't need LLM
        )

        # Should work without LLM for non-trigger messages
        assert result.intent == "chat"


class TestCriticalBrowseScenarios:
    """Test critical scenarios that were previously broken."""

    def test_browse_after_rag_answer(self):
        """CRITICAL: Browse after RAG answer should NOT be classified as chat."""
        mock_llm = Mock()
        mock_llm.complete.side_effect = [
            # Classification should return browse
            Mock(
                __str__=lambda _: (
                    '{"intent": "browse", "query": "this concept", '
                    '"reason": "explicit_browse"}'
                )
            ),
            # Enhancement
            Mock(__str__=lambda _: "backpropagation"),
        ]

        conversation = [
            {"role": "user", "content": "Tell me about backpropagation"},
            {
                "role": "assistant",
                "content": "Backpropagation is a fundamental algorithm...",
            },
        ]

        result = detect_and_classify(
            "browse more about this concept online", conversation, llm=mock_llm
        )

        # MUST be browse, not chat!
        assert (
            result.intent == "browse"
        ), "Browse trigger must override follow-up detection"
        assert result.query == "backpropagation"

    def test_online_keyword_implies_browse(self):
        """Keywords like 'online', 'web', 'internet' with browse triggers
        should classify as browse."""
        mock_llm = Mock()
        mock_llm.complete.return_value = Mock(
            __str__=lambda _: (
                '{"intent": "browse", "query": "backpropagation", '
                '"reason": "online_keyword"}'
            )
        )

        # Note: "find out" is a trigger word, "online" reinforces the intent
        result = detect_and_classify(
            "find out more information online about this",
            [{"role": "assistant", "content": "Backpropagation is..."}],
            llm=mock_llm,
        )

        assert result.intent == "browse"

    def test_explicit_triggers_always_win(self):
        """Explicit triggers (browse, search) must ALWAYS override
        follow-up detection."""
        test_cases = [
            ("browse this topic", "browse"),
            ("research more about it", "browse"),
            ("find out about that online", "browse"),
            ("search for information on this", "search"),
            ("google more details", "search"),
        ]

        for message, expected_intent in test_cases:
            mock_llm = Mock()
            # Create a proper response string
            response_str = (
                f'{{"intent": "{expected_intent}", '
                f'"query": "test", "reason": "explicit"}}'
            )
            mock_response = Mock()
            mock_response.__str__ = Mock(return_value=response_str)
            mock_llm.complete.return_value = mock_response

            result = detect_and_classify(
                message,
                [{"role": "assistant", "content": "Previous answer..."}],
                llm=mock_llm,
            )

            assert result.intent == expected_intent, f"Failed for: {message}"


class TestPromptConstruction:
    """Test that classification prompt is constructed correctly."""

    def test_classification_prompt_includes_priority_rules(self):
        """Prompt should have explicit priority ordering."""
        from tensortruth.app_utils.intent_classifier import CLASSIFICATION_PROMPT

        # Check for critical/priority keywords and browse instruction
        assert (
            "CRITICAL" in CLASSIFICATION_PROMPT
            or "first" in CLASSIFICATION_PROMPT.lower()
            or "check in order" in CLASSIFICATION_PROMPT.lower()
        )
        assert "browse" in CLASSIFICATION_PROMPT

    def test_classification_prompt_has_contextual_example(self):
        """Prompt should include example with browse trigger."""
        from tensortruth.app_utils.intent_classifier import CLASSIFICATION_PROMPT

        # Should have examples showing browse classification
        assert "Browse" in CLASSIFICATION_PROMPT or "Research" in CLASSIFICATION_PROMPT
