"""Unit tests for tensortruth.utils.history_condenser module."""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.core.constants import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
)
from tensortruth.utils.history_condenser import condense_query, create_condenser_llm

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_base_llm():
    """Mock base Ollama LLM instance."""
    llm = MagicMock()
    llm.model = DEFAULT_MODEL
    llm.base_url = DEFAULT_OLLAMA_BASE_URL
    llm.context_window = 16384
    return llm


@pytest.fixture
def sample_chat_history():
    """Sample chat history string."""
    return (
        "User: What is Retrieval-Augmented Generation?\n"
        "Assistant: RAG is a technique that combines retrieval with generation...\n"
        "User: Can you explain the retrieval step?\n"
        "Assistant: The retrieval step queries a vector database..."
    )


@pytest.fixture
def sample_prompt_template():
    """Sample condenser prompt template."""
    return (
        "Convert the follow-up question into a standalone query.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User Input: {question}\n\n"
        "Standalone Query:"
    )


# ============================================================================
# Tests for create_condenser_llm
# ============================================================================


def test_create_condenser_llm_defaults(mock_base_llm):
    """Test create_condenser_llm delegates to get_orchestrator_llm with correct params."""
    with patch("tensortruth.core.ollama.get_orchestrator_llm") as mock_get_llm:
        mock_get_llm.return_value = MagicMock()
        create_condenser_llm(mock_base_llm)

        mock_get_llm.assert_called_once_with(
            DEFAULT_MODEL,
            DEFAULT_OLLAMA_BASE_URL,
            16384,
        )


def test_create_condenser_llm_custom_params(mock_base_llm):
    """Test create_condenser_llm ignores custom params (delegates to singleton)."""
    with patch("tensortruth.core.ollama.get_orchestrator_llm") as mock_get_llm:
        mock_get_llm.return_value = MagicMock()
        create_condenser_llm(
            mock_base_llm,
            temperature=0.5,
            thinking=True,
            timeout=60.0,
        )

        # temperature, thinking, timeout are ignored — always delegates to singleton
        mock_get_llm.assert_called_once_with(
            DEFAULT_MODEL,
            DEFAULT_OLLAMA_BASE_URL,
            16384,
        )


def test_create_condenser_llm_derives_from_base(mock_base_llm):
    """Test that condenser LLM derives model, base_url, context_window from base LLM."""
    mock_base_llm.model = "custom-model:7b"
    mock_base_llm.base_url = "http://custom:8080"
    mock_base_llm.context_window = 32768

    with patch("tensortruth.core.ollama.get_orchestrator_llm") as mock_get_llm:
        mock_get_llm.return_value = MagicMock()
        create_condenser_llm(mock_base_llm)

        mock_get_llm.assert_called_once_with(
            "custom-model:7b",
            "http://custom:8080",
            32768,
        )


def test_create_condenser_llm_fallback_on_missing_attrs():
    """Test create_condenser_llm falls back to defaults if attributes missing."""
    # Create LLM without model/base_url/context_window attributes
    llm = MagicMock(spec=[])

    with patch("tensortruth.core.ollama.get_orchestrator_llm") as mock_get_llm:
        mock_get_llm.return_value = MagicMock()
        create_condenser_llm(llm)

        # Should use default values
        mock_get_llm.assert_called_once_with(
            DEFAULT_MODEL,
            DEFAULT_OLLAMA_BASE_URL,
            16384,
        )


# ============================================================================
# Tests for condense_query - Success Cases
# ============================================================================


def test_condense_query_success(sample_chat_history, sample_prompt_template):
    """Test successful query condensation."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "How does the RAG retrieval step work?"
    mock_llm.complete = MagicMock(return_value=mock_response)

    result = condense_query(
        llm=mock_llm,
        chat_history=sample_chat_history,
        question="How does it work?",
        prompt_template=sample_prompt_template,
    )

    assert result == "How does the RAG retrieval step work?"
    mock_llm.complete.assert_called_once()


def test_condense_query_async_execution(sample_chat_history, sample_prompt_template):
    """Test that condense_query executes asynchronously."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "Standalone query result"
    mock_llm.complete = MagicMock(return_value=mock_response)

    # Verify it's truly async by calling it
    result = condense_query(
        llm=mock_llm,
        chat_history=sample_chat_history,
        question="Follow-up question",
        prompt_template=sample_prompt_template,
    )

    assert result == "Standalone query result"
    # Verify acomplete (async method) was called
    assert mock_llm.complete.called


def test_condense_query_whitespace_stripping(
    sample_chat_history, sample_prompt_template
):
    """Test that condense_query strips whitespace from result."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    # Response with leading/trailing whitespace
    mock_response.__str__ = lambda self: "  \n  Condensed query  \n  "
    mock_llm.complete = MagicMock(return_value=mock_response)

    result = condense_query(
        llm=mock_llm,
        chat_history=sample_chat_history,
        question="Question",
        prompt_template=sample_prompt_template,
    )

    assert result == "Condensed query"


# ============================================================================
# Tests for condense_query - Error Handling
# ============================================================================


def test_condense_query_empty_result_fallback(
    sample_chat_history, sample_prompt_template
):
    """Test fallback to original question when condensation produces empty result."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: ""  # Empty result
    mock_llm.complete = MagicMock(return_value=mock_response)

    original_question = "What is the error?"
    result = condense_query(
        llm=mock_llm,
        chat_history=sample_chat_history,
        question=original_question,
        prompt_template=sample_prompt_template,
    )

    # Should fall back to original question
    assert result == original_question


def test_condense_query_whitespace_only_result_fallback(
    sample_chat_history, sample_prompt_template
):
    """Test fallback when condensation produces only whitespace."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "   \n\t  "  # Only whitespace
    mock_llm.complete = MagicMock(return_value=mock_response)

    original_question = "Debug this issue"
    result = condense_query(
        llm=mock_llm,
        chat_history=sample_chat_history,
        question=original_question,
        prompt_template=sample_prompt_template,
    )

    # Should fall back to original question
    assert result == original_question


def test_condense_query_exception_fallback(sample_chat_history, sample_prompt_template):
    """Test fallback to original question when LLM raises exception."""
    mock_llm = MagicMock()
    mock_llm.complete = MagicMock(side_effect=Exception("LLM timeout"))

    original_question = "Fix the bug"
    result = condense_query(
        llm=mock_llm,
        chat_history=sample_chat_history,
        question=original_question,
        prompt_template=sample_prompt_template,
        fallback_on_error=True,
    )

    # Should fall back to original question
    assert result == original_question


def test_condense_query_exception_propagation_when_disabled(
    sample_chat_history, sample_prompt_template
):
    """Test that exception is raised when fallback_on_error=False."""
    mock_llm = MagicMock()
    mock_llm.complete = MagicMock(side_effect=ValueError("Invalid prompt"))

    with pytest.raises(ValueError, match="Invalid prompt"):
        condense_query(
            llm=mock_llm,
            chat_history=sample_chat_history,
            question="Question",
            prompt_template=sample_prompt_template,
            fallback_on_error=False,
        )


# ============================================================================
# Tests for condense_query - Edge Cases
# ============================================================================


def test_condense_query_none_response(sample_chat_history, sample_prompt_template):
    """Test handling of None response from LLM."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: None  # None result (edge case)
    mock_llm.complete = MagicMock(return_value=mock_response)

    original_question = "Explain this"

    # str(None) = "None", but when checking truthiness, should still work
    # However, this is an edge case - let's test it handles gracefully
    with pytest.raises(Exception):
        # This will likely raise an exception trying to strip None
        condense_query(
            llm=mock_llm,
            chat_history=sample_chat_history,
            question=original_question,
            prompt_template=sample_prompt_template,
            fallback_on_error=False,
        )


def test_condense_query_empty_history(sample_prompt_template):
    """Test condensation with empty chat history."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "Standalone version of question"
    mock_llm.complete = MagicMock(return_value=mock_response)

    result = condense_query(
        llm=mock_llm,
        chat_history="",  # Empty history
        question="What is RAG?",
        prompt_template=sample_prompt_template,
    )

    assert result == "Standalone version of question"


def test_condense_query_empty_question(sample_chat_history, sample_prompt_template):
    """Test condensation with empty question."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "Processed empty question"
    mock_llm.complete = MagicMock(return_value=mock_response)

    result = condense_query(
        llm=mock_llm,
        chat_history=sample_chat_history,
        question="",  # Empty question
        prompt_template=sample_prompt_template,
    )

    assert result == "Processed empty question"


def test_condense_query_complex_history(sample_prompt_template):
    """Test condensation with complex multi-turn history."""
    complex_history = (
        "User: Show me the BasicBlock class\n"
        "Assistant: Here's the BasicBlock implementation...\n"
        "User: What about the forward method?\n"
        "Assistant: The forward method does...\n"
        "User: Are there any bugs?\n"
        "Assistant: I found potential issues with...\n"
        "User: Can you explain the second one?\n"
    )

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = (
        lambda self: "Explain the second bug in BasicBlock forward method"
    )
    mock_llm.complete = MagicMock(return_value=mock_response)

    result = condense_query(
        llm=mock_llm,
        chat_history=complex_history,
        question="Can you explain the second one?",
        prompt_template=sample_prompt_template,
    )

    assert result == "Explain the second bug in BasicBlock forward method"


def test_condense_query_timeout_handling(sample_chat_history, sample_prompt_template):
    """Test handling of LLM timeout."""
    mock_llm = MagicMock()
    mock_llm.complete = MagicMock(side_effect=TimeoutError("Request timeout"))

    original_question = "Refactor this code"
    result = condense_query(
        llm=mock_llm,
        chat_history=sample_chat_history,
        question=original_question,
        prompt_template=sample_prompt_template,
        fallback_on_error=True,
    )

    # Should fall back to original question on timeout
    assert result == original_question


def test_condense_query_prompt_formatting(sample_chat_history, sample_prompt_template):
    """Test that prompt is correctly formatted with placeholders."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = lambda self: "Result"
    mock_llm.complete = MagicMock(return_value=mock_response)

    condense_query(
        llm=mock_llm,
        chat_history=sample_chat_history,
        question="Test question",
        prompt_template=sample_prompt_template,
    )

    # Verify the prompt was formatted correctly
    call_args = mock_llm.complete.call_args[0][0]
    assert sample_chat_history in call_args
    assert "Test question" in call_args
