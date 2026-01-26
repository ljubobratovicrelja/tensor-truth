"""Unit tests for synthesis engine."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tensortruth.core.synthesis import (
    CHARS_PER_TOKEN,
    CitationStyle,
    SynthesisConfig,
    build_synthesis_prompt,
    fit_sources_to_context,
    format_pages_for_synthesis,
    synthesize_with_llm_stream,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_pages():
    """Sample pages for testing."""
    return [
        {
            "url": "https://example.com/page1",
            "title": "Page 1",
            "content": "This is the content of page 1. " * 50,  # ~1500 chars
            "status": "success",
        },
        {
            "url": "https://example.com/page2",
            "title": "Page 2",
            "content": "This is the content of page 2. " * 50,
            "status": "success",
        },
        {
            "url": "https://example.com/page3",
            "title": "Page 3",
            "content": "This is the content of page 3. " * 50,
            "status": "success",
        },
    ]


@pytest.fixture
def sample_pages_with_failure():
    """Sample pages with some failures."""
    return [
        {
            "url": "https://example.com/page1",
            "title": "Page 1",
            "content": "Content 1",
            "status": "success",
        },
        {
            "url": "https://example.com/page2",
            "title": "Page 2",
            "content": "",
            "status": "failed",
        },
        {
            "url": "https://example.com/page3",
            "title": "Page 3",
            "content": "Content 3",
            "status": "success",
        },
    ]


@pytest.fixture
def source_scores():
    """Sample source scores."""
    return {
        "https://example.com/page1": 0.95,
        "https://example.com/page2": 0.75,
        "https://example.com/page3": 0.60,
    }


@pytest.fixture
def mock_llm():
    """Mock Ollama LLM instance."""
    llm = MagicMock()
    llm.context_window = 16384
    llm.model = "test-model"

    # Mock streaming response
    async def mock_stream():
        chunks = ["This ", "is ", "a ", "test ", "response."]
        for chunk_text in chunks:
            chunk = MagicMock()
            chunk.delta = chunk_text
            yield chunk

    llm.astream_complete = AsyncMock(return_value=mock_stream())
    return llm


# =============================================================================
# Context Management Tests
# =============================================================================


def test_fit_sources_to_context_basic(sample_pages):
    """Test basic context fitting with pages that all fit."""
    context_window = 16384
    fitted, allocations = fit_sources_to_context(
        sample_pages,
        source_scores=None,
        context_window=context_window,
    )

    # All pages should fit
    assert len(fitted) == 3
    assert len(allocations) == 3

    # Check that allocations match content lengths
    for page in fitted:
        assert allocations[page["url"]] == len(page["content"])


def test_fit_sources_to_context_overflow(sample_pages):
    """Test context fitting when pages exceed budget."""
    # Very small context window - only ~1200 chars available for input
    context_window = 300  # 300 tokens * 0.6 * 4 = 720 chars

    fitted, allocations = fit_sources_to_context(
        sample_pages,
        source_scores=None,
        context_window=context_window,
        input_context_pct=0.6,
    )

    # Should fit fewer pages or truncate
    assert len(fitted) <= len(sample_pages)

    # Total allocated should not exceed budget
    total_allocated = sum(allocations.values())
    max_budget = int(context_window * 0.6 * CHARS_PER_TOKEN)
    assert total_allocated <= max_budget


def test_fit_sources_to_context_with_scores(sample_pages, source_scores):
    """Test context fitting with relevance scores."""
    context_window = 16384

    fitted, allocations = fit_sources_to_context(
        sample_pages,
        source_scores=source_scores,
        context_window=context_window,
    )

    # Should fit all pages in this case
    assert len(fitted) == 3

    # Verify allocations exist for all fitted pages
    for page in fitted:
        assert page["url"] in allocations
        assert allocations[page["url"]] > 0


def test_fit_sources_to_context_per_source_cap(sample_pages):
    """Test that individual sources respect per-source cap."""
    context_window = 16384
    max_source_context_pct = 0.05  # Very small per-source cap

    fitted, allocations = fit_sources_to_context(
        sample_pages,
        source_scores=None,
        context_window=context_window,
        max_source_context_pct=max_source_context_pct,
    )

    max_per_source = int(context_window * max_source_context_pct * CHARS_PER_TOKEN)

    # Each allocation should respect the cap
    for url, chars in allocations.items():
        assert chars <= max_per_source


def test_fit_sources_to_context_empty():
    """Test context fitting with empty pages list."""
    fitted, allocations = fit_sources_to_context(
        [],
        source_scores=None,
        context_window=16384,
    )

    assert fitted == []
    assert allocations == {}


def test_context_window_calculation():
    """Test context window budget calculations."""
    context_window = 16384
    input_pct = 0.6

    # Expected budget
    expected_budget = int(context_window * input_pct * CHARS_PER_TOKEN)

    # Create pages that exactly fill the budget
    total_chars = expected_budget
    pages = [
        {
            "url": f"https://example.com/page{i}",
            "title": f"Page {i}",
            "content": "x" * (total_chars // 3),
            "status": "success",
        }
        for i in range(3)
    ]

    fitted, allocations = fit_sources_to_context(
        pages,
        source_scores=None,
        context_window=context_window,
        input_context_pct=input_pct,
    )

    # Total allocated should be close to budget (within rounding)
    total_allocated = sum(allocations.values())
    assert total_allocated <= expected_budget


# =============================================================================
# Citation Format Tests
# =============================================================================


def test_hyperlink_citation_format(sample_pages):
    """Test hyperlink citation format (used by /web)."""
    sources_text, combined_text = format_pages_for_synthesis(
        sample_pages,
        CitationStyle.HYPERLINK,
    )

    # Check sources list has hyperlinks
    assert "[Page 1](https://example.com/page1)" in sources_text
    assert "[Page 2](https://example.com/page2)" in sources_text

    # Check combined content has hyperlinks
    assert "[Page 1](https://example.com/page1)" in combined_text
    assert "### Source 1:" in combined_text


def test_hyperlink_citation_with_scores(sample_pages, source_scores):
    """Test hyperlink citations with relevance scores."""
    sources_text, combined_text = format_pages_for_synthesis(
        sample_pages,
        CitationStyle.HYPERLINK,
        source_scores=source_scores,
    )

    # Check that relevance scores are included
    assert "Relevance: 95%" in sources_text
    assert "Relevance: 75%" in sources_text


def test_bracket_citation_format(sample_pages):
    """Test bracket citation format (used by /browse)."""
    sources_text, combined_text = format_pages_for_synthesis(
        sample_pages,
        CitationStyle.BRACKET,
    )

    # Check sources list has simple numbered format
    assert "1. Page 1" in sources_text
    assert "2. Page 2" in sources_text

    # Should NOT have hyperlinks in source list
    assert "](https://" not in sources_text

    # Check combined content has source headers
    assert "## Source 1: Page 1" in combined_text
    assert "URL: https://example.com/page1" in combined_text


def test_format_pages_max_per_page_truncation(sample_pages):
    """Test max_per_page parameter truncates content."""
    max_chars = 100

    sources_text, combined_text = format_pages_for_synthesis(
        sample_pages,
        CitationStyle.HYPERLINK,
        max_per_page=max_chars,
    )

    # Content should be truncated
    assert "[Content truncated...]" in combined_text


# =============================================================================
# Prompt Building Tests
# =============================================================================


def test_prompt_template_hyperlink():
    """Test prompt generation for hyperlink style."""
    config = SynthesisConfig(
        query="What is machine learning?",
        context_window=16384,
        citation_style=CitationStyle.HYPERLINK,
    )

    sources_text = "1. [ML Guide](https://example.com/ml)"
    combined_text = "### Source 1: [ML Guide](https://example.com/ml)\n\nContent here."

    prompt = build_synthesis_prompt(
        config,
        sources_text,
        combined_text,
        first_page_url="https://example.com/ml",
    )

    # Check prompt includes key elements
    assert "What is machine learning?" in prompt
    assert "CRITICAL CITATION RULES" in prompt
    assert "markdown hyperlinks" in prompt
    assert "ALWAYS cite using markdown hyperlinks" in prompt
    assert sources_text in prompt
    assert combined_text in prompt


def test_prompt_template_bracket():
    """Test prompt generation for bracket style."""
    config = SynthesisConfig(
        query="What is machine learning?",
        context_window=16384,
        citation_style=CitationStyle.BRACKET,
    )

    sources_text = "1. ML Guide"
    combined_text = (
        "## Source 1: ML Guide\nURL: https://example.com/ml\n\nContent here."
    )

    prompt = build_synthesis_prompt(
        config,
        sources_text,
        combined_text,
    )

    # Check prompt includes key elements
    assert "What is machine learning?" in prompt
    assert "[Source N] format" in prompt
    assert "Synthesize a comprehensive answer" in prompt
    # BRACKET style doesn't include sources_text separately, only in combined_text
    assert combined_text in prompt

    # Should NOT have hyperlink-specific instructions
    assert "CRITICAL CITATION RULES" not in prompt
    assert "markdown hyperlinks" not in prompt


def test_custom_instructions_integration():
    """Test that custom instructions are included in prompt."""
    config = SynthesisConfig(
        query="What is ML?",
        context_window=16384,
        citation_style=CitationStyle.HYPERLINK,
        custom_instructions="Focus on practical applications",
    )

    prompt = build_synthesis_prompt(
        config,
        "sources",
        "content",
    )

    assert "Additional Instructions" in prompt
    assert "Focus on practical applications" in prompt


def test_relevance_guidance_in_prompt():
    """Test that relevance guidance appears when scores provided."""
    config = SynthesisConfig(
        query="What is ML?",
        context_window=16384,
        citation_style=CitationStyle.HYPERLINK,
        source_scores={"https://example.com": 0.95},
    )

    prompt = build_synthesis_prompt(
        config,
        "sources",
        "content",
    )

    assert "ordered by relevance" in prompt
    assert "higher-scored sources" in prompt


# =============================================================================
# Streaming Tests
# =============================================================================


@pytest.mark.asyncio
async def test_streaming_behavior(mock_llm, sample_pages):
    """Test that streaming works correctly."""
    config = SynthesisConfig(
        query="Test query",
        context_window=16384,
        citation_style=CitationStyle.HYPERLINK,
    )

    result_tokens = []
    async for token in synthesize_with_llm_stream(mock_llm, config, sample_pages):
        result_tokens.append(token)

    # Should receive all streamed tokens
    assert len(result_tokens) > 0
    full_response = "".join(result_tokens)
    assert "test response" in full_response.lower()


@pytest.mark.asyncio
async def test_streaming_error_handling(mock_llm, sample_pages):
    """Test error handling during streaming."""
    config = SynthesisConfig(
        query="Test query",
        context_window=16384,
        citation_style=CitationStyle.HYPERLINK,
    )

    # Make the LLM raise an error
    async def error_stream():
        raise ValueError("Test error")

    mock_llm.astream_complete = AsyncMock(return_value=error_stream())

    result_tokens = []
    async for token in synthesize_with_llm_stream(mock_llm, config, sample_pages):
        result_tokens.append(token)

    full_response = "".join(result_tokens)
    assert "Synthesis error" in full_response or "error" in full_response.lower()


@pytest.mark.asyncio
async def test_streaming_filters_failed_pages(mock_llm, sample_pages_with_failure):
    """Test that failed pages are filtered out."""
    config = SynthesisConfig(
        query="Test query",
        context_window=16384,
        citation_style=CitationStyle.HYPERLINK,
    )

    result_tokens = []
    async for token in synthesize_with_llm_stream(
        mock_llm, config, sample_pages_with_failure
    ):
        result_tokens.append(token)

    # Should still get results from successful pages
    assert len(result_tokens) > 0


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.asyncio
async def test_empty_pages_handling(mock_llm):
    """Test handling of empty pages list."""
    config = SynthesisConfig(
        query="Test query",
        context_window=16384,
        citation_style=CitationStyle.HYPERLINK,
    )

    result_tokens = []
    async for token in synthesize_with_llm_stream(mock_llm, config, []):
        result_tokens.append(token)

    full_response = "".join(result_tokens)
    assert "No pages could be fetched" in full_response


@pytest.mark.asyncio
async def test_single_page_synthesis(mock_llm):
    """Test synthesis with a single page."""
    config = SynthesisConfig(
        query="Test query",
        context_window=16384,
        citation_style=CitationStyle.HYPERLINK,
    )

    pages = [
        {
            "url": "https://example.com/page1",
            "title": "Single Page",
            "content": "Content here",
            "status": "success",
        }
    ]

    result_tokens = []
    async for token in synthesize_with_llm_stream(mock_llm, config, pages):
        result_tokens.append(token)

    # Should work with single page
    assert len(result_tokens) > 0


@pytest.mark.asyncio
async def test_max_context_overflow(mock_llm):
    """Test handling when content exceeds maximum context."""
    config = SynthesisConfig(
        query="Test query",
        context_window=100,  # Very small context
        citation_style=CitationStyle.HYPERLINK,
    )

    # Create pages with lots of content
    pages = [
        {
            "url": f"https://example.com/page{i}",
            "title": f"Page {i}",
            "content": "x" * 10000,  # Very long content
            "status": "success",
        }
        for i in range(5)
    ]

    result_tokens = []
    async for token in synthesize_with_llm_stream(mock_llm, config, pages):
        result_tokens.append(token)

    # Should handle gracefully (either truncate or warn)
    # At minimum, should not crash
    assert len(result_tokens) >= 0


@pytest.mark.asyncio
async def test_all_pages_failed(mock_llm):
    """Test when all pages have failed status."""
    config = SynthesisConfig(
        query="Test query",
        context_window=16384,
        citation_style=CitationStyle.HYPERLINK,
    )

    pages = [
        {
            "url": "https://example.com/page1",
            "title": "Failed Page",
            "content": "",
            "status": "failed",
        },
        {
            "url": "https://example.com/page2",
            "title": "Another Failed Page",
            "content": "",
            "status": "failed",
        },
    ]

    result_tokens = []
    async for token in synthesize_with_llm_stream(mock_llm, config, pages):
        result_tokens.append(token)

    full_response = "".join(result_tokens)
    assert "No pages" in full_response or "could not" in full_response.lower()
