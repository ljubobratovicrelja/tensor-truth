"""Unified synthesis engine for LLM-based summarization of web research results.

This module consolidates the duplicated summarization logic from:
- /web command (web_search.py)
- /browse agent (browse/agent.py)

The engine supports different citation formats while maintaining a single
source of truth for context management, prompt building, and streaming.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)

# Token-to-character conversion constant (rough approximation)
CHARS_PER_TOKEN = 4


class CitationStyle(str, Enum):
    """Citation format styles for synthesis output."""

    HYPERLINK = "hyperlink"  # [Title](url) - used by /web
    BRACKET = "bracket"  # [Source N] - used by /browse


class QueryType(str, Enum):
    """Query type for adaptive prompt structure."""

    PERSON = "person"
    COMPARISON = "comparison"
    NEWS_EVENT = "news_event"
    TECHNICAL = "technical"
    GENERAL = "general"


@dataclass
class ModelPromptConfig:
    """Model-specific prompt configuration."""

    use_system_prompt: bool = True
    temperature_override: Optional[float] = None
    include_reasoning_directives: bool = False
    model_family: Optional[str] = None


@dataclass
class SynthesisConfig:
    """Configuration for synthesis engine.

    Args:
        query: User's original query
        context_window: LLM context window size in tokens
        citation_style: Citation format to use in output
        custom_instructions: Optional custom instructions for the LLM
        input_context_pct: Percentage of context for input (rest for output)
        output_context_pct: Percentage of context for output
        source_scores: Optional dict of url -> relevance score for prioritization
        max_source_context_pct: Max percentage of context per individual source
        model_name: Optional model name for model-specific prompt adaptation
    """

    query: str
    context_window: int
    citation_style: CitationStyle
    custom_instructions: Optional[str] = None
    input_context_pct: float = 0.6
    output_context_pct: float = 0.4
    source_scores: Optional[Dict[str, float]] = None
    max_source_context_pct: float = 0.15
    model_name: Optional[str] = None


# Query type detection patterns
_PERSON_PATTERNS = re.compile(
    r"\b(who is|biography|life of|about .+ person|background of)\b", re.I
)
_COMPARISON_PATTERNS = re.compile(
    r"\b(compare|versus|vs\.?|difference between|pros and cons)\b", re.I
)
_NEWS_PATTERNS = re.compile(
    r"\b(news|latest|recent|what happened|timeline|developments)\b", re.I
)
_TECHNICAL_PATTERNS = re.compile(
    r"\b(how to|implementation|algorithm|API|documentation|tutorial)\b", re.I
)


def detect_query_type(query: str) -> QueryType:
    """Detect query type using rule-based keyword matching.

    Args:
        query: User's query string

    Returns:
        QueryType enum value
    """
    if _PERSON_PATTERNS.search(query):
        return QueryType.PERSON
    if _COMPARISON_PATTERNS.search(query):
        return QueryType.COMPARISON
    if _NEWS_PATTERNS.search(query):
        return QueryType.NEWS_EVENT
    if _TECHNICAL_PATTERNS.search(query):
        return QueryType.TECHNICAL
    return QueryType.GENERAL


def get_model_prompt_config(model_name: Optional[str]) -> ModelPromptConfig:
    """Get model-specific prompt configuration.

    Args:
        model_name: Name of the model (e.g., "deepseek-r1:8b", "qwen3:8b-q8_0")

    Returns:
        ModelPromptConfig with model-specific settings
    """
    if not model_name:
        return ModelPromptConfig()

    model_lower = model_name.lower()

    # DeepSeek R1: no system prompt, add reasoning directives
    if "deepseek-r1" in model_lower:
        return ModelPromptConfig(
            use_system_prompt=False,
            temperature_override=0.6,
            include_reasoning_directives=True,
            model_family="deepseek-r1",
        )

    if "qwen3" in model_lower:
        return ModelPromptConfig(model_family="qwen3")

    if "llama" in model_lower:
        return ModelPromptConfig(model_family="llama")

    return ModelPromptConfig()


def fit_sources_to_context(
    pages: List[Dict[str, str]],
    source_scores: Optional[Dict[str, float]],
    context_window: int,
    input_context_pct: float = 0.6,
    max_source_context_pct: float = 0.15,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Fit sources to context window using fill-from-top strategy.

    Pages should be pre-sorted by relevance (highest first) when using source_scores.
    Uses a greedy algorithm to maximize content from highest-ranked sources.

    Args:
        pages: List of dicts with keys: url, title, content, status
        source_scores: Optional dict of url -> relevance score for each source
        context_window: Total context window size in tokens
        input_context_pct: Percentage of context window for input (rest for output)
        max_source_context_pct: Max percentage of context per source

    Returns:
        Tuple of:
        - List of page dicts with truncated content that fit
        - Dict of url -> allocated_chars for each included source

    Strategy:
    1. Calculate total budget: context_window * input_context_pct * CHARS_PER_TOKEN
    2. Calculate per-source cap: context_window * max_source_context_pct * CHARS_PER_TOKEN
    3. Greedily fill from top until budget exhausted
    4. Last source may be truncated to fit remaining budget
    """
    if not pages:
        return [], {}

    # Calculate budgets in characters
    total_budget = int(context_window * input_context_pct * CHARS_PER_TOKEN)
    per_source_cap = int(context_window * max_source_context_pct * CHARS_PER_TOKEN)

    fitted: List[Dict[str, str]] = []
    allocations: Dict[str, int] = {}
    remaining_budget = total_budget

    for page in pages:
        if remaining_budget <= 0:
            break

        content = page.get("content", "")
        url = page["url"]

        # Cap content at per-source maximum
        content_to_use = content[:per_source_cap]

        # If content fits fully within remaining budget
        if len(content_to_use) <= remaining_budget:
            fitted_page = page.copy()
            fitted_page["content"] = content_to_use
            fitted.append(fitted_page)
            allocations[url] = len(content_to_use)
            remaining_budget -= len(content_to_use)
        else:
            # Truncate to fit remaining budget
            truncated = content_to_use[:remaining_budget]
            fitted_page = page.copy()
            fitted_page["content"] = truncated
            fitted.append(fitted_page)
            allocations[url] = len(truncated)
            remaining_budget = 0

    return fitted, allocations


def build_citation_instructions(
    citation_style: CitationStyle,
    example_title: str,
    example_url: str,
    include_reasoning: bool = False,
) -> str:
    """Build citation instructions with explicit examples.

    Args:
        citation_style: Citation format to use
        example_title: Title of first page for realistic example
        example_url: URL of first page for realistic example
        include_reasoning: Whether to include chain-of-thought guidance

    Returns:
        Formatted citation instructions
    """
    if citation_style == CitationStyle.HYPERLINK:
        reasoning_guidance = ""
        if include_reasoning:
            reasoning_guidance = """

**REASONING PROCESS (Optional):**
Before writing, think step by step:
1. What specific fact or claim am I making?
2. Which source(s) support this?
3. How should I cite them inline?
"""

        return f"""CRITICAL CITATION RULES - READ CAREFULLY:

1. **ALWAYS cite using markdown hyperlinks**: `[Title](url)`
   - NEVER use plain [1] or [2] style
   - NEVER write bare numbers or brackets

2. **Four correct citation examples:**
   - Example 1: "According to [{example_title}]({example_url}), the key finding is..."
   - Example 2: "The [YOLO algorithm](https://pjreddie.com/darknet/yolo/) detects objects..."
   - Example 3: "Research shows that [Machine Learning Basics](https://example.com/ml) involves..."
   - Example 4: "As explained in [{example_title}]({example_url}), neural networks..."

3. **WRONG examples (DO NOT use):**
   - "According to [1], the algorithm..." (incorrect)
   - "The study [2] shows..." (incorrect)
   - "Source 3 indicates..." (incorrect)

4. **RIGHT examples (ALWAYS use):**
   - "According to [{example_title}]({example_url}), the algorithm..." (correct)
   - "The [Study Name](url) shows..." (correct)
   - "[Source Title](url) indicates..." (correct)

5. **Additional rules:**
   - Preserve ALL existing hyperlinks from source content
   - Link technical terms to their definitions when sources provide them
   - Cite multiple sources together: [Source A](url1), [Source B](url2){reasoning_guidance}"""

    else:  # BRACKET style
        return """**Citation Instructions:**
- Use [Source N] format where N is the source number
- Place citations immediately after relevant claims
- Example: "The algorithm achieves 95% accuracy [Source 1]."
- Cite multiple sources when appropriate: [Source 1], [Source 2]"""


def build_structure_template(
    query_type: QueryType, citation_style: CitationStyle
) -> str:
    """Build query-adaptive structure template.

    Uses hybrid approach: enforced Overview section + flexible subsections.

    Args:
        query_type: Type of query detected
        citation_style: Citation format being used

    Returns:
        Structure template guidance
    """
    # Citation example placeholder for bracket style
    cite_example = (
        "[Source N]" if citation_style == CitationStyle.BRACKET else "[title](url)"
    )

    base_template = f"""**Response Structure:**

Your response should include:

### Overview
[Required: Brief introduction to the topic with inline citations {cite_example}]
"""

    if query_type == QueryType.PERSON:
        return base_template + """
You may organize the rest using sections like:
### Background & Early Life (if relevant)
### Career & Achievements (if relevant)
### Impact & Legacy (if relevant)

Use sections that best fit the available information."""

    elif query_type == QueryType.COMPARISON:
        return base_template + """
You may organize the rest using sections like:
### Key Similarities (if relevant)
### Key Differences (if relevant)
### Comparative Analysis (if relevant)
### Recommendations or Conclusion (if relevant)

Focus on clear, side-by-side comparisons where appropriate."""

    elif query_type == QueryType.NEWS_EVENT:
        return base_template + """
You may organize the rest using sections like:
### Timeline of Events (if relevant)
### Key Developments (if relevant)
### Impact & Analysis (if relevant)

Prioritize recent information and chronological clarity."""

    elif query_type == QueryType.TECHNICAL:
        return base_template + """
You may organize the rest using sections like:
### Technical Details (if relevant)
### Implementation Steps (if relevant)
### Examples & Use Cases (if relevant)
### Key Considerations (if relevant)

Focus on practical, actionable information."""

    else:  # GENERAL
        return base_template + """
You may organize the rest using sections appropriate to the topic, such as:
### Key Concepts (if relevant)
### Detailed Analysis (if relevant)
### Important Takeaways (if relevant)

Structure your response based on the available information."""


def build_fusion_instructions() -> str:
    """Build source fusion guidance for synthesis.

    Returns:
        Fusion instructions for combining multiple sources
    """
    return """
**SOURCE SYNTHESIS:**
- Synthesize overlapping information into cohesive paragraphs
- Cite multiple sources together when they agree: [Source A](url1), [Source B](url2)
- Note conflicts explicitly: "Source A suggests X, while Source B indicates Y."
- Integrate information naturally rather than reporting source-by-source"""


def format_pages_for_synthesis(
    pages: List[Dict[str, str]],
    citation_style: CitationStyle,
    source_scores: Optional[Dict[str, float]] = None,
    max_per_page: Optional[int] = None,
) -> Tuple[str, str]:
    """Format pages for LLM consumption.

    Args:
        pages: List of page dicts with url, title, content, status
        citation_style: Citation format to use
        source_scores: Optional relevance scores for sources
        max_per_page: Optional max chars per page (for additional truncation)

    Returns:
        Tuple of (sources_list_text, combined_content_text)
    """
    sources_list = []
    combined = []

    for idx, page in enumerate(pages, 1):
        url = page["url"]
        title = page["title"]
        content = page.get("content", "")

        # Build source list entry
        if citation_style == CitationStyle.HYPERLINK:
            # Add relevance score if available
            if source_scores and url in source_scores:
                score = source_scores[url]
                sources_list.append(
                    f"{idx}. [{title}]({url}) — Relevance: {score*100:.0f}%"
                )
            else:
                sources_list.append(f"{idx}. [{title}]({url})")
        else:  # BRACKET
            sources_list.append(f"{idx}. {title}")

        # Truncate content if max_per_page specified
        if max_per_page and len(content) > max_per_page:
            content = content[:max_per_page]
            content += "\n\n[Content truncated...]"

        # Build combined content entry
        if citation_style == CitationStyle.HYPERLINK:
            combined.append(f"### Source {idx}: [{title}]({url})\n\n{content}\n\n---\n")
        else:  # BRACKET
            combined.append(f"## Source {idx}: {title}\nURL: {url}\n\n{content}\n")

    sources_text = "\n".join(sources_list)
    combined_text = "\n".join(combined)

    # Add separator for bracket style
    if citation_style == CitationStyle.BRACKET and combined:
        bracket_sources = []
        for i in range(len(pages)):
            if pages[i].get("content"):
                page = pages[i]
                source = (
                    f"## Source {i+1}: {page['title']}\n"
                    f"URL: {page['url']}\n\n{page.get('content', '')}\n"
                )
                bracket_sources.append(source)
        combined_text = "\n---\n\n".join(bracket_sources)

    return sources_text, combined_text


def build_synthesis_prompt(
    config: SynthesisConfig,
    sources_text: str,
    combined_text: str,
    first_page_url: Optional[str] = None,
    first_page_title: Optional[str] = None,
) -> str:
    """Build synthesis prompt based on citation style and query type.

    Args:
        config: Synthesis configuration
        sources_text: Formatted source list
        combined_text: Combined page content
        first_page_url: URL of first page (for example citations)
        first_page_title: Title of first page (for example citations)

    Returns:
        Complete prompt for LLM
    """
    # Detect query type and get model config
    query_type = detect_query_type(config.query)
    model_config = get_model_prompt_config(config.model_name)

    logger.debug(
        f"Building synthesis prompt: query_type={query_type.value}, "
        f"model_family={model_config.model_family}, "
        f"citation_style={config.citation_style.value}"
    )

    max_output_tokens = int(config.context_window * config.output_context_pct)
    target_words = max_output_tokens

    # Build modular components
    example_title = first_page_title or "Source Title"
    example_url = first_page_url or "url"
    citation_instructions = build_citation_instructions(
        config.citation_style,
        example_title,
        example_url,
        include_reasoning=model_config.include_reasoning_directives,
    )
    structure_template = build_structure_template(query_type, config.citation_style)
    fusion_instructions = build_fusion_instructions()

    # Add custom instructions if provided
    custom_instruction_text = ""
    if config.custom_instructions:
        custom_instruction_text = (
            f"\n\n**Additional Instructions:** {config.custom_instructions}"
        )

    # Add relevance guidance when scores are available
    relevance_guidance = ""
    if config.source_scores:
        relevance_guidance = (
            "\n\n**Note:** Sources are ordered by relevance. "
            "Prioritize information from higher-scored sources when synthesizing your answer."
        )

    if config.citation_style == CitationStyle.HYPERLINK:
        # Web command style - comprehensive with markdown hyperlinks
        # Adapt system prompt based on model config
        system_role = (
            "You are a research assistant. " if model_config.use_system_prompt else ""
        )

        prompt = f"""{system_role}User asked: "{config.query}"

## Available Sources
{sources_text}{relevance_guidance}

## Content from Sources
{combined_text}

{citation_instructions}

{fusion_instructions}

{structure_template}

Provide a comprehensive answer ({target_words} words approx.) using the structure \
above.{custom_instruction_text}

Begin your response:"""

    else:  # BRACKET
        # Browse agent style - simpler with bracket citations
        prompt = f"""Synthesize a comprehensive answer from the research results below.

Query: {config.query}

Research Results ({len(sources_text.splitlines())} pages fetched):
{combined_text}

{citation_instructions}

{fusion_instructions}

{structure_template}

Instructions:
- Provide a clear, comprehensive answer to the query
- Follow the citation and structure guidance above
- Include relevant details and examples{custom_instruction_text}

Answer:"""

    return prompt


async def synthesize_with_llm_stream(
    llm: Ollama,
    config: SynthesisConfig,
    pages: List[Dict[str, str]],
) -> AsyncGenerator[str, None]:
    """Stream LLM synthesis from web sources.

    This is the core synthesis engine that handles context management,
    page formatting, prompt building, and streaming for both /web and /browse.

    Args:
        llm: Pre-configured Ollama LLM instance
        config: Synthesis configuration
        pages: List of page dicts with keys: url, title, content, status

    Yields:
        str: Individual tokens from the LLM response

    Raises:
        Exception: If LLM streaming fails
    """
    logger.info(
        f"Starting synthesis of {len(pages)} pages with {config.citation_style.value} citations..."
    )

    if not pages:
        yield "**No pages could be fetched.** Please try a different query."
        return

    # Filter to successful pages only
    successful_pages = [
        p for p in pages if p.get("status") == "success" and p.get("content")
    ]

    if not successful_pages:
        yield "**No pages with content could be fetched.** Please try a different query."
        return

    # Use pages as-is (they should already be fitted by the pipeline)
    # NOTE: For backward compatibility with old code that doesn't use pipeline,
    # we could add a flag here, but for now assume pages are pre-fitted
    fitted_pages = successful_pages

    logger.info(f"Using {len(fitted_pages)} pre-fitted pages for synthesis")

    # Calculate max chars per page for initial formatting
    max_input_tokens = int(config.context_window * config.input_context_pct)
    max_total_chars = max_input_tokens * CHARS_PER_TOKEN
    max_per_page = min(max_total_chars // len(fitted_pages), 4000)

    # Debug: Log what we're formatting
    logger.info(f"Formatting {len(fitted_pages)} pages for synthesis:")
    for i, p in enumerate(fitted_pages, 1):
        logger.info(
            f"  {i}. title='{p.get('title', 'NO TITLE')}' url={p.get('url', 'NO URL')[:50]}..."
        )

    # Format pages for synthesis
    sources_text, combined_text = format_pages_for_synthesis(
        fitted_pages,
        config.citation_style,
        config.source_scores,
        max_per_page,
    )

    logger.info(f"Generated sources_text preview: {sources_text[:200]}...")

    # Additional safety check for total content size
    max_total_chars_strict = max_input_tokens * CHARS_PER_TOKEN
    if len(combined_text) > max_total_chars_strict:
        logger.warning(
            f"Combined content ({len(combined_text)} chars) exceeds strict limit "
            f"({max_total_chars_strict} chars). Truncating."
        )
        combined_text = combined_text[:max_total_chars_strict]
        combined_text += "\n\n[Additional content truncated for length...]"

    # Build prompt
    first_url = fitted_pages[0]["url"] if fitted_pages else None
    first_title = fitted_pages[0]["title"] if fitted_pages else None
    prompt = build_synthesis_prompt(
        config, sources_text, combined_text, first_url, first_title
    )

    # Log prompt size for debugging
    prompt_chars = len(prompt)
    prompt_tokens_approx = prompt_chars // CHARS_PER_TOKEN
    logger.info(
        f"Prompt size: {prompt_chars} chars (~{prompt_tokens_approx} tokens), "
        f"context window: {config.context_window} tokens"
    )

    try:
        # Stream synthesis
        async for chunk in await llm.astream_complete(prompt):
            if chunk.delta:
                yield chunk.delta

        logger.info("Synthesis streaming completed successfully")

    except Exception as e:
        logger.error(f"LLM streaming synthesis failed: {e}")
        yield f"\n\n**Synthesis error:** {str(e)}\n\nPlease try again."
