"""
Utility for condensing chat history and questions into standalone queries.

This module provides functions to transform follow-up questions that reference
previous conversation context into self-contained standalone queries suitable
for RAG retrieval or web search.
"""

import logging

from llama_index.llms.ollama import Ollama

from tensortruth.core.constants import (
    DEFAULT_AGENT_REASONING_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)


def create_condenser_llm(
    base_llm: Ollama,
    temperature: float = 0.0,
    thinking: bool = False,
    timeout: float = 30.0,
) -> Ollama:
    """
    Create an Ollama LLM instance optimized for query condensation.

    This creates a new Ollama instance that reuses the same model and base_url
    as the provided base_llm, but with settings optimized for fast, deterministic
    condensation. The same model is used server-side (no extra VRAM required).

    Args:
        base_llm: The base Ollama LLM to derive configuration from
        temperature: Temperature for generation (0.0 = deterministic)
        thinking: Whether to enable extended thinking (False = faster)
        timeout: Request timeout in seconds

    Returns:
        Configured Ollama instance for condensation

    Example:
        >>> main_llm = Ollama(model="llama3.1:8b")
        >>> condenser = create_condenser_llm(main_llm)
        >>> # condenser uses same model/url but optimized settings
    """
    # Access Ollama-specific attributes via getattr for type safety
    model = getattr(base_llm, "model", DEFAULT_AGENT_REASONING_MODEL)
    base_url = getattr(base_llm, "base_url", DEFAULT_OLLAMA_BASE_URL)

    return Ollama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        thinking=thinking,
        request_timeout=timeout,
    )


def condense_query(
    llm: Ollama,
    chat_history: str,
    question: str,
    prompt_template: str,
    fallback_on_error: bool = True,
) -> str:
    """
    Condense a chat history and follow-up question into a standalone query.

    Takes a conversation history and a follow-up question, then uses an LLM to
    transform them into a self-contained query that can be used for retrieval
    or search without needing the conversation context.

    Args:
        llm: Ollama LLM instance to use for condensation
        chat_history: Formatted chat history string
        question: User's follow-up question
        prompt_template: Template string with {chat_history} and {question} placeholders
        fallback_on_error: If True, return original question on error; if False, raise

    Returns:
        Condensed standalone query string, or original question if condensation fails
        (when fallback_on_error=True)

    Raises:
        Exception: If condensation fails and fallback_on_error=False

    Example:
        >>> history = "User: What is RAG?\\nAssistant: RAG is..."
        >>> question = "How does it work?"
        >>> template = "History: {chat_history}\\nQuestion: {question}\\nStandalone:"
        >>> result = condense_query(llm, history, question, template)
        >>> # result: "How does Retrieval-Augmented Generation work?"
    """
    try:
        # Format the prompt with history and question
        condenser_prompt = prompt_template.format(
            chat_history=chat_history,
            question=question,
        )

        # Call LLM to condense (synchronous complete)
        logger.info(f"Condensing query: {question[:100]}...")
        condensed_response = llm.complete(condenser_prompt)
        condensed_query = str(condensed_response).strip()

        # Validate: If condensation produces empty/whitespace, use original
        if not condensed_query:
            logger.warning(
                "Condensation produced empty result, using original question"
            )
            return question

        logger.info(f"Condensed to: {condensed_query[:100]}...")
        return condensed_query

    except Exception as e:
        # Fall back to original question if condensation fails
        logger.error(f"Condensation failed: {e}", exc_info=True)

        if fallback_on_error:
            logger.info("Falling back to original question")
            return question
        else:
            raise
