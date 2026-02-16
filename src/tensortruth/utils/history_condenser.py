"""
Utility for condensing chat history and questions into standalone queries.

This module provides functions to transform follow-up questions that reference
previous conversation context into self-contained standalone queries suitable
for RAG retrieval or web search.
"""

import logging

from llama_index.llms.ollama import Ollama

from tensortruth.core.constants import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)


def create_condenser_llm(
    base_llm: Ollama,
    temperature: float = 0.0,
    thinking: bool = False,
    timeout: float = 30.0,
) -> Ollama:
    """Return the shared orchestrator LLM singleton for query condensation.

    Reuses the cached non-thinking LLM so that Ollama never sees a
    ``num_ctx`` change (which would trigger a model reload).  The
    singleton's temperature (0.2) is close enough to 0.0 for condensation.

    Args:
        base_llm: The base Ollama LLM to derive model/url/context_window from.
        temperature: Ignored (kept for API compatibility).
        thinking: Ignored (kept for API compatibility).
        timeout: Ignored (kept for API compatibility).

    Returns:
        The shared orchestrator Ollama instance.
    """
    from tensortruth.core.ollama import get_orchestrator_llm

    model = getattr(base_llm, "model", DEFAULT_MODEL)
    base_url = getattr(base_llm, "base_url", DEFAULT_OLLAMA_BASE_URL)
    context_window = getattr(base_llm, "context_window", 16384)

    return get_orchestrator_llm(model, base_url, context_window)


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
