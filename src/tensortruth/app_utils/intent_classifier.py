"""Intent classification for natural language agent routing.

This module provides intent detection to route user messages to appropriate
handlers (chat, browse agent, web search) based on natural language patterns.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Optional

from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)

# Trigger word patterns for pre-filtering
BROWSE_TRIGGERS = re.compile(
    r"\b(browse|research|find out|look up|investigate|dig into)\b", re.IGNORECASE
)
SEARCH_TRIGGERS = re.compile(
    r"\b(search|google|web search|look online|search the web)\b", re.IGNORECASE
)

# Classification prompt template
CLASSIFICATION_PROMPT = """\
You are an intent classifier. Analyze the user's message and classify their intent.

Recent conversation context:
{context}

User message: "{message}"

Classify the intent as one of:
- "browse": User wants autonomous web research (browse, research, find out, look up)
- "search": User wants a quick web search with summary (search, google, look online)
- "chat": Normal conversation, questions about loaded documents, or follow-up

Output ONLY valid JSON (no markdown, no explanation):
{{"intent": "chat|browse|search", "query": "extracted query or null", "reason": "brief"}}

Rules:
- If message contains "browse", "research", "find out", "look up" + a topic → browse
- If message contains "search", "google", "web search" + a topic → search
- If message is a follow-up or question about existing context → chat
- If ambiguous, default to chat
- Extract the research/search query from the message (remove trigger words)

Examples:
- "Browse the latest AI news" → {{"intent": "browse", "query": "latest AI news"}}
- "Research transformers" → {{"intent": "browse", "query": "transformers"}}
- "Search for Python 3.12 features" → {{"intent": "search", "query": "Python 3.12 features"}}
- "What does the docs say about tensors?" → {{"intent": "chat", "query": null}}
- "Tell me more about that" → {{"intent": "chat", "query": null}}
"""


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: Literal["chat", "browse", "search"]
    query: Optional[str]
    reason: str


def has_agent_triggers(message: str) -> bool:
    """Check if message contains trigger words that might indicate agent intent.

    This is a fast pre-filter to avoid LLM calls on obvious chat messages.

    Args:
        message: User's input message

    Returns:
        True if message contains browse or search trigger words
    """
    return bool(BROWSE_TRIGGERS.search(message)) or bool(
        SEARCH_TRIGGERS.search(message)
    )


def _build_context_string(recent_messages: list[dict], max_messages: int = 4) -> str:
    """Build context string from recent messages for classification.

    Args:
        recent_messages: List of recent message dicts with 'role' and 'content'
        max_messages: Maximum number of messages to include

    Returns:
        Formatted context string
    """
    if not recent_messages:
        return "(No previous context)"

    # Take last N messages, excluding commands
    relevant = [
        m
        for m in recent_messages[-max_messages:]
        if m.get("role") in ("user", "assistant")
    ]

    if not relevant:
        return "(No previous context)"

    lines = []
    for msg in relevant:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")[:200]  # Truncate long messages
        if len(msg.get("content", "")) > 200:
            content += "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _parse_classification_response(response_text: str) -> IntentResult:
    """Parse LLM classification response into IntentResult.

    Args:
        response_text: Raw LLM response text

    Returns:
        Parsed IntentResult, defaults to chat on parse failure
    """
    try:
        # Clean up response - remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)

        intent = data.get("intent", "chat")
        if intent not in ("chat", "browse", "search"):
            intent = "chat"

        return IntentResult(
            intent=intent,
            query=data.get("query"),
            reason=data.get("reason", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse classification response: {e}")
        return IntentResult(intent="chat", query=None, reason="parse_error")


def classify_intent(
    message: str,
    recent_messages: list[dict],
    llm: Ollama,
) -> IntentResult:
    """Classify user intent using LLM.

    Args:
        message: User's input message
        recent_messages: Recent conversation messages for context
        llm: Ollama LLM instance for classification

    Returns:
        IntentResult with classified intent and extracted query
    """
    context = _build_context_string(recent_messages)
    prompt = CLASSIFICATION_PROMPT.format(
        context=context,
        message=message,
    )

    try:
        response = llm.complete(prompt)
        return _parse_classification_response(str(response))
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return IntentResult(intent="chat", query=None, reason=f"error: {e}")


def enhance_query_with_context(
    query: str,
    recent_messages: list[dict],
    llm: Ollama,
) -> str:
    """Enhance a contextual query using conversation history.

    For queries like "browse more about this", extract the topic from context.

    Args:
        query: The extracted query (may be incomplete)
        recent_messages: Recent conversation for context
        llm: Ollama LLM for enhancement

    Returns:
        Enhanced query with context filled in
    """
    # Check if query references context
    contextual_patterns = [
        r"\b(this|that|it|these|those)\b",
        r"\b(more about|more on|further)\b",
        r"\b(same|similar)\b",
    ]

    needs_enhancement = any(
        re.search(pattern, query, re.IGNORECASE) for pattern in contextual_patterns
    )

    if not needs_enhancement:
        return query

    # Build enhancement prompt
    context = _build_context_string(recent_messages)
    enhancement_prompt = f"""Given this conversation context:
{context}

The user wants to research: "{query}"

The query references something from the context. Extract what they're referring to
and create a complete, standalone search query.

Output ONLY the enhanced query (no explanation, no quotes):"""

    try:
        response = llm.complete(enhancement_prompt)
        enhanced = str(response).strip().strip("\"'")
        if enhanced and len(enhanced) > 3:
            logger.debug(f"Enhanced query: '{query}' → '{enhanced}'")
            return enhanced
    except Exception as e:
        logger.warning(f"Query enhancement failed: {e}")

    return query


def detect_and_classify(
    message: str,
    recent_messages: list[dict],
    llm: Optional[Ollama] = None,
    ollama_url: str = "http://localhost:11434",
    classifier_model: str = "llama3.2:3b",
) -> IntentResult:
    """Main entry point: detect triggers and classify if needed.

    This combines the trigger word pre-filter with LLM classification.
    Only invokes the LLM if trigger words are detected.

    Args:
        message: User's input message
        recent_messages: Recent conversation messages
        llm: Optional pre-configured LLM (created if not provided)
        ollama_url: Ollama API URL (used if llm not provided)
        classifier_model: Model to use for classification

    Returns:
        IntentResult with classified intent
    """
    # Fast path: no triggers = chat
    if not has_agent_triggers(message):
        return IntentResult(intent="chat", query=None, reason="no_triggers")

    # Create LLM if not provided
    if llm is None:
        llm = Ollama(
            model=classifier_model,
            base_url=ollama_url,
            temperature=0.0,  # Deterministic for classification
            request_timeout=30.0,
        )

    # Classify with LLM
    result = classify_intent(message, recent_messages, llm)

    # Enhance query if needed
    if result.intent in ("browse", "search") and result.query:
        result.query = enhance_query_with_context(result.query, recent_messages, llm)

    return result
