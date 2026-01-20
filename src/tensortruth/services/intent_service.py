"""Intent classification service for natural language agent routing.

This service wraps the existing pure intent_classifier module with
dependency injection for the Ollama URL and model configuration.
"""

from typing import Dict, List, Optional

from llama_index.llms.ollama import Ollama

from tensortruth.app_utils.intent_classifier import (
    classify_intent,
    enhance_query_with_context,
    has_agent_triggers,
)

from .models import IntentResult


class IntentService:
    """Service for classifying user intent to route to appropriate handlers.

    Provides a clean interface for intent classification with dependency
    injection for configuration.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        classifier_model: str = "llama3.2:3b",
    ):
        """Initialize intent service.

        Args:
            ollama_url: Ollama API base URL.
            classifier_model: Model name for intent classification.
        """
        self.ollama_url = ollama_url
        self.classifier_model = classifier_model
        self._llm: Optional[Ollama] = None

    def _get_llm(self) -> Ollama:
        """Get or create the LLM instance.

        Returns:
            Ollama LLM instance for classification.
        """
        if self._llm is None:
            self._llm = Ollama(
                model=self.classifier_model,
                base_url=self.ollama_url,
                temperature=0.0,
                request_timeout=30.0,
            )
        return self._llm

    def has_triggers(self, message: str) -> bool:
        """Check if message contains trigger words that might indicate agent intent.

        This is a fast pre-filter to avoid LLM calls on obvious chat messages.

        Args:
            message: User's input message.

        Returns:
            True if message contains browse or search trigger words.
        """
        return has_agent_triggers(message)

    def classify(
        self,
        message: str,
        recent_messages: List[Dict],
        llm: Optional[Ollama] = None,
    ) -> IntentResult:
        """Classify user intent and route to appropriate handler.

        Combines trigger word pre-filtering with LLM classification.
        Only invokes the LLM if trigger words are detected.

        Args:
            message: User's input message.
            recent_messages: Recent conversation messages for context.
            llm: Optional pre-configured LLM (reuses already-loaded model).

        Returns:
            IntentResult with classified intent and extracted query.
        """
        # Fast path: no triggers = chat
        if not self.has_triggers(message):
            return IntentResult(intent="chat", query=None, reason="no_triggers")

        # Use provided LLM or create one
        classification_llm = llm or self._get_llm()

        # Classify with LLM
        raw_result = classify_intent(message, recent_messages, classification_llm)

        # Convert to service model
        result = IntentResult(
            intent=raw_result.intent,
            query=raw_result.query,
            reason=raw_result.reason,
        )

        # Enhance query if needed
        if result.intent in ("browse", "search") and result.query:
            enhanced_query = enhance_query_with_context(
                result.query, recent_messages, classification_llm
            )
            result = IntentResult(
                intent=result.intent,
                query=enhanced_query,
                reason=result.reason,
            )

        return result

    def reset(self) -> None:
        """Reset the cached LLM instance.

        Call this if the configuration changes.
        """
        self._llm = None
