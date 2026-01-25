"""Routing logic for browse agent."""

import json
import logging
from typing import Optional

from llama_index.llms.ollama import Ollama

from tensortruth.agents.router.browse.prompts import ROUTER_PROMPT_TEMPLATE
from tensortruth.agents.router.browse.state import BrowseState

logger = logging.getLogger(__name__)


class BrowseRouter:
    """Routes browse agent to next action using LLM with deterministic fallback."""

    def __init__(self, llm: Ollama):
        """Initialize router.

        Args:
            llm: Small, fast model for routing decisions
        """
        self.llm = llm

    async def route(self, state: BrowseState) -> str:
        """Route to next action based on state.

        Uses LLM with structured output for routing. Falls back to
        deterministic logic if LLM fails or returns invalid action.

        Args:
            state: Current browse state

        Returns:
            Action name: "search_web", "fetch_pages_batch", or "done"
        """
        prompt = self._build_prompt(state)

        try:
            # Call LLM for routing decision
            response = self.llm.complete(prompt)
            action = self._parse_action(response.text)

            if action and self._validate_action(action):
                logger.debug(f"Router LLM selected action: {action}")
                return action
            else:
                logger.warning(
                    f"Router LLM returned invalid action: {action}, using fallback"
                )
                return self._deterministic_route(state)

        except Exception as e:
            logger.warning(f"Router LLM failed: {e}, using deterministic fallback")
            return self._deterministic_route(state)

    def _build_prompt(self, state: BrowseState) -> str:
        """Build routing prompt from state.

        Args:
            state: Current browse state

        Returns:
            Formatted prompt string
        """
        n_results = len(state.search_results) if state.search_results else 0
        page_count = len(state.pages) if state.pages else 0
        last_action = state.actions_taken[-1] if state.actions_taken else "none"

        return ROUTER_PROMPT_TEMPLATE.format(
            n_results=n_results,
            page_count=page_count,
            min_pages=state.min_pages_required,
            last_action=last_action,
        )

    def _parse_action(self, response_text: str) -> Optional[str]:
        """Parse action from LLM response.

        Args:
            response_text: Raw LLM response

        Returns:
            Action name or None if parsing fails
        """
        try:
            # Try to parse as JSON
            data = json.loads(response_text.strip())
            return data.get("action")
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse JSON from response: {response_text}")
            return None

    def _validate_action(self, action: str) -> bool:
        """Validate action is in allowed list.

        Args:
            action: Action name

        Returns:
            True if valid, False otherwise
        """
        valid_actions = ["search_web", "fetch_pages_batch", "done"]
        return action in valid_actions

    def _deterministic_route(self, state: BrowseState) -> str:
        """Deterministic routing fallback.

        This logic ensures the agent always makes progress even if the
        LLM fails or returns invalid actions.

        Args:
            state: Current browse state

        Returns:
            Action name guaranteed to be valid
        """
        # Check completion conditions first (overflow or min pages met)
        if state.content_overflow:
            return "done"

        if state.pages and len(state.pages) >= state.min_pages_required:
            return "done"

        # No search results → search
        if not state.search_results or len(state.search_results) == 0:
            return "search_web"

        # Have results but no pages → fetch
        if not state.pages or len(state.pages) == 0:
            return "fetch_pages_batch"

        # Need more pages and can retry
        if state.fetch_iterations < state.max_fetch_iterations:
            if state.next_url_index < len(state.search_results):
                return "fetch_pages_batch"

        # Fallback to done if we've exhausted options
        return "done"
