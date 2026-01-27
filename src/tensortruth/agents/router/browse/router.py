"""Routing logic for browse agent."""

import json
import logging
from typing import Dict, List, Optional

from llama_index.llms.ollama import Ollama

from tensortruth.agents.router.browse.prompts import (
    QUERY_GENERATION_PROMPT_TEMPLATE,
    ROUTER_PROMPT_TEMPLATE,
)
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

    async def generate_queries(self, state: BrowseState) -> List[str]:
        """Generate context-aware search queries using router LLM.

        Falls back to deterministic queries on failure.

        Args:
            state: Current browse state with query and optional conversation history

        Returns:
            List of 3 search queries
        """
        history_context = "(No conversation history)"
        if state.conversation_history and not state.conversation_history.is_empty:
            history_context = state.conversation_history.to_prompt_string()

        # Build failure context if this is a retry after all pages were rejected
        failure_context = ""
        if state.rejected_titles and len(state.rejected_titles) > 0:
            titles_list = "\n- ".join(state.rejected_titles[:5])  # Limit to 5
            failure_context = (
                "\n**Previous Search Failed:**\n"
                "The previous search queries found these pages, but they were "
                "all rejected as irrelevant:\n"
                f"- {titles_list}\n\n"
                "Generate DIFFERENT queries that approach the topic from a new angle. "
                "Avoid similar terms that led to these irrelevant results.\n"
            )
            logger.info(
                f"Retry cycle {state.search_cycles}: adding failure context "
                f"with {len(state.rejected_titles)} rejected titles"
            )

        prompt = QUERY_GENERATION_PROMPT_TEMPLATE.format(
            query=state.query,
            history_context=history_context,
            failure_context=failure_context,
        )

        try:
            response = self.llm.complete(prompt)
            result = self._parse_query_generation(response.text)

            if result and "queries" in result and len(result["queries"]) > 0:
                queries = result["queries"][:3]
                state.custom_instructions = result.get("custom_instructions")

                logger.info(f"Generated {len(queries)} queries: {queries}")
                if state.custom_instructions:
                    logger.info(f"Custom instructions: {state.custom_instructions}")

                return queries
            else:
                logger.warning("Invalid query format, using fallback")
                return self._fallback_queries(state.query)

        except Exception as e:
            logger.warning(f"Query generation failed: {e}, using fallback")
            return self._fallback_queries(state.query)

    def _parse_query_generation(self, response_text: str) -> Optional[Dict]:
        """Parse JSON from query generation response.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed dict with 'queries' and optional 'custom_instructions', or None
        """
        try:
            # Clean markdown code blocks (like intent_classifier does)
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            data = json.loads(text)

            # Validate
            if "queries" not in data or not isinstance(data["queries"], list):
                return None

            return data

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Failed to parse query generation: {e}")
            return None

    def _fallback_queries(self, query: str) -> List[str]:
        """Deterministic fallback queries.

        Args:
            query: User's original query

        Returns:
            List of 3 fallback queries
        """
        return [
            f"{query} overview",
            f"{query} information details",
            f"{query} recent 2026",
        ]

    async def route(self, state: BrowseState) -> str:
        """Route to next action based on state.

        Uses LLM with structured output for routing. Falls back to
        deterministic logic if LLM fails or returns invalid action.

        Args:
            state: Current browse state

        Returns:
            Action name: "search_web", "fetch_sources", or "done"
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
        valid_actions = ["generate_queries", "search_web", "fetch_sources", "done"]
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
        # NEW: If initial and no queries generated yet
        from tensortruth.agents.router.browse.state import WorkflowPhase

        if state.phase == WorkflowPhase.INITIAL:
            if not state.generated_queries:
                return "generate_queries"

        # Check completion conditions first (overflow or min pages met)
        if state.content_overflow:
            return "done"

        if state.pages and len(state.pages) >= state.min_pages_required:
            return "done"

        # No search results → search
        if not state.search_results or len(state.search_results) == 0:
            return "search_web"

        # Have results but no pages (all rejected or not fetched yet)
        if not state.pages or len(state.pages) == 0:
            # Already tried fetching this cycle? (fetch_iterations > 0 but no pages)
            if state.fetch_iterations > 0:
                # All pages were rejected - try new search cycle with different queries
                if state.search_cycles < state.max_search_cycles:
                    logger.info(
                        f"All pages rejected, generating new queries "
                        f"(cycle {state.search_cycles}/{state.max_search_cycles})"
                    )
                    return "generate_queries"
                else:
                    logger.warning(
                        f"Exhausted {state.max_search_cycles} search cycles, giving up"
                    )
                    return "done"
            # Haven't tried fetching yet - do it
            return "fetch_sources"

        # Need more pages and can retry
        if state.fetch_iterations < state.max_fetch_iterations:
            if state.next_url_index < len(state.search_results):
                return "fetch_sources"

        # Fallback to done if we've exhausted options
        return "done"
