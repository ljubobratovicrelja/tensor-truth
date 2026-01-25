"""Tests for BrowseRouter."""

from unittest.mock import MagicMock

import pytest

from tensortruth.agents.router.browse.router import BrowseRouter
from tensortruth.agents.router.browse.state import BrowseState, WorkflowPhase


@pytest.mark.asyncio
class TestBrowseRouter:
    """Test BrowseRouter routing logic."""

    async def test_router_deterministic_fallback_no_search_results(self):
        """Should route to search_web when no search results."""
        mock_llm = MagicMock()
        router = BrowseRouter(mock_llm)

        state = BrowseState(query="test", phase=WorkflowPhase.INITIAL)
        action = router._deterministic_route(state)

        assert action == "search_web"

    async def test_router_deterministic_fallback_has_results_no_pages(self):
        """Should route to fetch_pages_batch when has results but no pages."""
        mock_llm = MagicMock()
        router = BrowseRouter(mock_llm)

        state = BrowseState(query="test", phase=WorkflowPhase.SEARCHED)
        state.search_results = [{"url": "url1"}, {"url": "url2"}]

        action = router._deterministic_route(state)

        assert action == "fetch_pages_batch"

    async def test_router_deterministic_fallback_min_pages_met(self):
        """Should route to done when min pages requirement met."""
        mock_llm = MagicMock()
        router = BrowseRouter(mock_llm)

        state = BrowseState(
            query="test", phase=WorkflowPhase.FETCHED, min_pages_required=3
        )
        state.search_results = [{"url": "url1"}]
        state.pages = [{"url": "url1"}, {"url": "url2"}, {"url": "url3"}]

        action = router._deterministic_route(state)

        assert action == "done"

    async def test_router_deterministic_fallback_overflow(self):
        """Should route to done when content overflow."""
        mock_llm = MagicMock()
        router = BrowseRouter(mock_llm)

        state = BrowseState(query="test", phase=WorkflowPhase.FETCHED)
        state.content_overflow = True
        state.pages = [{"url": "url1"}]

        action = router._deterministic_route(state)

        assert action == "done"

    async def test_router_uses_llm_with_structured_output(self):
        """Should call LLM with structured output format."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"action": "search_web"}'
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(query="test", phase=WorkflowPhase.INITIAL)

        action = await router.route(state)

        # Should have called LLM
        assert mock_llm.complete.called
        assert action == "search_web"

    async def test_router_falls_back_on_llm_failure(self):
        """Should use deterministic fallback if LLM fails."""
        mock_llm = MagicMock()
        mock_llm.complete = MagicMock(side_effect=Exception("LLM error"))

        router = BrowseRouter(mock_llm)
        state = BrowseState(query="test", phase=WorkflowPhase.INITIAL)

        action = await router.route(state)

        # Should fallback to deterministic routing
        assert action == "search_web"

    async def test_router_parses_json_response(self):
        """Should parse JSON response from LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"action": "fetch_pages_batch"}'
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(query="test", phase=WorkflowPhase.SEARCHED)

        action = await router.route(state)

        assert action == "fetch_pages_batch"

    async def test_router_handles_invalid_json(self):
        """Should fallback if JSON is invalid."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "invalid json"
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(query="test", phase=WorkflowPhase.INITIAL)

        action = await router.route(state)

        # Should fallback to deterministic routing
        assert action == "search_web"

    async def test_router_validates_action(self):
        """Should validate action is in allowed list."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"action": "invalid_action"}'
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(query="test", phase=WorkflowPhase.INITIAL)

        action = await router.route(state)

        # Should fallback due to invalid action
        assert action in ["search_web", "fetch_pages_batch", "done"]


class TestRouterPromptBuilding:
    """Test prompt building."""

    def test_build_prompt_includes_state(self):
        """Should build prompt with state information."""
        mock_llm = MagicMock()
        router = BrowseRouter(mock_llm)

        state = BrowseState(query="test", phase=WorkflowPhase.SEARCHED)
        state.search_results = [{"url": "url1"}]
        state.pages = []
        state.min_pages_required = 3
        state.actions_taken = ["search_web"]

        prompt = router._build_prompt(state)

        assert "Results=1" in prompt
        assert "Pages=0/3" in prompt
        assert "Last=search_web" in prompt
