"""Tests for BrowseRouter."""

from unittest.mock import MagicMock

import pytest

from tensortruth.agents.router.browse.router import BrowseRouter
from tensortruth.agents.router.browse.state import BrowseState, WorkflowPhase
from tensortruth.services.chat_history import ChatHistory, ChatHistoryMessage


@pytest.mark.asyncio
class TestBrowseRouter:
    """Test BrowseRouter routing logic."""

    async def test_router_deterministic_fallback_no_search_results(self):
        """Should route to search_web when no search results (after query generation)."""
        mock_llm = MagicMock()
        router = BrowseRouter(mock_llm)

        state = BrowseState(
            query="test",
            phase=WorkflowPhase.INITIAL,
            generated_queries=["test query 1", "test query 2", "test query 3"],
        )
        action = router._deterministic_route(state)

        assert action == "search_web"

    async def test_router_deterministic_fallback_has_results_no_pages(self):
        """Should route to fetch_sources when has results but no pages."""
        mock_llm = MagicMock()
        router = BrowseRouter(mock_llm)

        state = BrowseState(query="test", phase=WorkflowPhase.SEARCHED)
        state.search_results = [{"url": "url1"}, {"url": "url2"}]

        action = router._deterministic_route(state)

        assert action == "fetch_sources"

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
        state = BrowseState(
            query="test",
            phase=WorkflowPhase.INITIAL,
            generated_queries=["test 1", "test 2", "test 3"],
        )

        action = await router.route(state)

        # Should fallback to deterministic routing (search after queries generated)
        assert action == "search_web"

    async def test_router_parses_json_response(self):
        """Should parse JSON response from LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"action": "fetch_sources"}'
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(query="test", phase=WorkflowPhase.SEARCHED)

        action = await router.route(state)

        assert action == "fetch_sources"

    async def test_router_handles_invalid_json(self):
        """Should fallback if JSON is invalid."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "invalid json"
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(
            query="test",
            phase=WorkflowPhase.INITIAL,
            generated_queries=["test 1", "test 2", "test 3"],
        )

        action = await router.route(state)

        # Should fallback to deterministic routing (search after queries generated)
        assert action == "search_web"

    async def test_router_validates_action(self):
        """Should validate action is in allowed list."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"action": "invalid_action"}'
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(
            query="test",
            phase=WorkflowPhase.INITIAL,
            generated_queries=["test 1", "test 2", "test 3"],
        )

        action = await router.route(state)

        # Should fallback due to invalid action
        assert action in ["generate_queries", "search_web", "fetch_sources", "done"]


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


@pytest.mark.asyncio
class TestQueryGeneration:
    """Test query generation with conversation history."""

    async def test_generate_queries_without_history(self):
        """Router generates fallback queries when no history."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """{
            "queries": [
                "neural networks overview",
                "neural networks technical details",
                "neural networks recent 2026"
            ],
            "custom_instructions": null
        }"""
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(
            query="neural networks",
            phase=WorkflowPhase.INITIAL,
            conversation_history=None,
        )

        queries = await router.generate_queries(state)

        assert len(queries) == 3
        assert all("neural networks" in q.lower() for q in queries)
        assert state.custom_instructions is None

    async def test_generate_queries_with_context_resolution(self):
        """Router resolves 'this' using history."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """{
            "queries": [
                "backpropagation algorithm overview",
                "backpropagation implementation details",
                "backpropagation recent advances"
            ],
            "custom_instructions": null
        }"""
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)

        history = ChatHistory(
            messages=(
                ChatHistoryMessage(role="user", content="What is backpropagation?"),
                ChatHistoryMessage(
                    role="assistant",
                    content="Backpropagation is an algorithm for training neural networks...",
                ),
            )
        )
        state = BrowseState(
            query="browse more about this",
            phase=WorkflowPhase.INITIAL,
            conversation_history=history,
        )

        queries = await router.generate_queries(state)

        assert len(queries) == 3
        assert any("backpropagation" in q.lower() for q in queries)

    async def test_extract_custom_instructions(self):
        """Router extracts custom instructions from query."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """{
            "queries": [
                "transformer SOTA methods",
                "transformer state of the art 2026",
                "latest transformer research"
            ],
            "custom_instructions": "focus on state-of-the-art methods only"
        }"""
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(
            query="transformers, focus on SOTA only", phase=WorkflowPhase.INITIAL
        )

        queries = await router.generate_queries(state)

        assert len(queries) == 3
        assert state.custom_instructions is not None
        assert (
            "sota" in state.custom_instructions.lower()
            or "state-of-the-art" in state.custom_instructions.lower()
        )

    async def test_fallback_on_llm_failure(self):
        """Falls back to deterministic queries on failure."""
        mock_llm = MagicMock()
        mock_llm.complete = MagicMock(side_effect=Exception("LLM failed"))

        router = BrowseRouter(mock_llm)
        state = BrowseState(query="test query", phase=WorkflowPhase.INITIAL)

        queries = await router.generate_queries(state)

        assert len(queries) == 3
        assert "test query overview" in queries[0]
        assert "test query information details" in queries[1]
        assert "test query recent 2026" in queries[2]

    async def test_fallback_on_invalid_json(self):
        """Falls back to deterministic queries on invalid JSON."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "invalid json response"
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(query="test query", phase=WorkflowPhase.INITIAL)

        queries = await router.generate_queries(state)

        assert len(queries) == 3
        assert all("test query" in q for q in queries)

    async def test_parse_query_generation_cleans_markdown(self):
        """Parser cleans markdown code blocks from response."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """```json
{
    "queries": ["query1", "query2", "query3"],
    "custom_instructions": null
}
```"""
        mock_llm.complete = MagicMock(return_value=mock_response)

        router = BrowseRouter(mock_llm)
        state = BrowseState(query="test", phase=WorkflowPhase.INITIAL)

        queries = await router.generate_queries(state)

        assert len(queries) == 3
        assert queries == ["query1", "query2", "query3"]

    async def test_deterministic_route_generates_queries_at_start(self):
        """Deterministic router generates queries at INITIAL phase with no queries."""
        mock_llm = MagicMock()
        router = BrowseRouter(mock_llm)

        state = BrowseState(
            query="test", phase=WorkflowPhase.INITIAL, generated_queries=None
        )

        action = router._deterministic_route(state)

        assert action == "generate_queries"

    async def test_deterministic_route_searches_after_query_generation(self):
        """Deterministic router moves to search after queries generated."""
        mock_llm = MagicMock()
        router = BrowseRouter(mock_llm)

        state = BrowseState(
            query="test",
            phase=WorkflowPhase.INITIAL,
            generated_queries=["query1", "query2", "query3"],
        )

        action = router._deterministic_route(state)

        assert action == "search_web"
