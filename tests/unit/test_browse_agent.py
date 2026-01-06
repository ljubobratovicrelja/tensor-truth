"""Unit tests for browse agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensortruth.utils.agent_framework import AgentAction, AgentActionType, AgentState
from tensortruth.utils.browse_agent import BrowseAgent, normalize_url


class TestURLNormalization:
    """Test URL normalization for duplicate detection."""

    def test_normalize_trailing_slash(self):
        """Test that trailing slashes are removed."""
        assert normalize_url("https://example.com/") == "https://example.com"
        assert normalize_url("https://example.com/path/") == "https://example.com/path"

    def test_normalize_root_path(self):
        """Test that root path keeps trailing slash."""
        assert normalize_url("https://example.com/") == "https://example.com"

    def test_normalize_fragment(self):
        """Test that URL fragments are removed."""
        assert (
            normalize_url("https://example.com/page#section")
            == "https://example.com/page"
        )

    def test_normalize_case(self):
        """Test that domain is lowercased."""
        assert normalize_url("https://Example.COM/Path") == "https://example.com/Path"

    def test_normalize_query_preserved(self):
        """Test that query parameters are preserved."""
        assert (
            normalize_url("https://example.com/page?foo=bar")
            == "https://example.com/page?foo=bar"
        )

    def test_normalize_identical_urls(self):
        """Test that variations of same URL normalize to same result."""
        urls = [
            "https://example.com/page",
            "https://example.com/page/",
            "https://EXAMPLE.com/page",
            "https://example.com/page#intro",
        ]
        normalized = [normalize_url(url) for url in urls]
        assert len(set(normalized)) == 1  # All should be identical


class TestActionParsing:
    """Test LLM response parsing into AgentAction."""

    def setup_method(self):
        """Create agent instance for testing."""
        self.agent = BrowseAgent(name="browse", description="Test")

    def test_parse_search_action(self):
        """Test parsing SEARCH action."""
        response = """
        THINKING: Need to find Python tutorials
        ACTION: SEARCH
        QUERY: best python tutorials 2024
        REASON: User wants to learn Python
        """
        action = self.agent._parse_action(response, AgentState(goal="test"))

        assert action.type == AgentActionType.SEARCH
        assert "python tutorials" in action.query.lower()
        assert action.reasoning == "User wants to learn Python"

    def test_parse_fetch_page_action(self):
        """Test parsing FETCH_PAGE action."""
        response = """
        THINKING: Need details from this page
        ACTION: FETCH_PAGE
        URL: https://example.com/page
        REASON: Contains relevant information
        """
        action = self.agent._parse_action(response, AgentState(goal="test"))

        assert action.type == AgentActionType.FETCH_PAGE
        assert action.url == "https://example.com/page"
        assert "relevant information" in action.reasoning.lower()

    def test_parse_conclude_action(self):
        """Test parsing CONCLUDE action."""
        response = """
        THINKING: I have enough information
        ACTION: CONCLUDE
        REASON: Goal satisfied
        """
        action = self.agent._parse_action(response, AgentState(goal="test"))

        assert action.type == AgentActionType.CONCLUDE
        assert action.reasoning == "Goal satisfied"

    def test_parse_malformed_response_fallback(self):
        """Test fallback to CONCLUDE on malformed response."""
        response = "This is not a valid response"
        action = self.agent._parse_action(response, AgentState(goal="test"))

        assert action.type == AgentActionType.CONCLUDE  # Safe fallback

    def test_parse_missing_query_fallback(self):
        """Test fallback when SEARCH action missing query."""
        response = """
        THINKING: Need to search
        ACTION: SEARCH
        REASON: Missing query
        """
        action = self.agent._parse_action(response, AgentState(goal="test"))

        assert action.type == AgentActionType.CONCLUDE  # Falls back

    def test_parse_missing_url_fallback(self):
        """Test fallback when FETCH_PAGE action missing URL."""
        response = """
        THINKING: Need to fetch page
        ACTION: FETCH_PAGE
        REASON: Missing URL
        """
        action = self.agent._parse_action(response, AgentState(goal="test"))

        assert action.type == AgentActionType.CONCLUDE  # Falls back

    def test_parse_case_insensitive(self):
        """Test parsing is case insensitive."""
        response = """
        action: search
        query: test query
        """
        action = self.agent._parse_action(response, AgentState(goal="test"))

        assert action.type == AgentActionType.SEARCH
        assert action.query == "test query"


class TestAgentStateManagement:
    """Test agent state tracking."""

    def test_state_initialization(self):
        """Test state initialization for browse agent."""
        state = AgentState(goal="Find Python docs", max_iterations=5)

        assert state.current_iteration == 0
        assert len(state.searches_performed) == 0
        assert len(state.pages_visited) == 0
        assert state.termination_reason is None

    def test_state_update_after_search(self):
        """Test state updates after search execution."""
        state = AgentState(goal="Test", max_iterations=5)
        state.searches_performed.append(
            ("test query", [{"url": "test.com", "title": "Test"}])
        )
        state.current_iteration += 1

        assert len(state.searches_performed) == 1
        assert state.current_iteration == 1
        assert state.searches_performed[0][0] == "test query"

    def test_state_update_after_fetch(self):
        """Test state updates after page fetch."""
        state = AgentState(goal="Test", max_iterations=5)
        state.pages_visited.append(("http://example.com", "Example", "Summary text"))

        assert len(state.pages_visited) == 1
        url, title, summary = state.pages_visited[0]
        assert url == "http://example.com"
        assert title == "Example"
        assert summary == "Summary text"

    def test_state_failed_fetches(self):
        """Test tracking of failed fetches."""
        state = AgentState(goal="Test", max_iterations=5)
        state.failed_fetches.append(("http://broken.com", "Timeout"))

        assert len(state.failed_fetches) == 1
        assert state.failed_fetches[0][0] == "http://broken.com"
        assert state.failed_fetches[0][1] == "Timeout"


@pytest.mark.asyncio
class TestAgentExecution:
    """Test agent execution with mocked dependencies."""

    def setup_method(self):
        """Create agent instance for testing."""
        self.agent = BrowseAgent(name="browse", description="Test")

    async def test_execute_search_action(self):
        """Test search action execution."""
        state = AgentState(goal="Test", max_iterations=5)
        action = AgentAction(type=AgentActionType.SEARCH, query="test query")

        with patch(
            "tensortruth.utils.browse_agent.search_duckduckgo",
            new=AsyncMock(
                return_value=[
                    {"url": "test.com", "title": "Test", "snippet": "Test snippet"}
                ]
            ),
        ):
            await self.agent._execute_search(action, state)

        assert len(state.searches_performed) == 1
        assert state.searches_performed[0][0] == "test query"
        assert len(state.searches_performed[0][1]) == 1

    async def test_execute_fetch_page_action_success(self):
        """Test successful page fetch."""
        state = AgentState(goal="Test", max_iterations=5)
        action = AgentAction(
            type=AgentActionType.FETCH_PAGE,
            url="http://test.com",
            title="Test Page",
        )

        mock_content = "This is test page content with lots of text."

        with patch(
            "tensortruth.utils.browse_agent.fetch_page_as_markdown",
            new=AsyncMock(return_value=(mock_content, "success", None)),
        ):
            await self.agent._execute_fetch_page(action, state)

        assert len(state.pages_visited) == 1
        assert state.pages_visited[0][0] == "http://test.com"
        assert state.pages_visited[0][1] == "Test Page"

    async def test_execute_fetch_page_action_failure(self):
        """Test failed page fetch."""
        state = AgentState(goal="Test", max_iterations=5)
        action = AgentAction(type=AgentActionType.FETCH_PAGE, url="http://broken.com")

        with patch(
            "tensortruth.utils.browse_agent.fetch_page_as_markdown",
            new=AsyncMock(return_value=(None, "timeout", "Connection timeout")),
        ):
            await self.agent._execute_fetch_page(action, state)

        assert len(state.pages_visited) == 0
        assert len(state.failed_fetches) == 1
        assert state.failed_fetches[0][0] == "http://broken.com"

    async def test_reason_next_action(self):
        """Test reasoning with mocked LLM."""
        state = AgentState(goal="Find Python docs", max_iterations=5)

        mock_llm_response = MagicMock()
        mock_llm_response.text = """
        THINKING: Should start with a search
        ACTION: SEARCH
        QUERY: python documentation
        REASON: Need to find official docs
        """

        with patch("tensortruth.utils.browse_agent.Ollama") as mock_ollama:
            mock_llm = AsyncMock()
            mock_llm.acomplete.return_value = mock_llm_response
            mock_ollama.return_value = mock_llm

            thinking, action = await self.agent.reason_next_action(
                state, "test-model", "http://localhost:11434", 16384
            )

            assert "Should start with a search" in thinking
            assert action.type == AgentActionType.SEARCH
            assert "python documentation" in action.query.lower()

    async def test_synthesize_final_answer(self):
        """Test final answer synthesis."""
        state = AgentState(goal="Test", max_iterations=5)
        state.searches_performed = [("test query", [])]
        state.pages_visited = [("http://test.com", "Test", "Test content summary")]

        mock_llm_response = MagicMock()
        mock_llm_response.text = "This is the synthesized answer"

        with patch("tensortruth.utils.browse_agent.Ollama") as mock_ollama:
            mock_llm = AsyncMock()
            mock_llm.acomplete.return_value = mock_llm_response
            mock_ollama.return_value = mock_llm

            answer = await self.agent.synthesize_final_answer(
                state, "test-model", "http://localhost:11434", 16384
            )

            assert "synthesized answer" in answer.lower()

    async def test_synthesize_with_no_data(self):
        """Test synthesis when no data was gathered."""
        state = AgentState(goal="Test", max_iterations=5)
        # No searches or pages visited

        answer = await self.agent.synthesize_final_answer(
            state, "test-model", "http://localhost:11434", 16384
        )

        assert "unable to gather information" in answer.lower()

    async def test_reason_next_action_llm_error_with_existing_data(self):
        """Test fallback to CONCLUDE when LLM fails but state has data."""
        state = AgentState(goal="Test", max_iterations=5)
        state.searches_performed.append(("query", [{"url": "test.com"}]))
        # Has existing data

        with patch("tensortruth.utils.browse_agent.Ollama") as mock_ollama:
            mock_llm = AsyncMock()
            mock_llm.acomplete.side_effect = Exception("LLM timeout")
            mock_ollama.return_value = mock_llm

            thinking, action = await self.agent.reason_next_action(
                state, "test-model", "http://localhost:11434", 16384
            )

            assert action.type == AgentActionType.CONCLUDE
            assert "timeout" in thinking.lower() or "error" in thinking.lower()

    async def test_reason_next_action_llm_error_first_iteration(self):
        """Test fallback to SEARCH when LLM fails on first iteration."""
        state = AgentState(goal="Find Python docs", max_iterations=5)
        # No existing data - first iteration

        with patch("tensortruth.utils.browse_agent.Ollama") as mock_ollama:
            mock_llm = AsyncMock()
            mock_llm.acomplete.side_effect = Exception("LLM timeout")
            mock_ollama.return_value = mock_llm

            thinking, action = await self.agent.reason_next_action(
                state, "test-model", "http://localhost:11434", 16384
            )

            assert action.type == AgentActionType.SEARCH
            assert action.query == "Find Python docs"  # Falls back to goal
            assert "fallback" in action.reasoning.lower()


class TestDetermineRequiredAction:
    """Test _determine_required_action helper method."""

    def setup_method(self):
        """Create agent instance for testing."""
        self.agent = BrowseAgent(name="browse", description="Test")

    def test_no_searches_requires_search(self):
        """Test that agent must search when no searches performed."""
        state = AgentState(goal="Test", max_iterations=10)
        # No searches, no pages

        result = self.agent._determine_required_action(state)

        assert "MUST do SEARCH" in result
        assert "no data yet" in result.lower()

    def test_searches_but_no_pages_requires_fetch(self):
        """Test that agent must fetch when searches exist but no pages."""
        state = AgentState(goal="Test", max_iterations=10)
        state.searches_performed.append(("query", [{"url": "test.com"}]))
        # Has searches, no pages

        result = self.agent._determine_required_action(state)

        assert "MUST do FETCH_PAGE" in result
        assert "haven't fetched any pages" in result.lower()

    def test_few_pages_requires_more(self):
        """Test that agent must fetch more when below minimum."""
        state = AgentState(goal="Test", max_iterations=10)
        state.searches_performed.append(("query", []))
        state.pages_visited.append(("url1", "Title1", "Summary1"))
        state.pages_visited.append(("url2", "Title2", "Summary2"))
        # Has 2 pages, need 5

        result = self.agent._determine_required_action(state, min_required_pages=5)

        assert "MUST do FETCH_PAGE" in result
        assert "at least 5 sources" in result.lower()
        assert "currently have 2" in result.lower()

    def test_enough_pages_allows_choice(self):
        """Test that agent can choose when minimum pages reached."""
        state = AgentState(goal="Test", max_iterations=10)
        state.searches_performed.append(("query", []))
        for i in range(5):
            state.pages_visited.append((f"url{i}", f"Title{i}", f"Summary{i}"))
        # Has 5 pages, meets minimum

        result = self.agent._determine_required_action(state, min_required_pages=5)

        assert "MAY do SEARCH or FETCH_PAGE" in result
        assert "5 sources" in result.lower()


class TestBrowseAgentPublicAPI:
    """Test public browse_agent function."""

    @patch("tensortruth.utils.browse_agent.asyncio.run")
    def test_browse_agent_sync_wrapper(self, mock_asyncio_run):
        """Test that browse_agent properly wraps async agent.run()."""
        from tensortruth.utils.browse_agent import browse_agent

        mock_state = MagicMock()
        mock_state.final_answer = "Test answer"
        mock_asyncio_run.return_value = mock_state

        result = browse_agent(
            goal="Test goal",
            model_name="test-model",
            ollama_url="http://localhost:11434",
        )

        assert mock_asyncio_run.called
        assert result.final_answer == "Test answer"
