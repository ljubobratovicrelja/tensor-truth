"""Unit tests for agent framework."""

from unittest.mock import patch

import pytest

from tensortruth.utils.agent_framework import (
    AgentAction,
    AgentActionType,
    AgentRegistry,
    AgentState,
    BaseAgent,
    register_agent,
)


class TestAgentState:
    """Test AgentState dataclass."""

    def test_state_initialization(self):
        """Test state initializes with correct defaults."""
        state = AgentState(goal="Test goal", max_iterations=5)

        assert state.goal == "Test goal"
        assert state.max_iterations == 5
        assert state.current_iteration == 0
        assert len(state.searches_performed) == 0
        assert len(state.pages_visited) == 0
        assert state.termination_reason is None
        assert state.final_answer is None

    def test_state_accumulation(self):
        """Test state accumulates observations correctly."""
        state = AgentState(goal="Test", max_iterations=5)

        # Add search
        state.searches_performed.append(
            ("test query", [{"url": "test.com", "title": "Test"}])
        )
        assert len(state.searches_performed) == 1

        # Add page visit
        state.pages_visited.append(("test.com", "Test Page", "Summary"))
        assert len(state.pages_visited) == 1

        # Increment iteration
        state.current_iteration += 1
        assert state.current_iteration == 1


class TestAgentAction:
    """Test AgentAction dataclass."""

    def test_search_action(self):
        """Test SEARCH action creation."""
        action = AgentAction(
            type=AgentActionType.SEARCH,
            query="python docs",
            reasoning="Need to find documentation",
        )

        assert action.type == AgentActionType.SEARCH
        assert action.query == "python docs"
        assert action.reasoning == "Need to find documentation"
        assert action.url is None

    def test_fetch_page_action(self):
        """Test FETCH_PAGE action creation."""
        action = AgentAction(
            type=AgentActionType.FETCH_PAGE,
            url="https://example.com",
            title="Example Page",
            reasoning="Need details from this page",
        )

        assert action.type == AgentActionType.FETCH_PAGE
        assert action.url == "https://example.com"
        assert action.title == "Example Page"
        assert action.query is None

    def test_conclude_action(self):
        """Test CONCLUDE action creation."""
        action = AgentAction(
            type=AgentActionType.CONCLUDE, reasoning="Have enough information"
        )

        assert action.type == AgentActionType.CONCLUDE
        assert action.reasoning == "Have enough information"
        assert action.query is None
        assert action.url is None


class MockAgent(BaseAgent):
    """Mock agent for testing base class."""

    async def reason_next_action(self, state, model_name, ollama_url, context_window):
        """Mock reasoning - always concludes."""
        return "Thinking about concluding", AgentAction(type=AgentActionType.CONCLUDE)

    async def execute_action(self, action, state, progress_callback=None):
        """Mock action execution."""
        pass

    async def synthesize_final_answer(
        self,
        state,
        model_name,
        ollama_url,
        context_window,
        progress_callback=None,
        synthesis_model=None,
    ):
        """Mock synthesis."""
        return "Mock final answer"


class TestBaseAgent:
    """Test BaseAgent abstract class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = MockAgent(name="test", description="Test agent")

        assert agent.name == "test"
        assert agent.description == "Test agent"

    def test_default_config(self):
        """Test default configuration."""
        agent = MockAgent(name="test", description="Test")
        config = agent.get_default_config()

        assert "max_iterations" in config
        assert "max_time_seconds" in config
        assert config["max_iterations"] == 10
        assert config["max_time_seconds"] == 300

    def test_should_continue_iteration_limit(self):
        """Test termination on iteration limit."""
        agent = MockAgent(name="test", description="Test")
        state = AgentState(goal="Test", max_iterations=5)
        state.current_iteration = 5

        should_continue, reason = agent.should_continue(state)

        assert not should_continue
        assert reason == "max_iterations"

    def test_should_continue_timeout(self):
        """Test termination on timeout."""
        agent = MockAgent(name="test", description="Test")
        state = AgentState(goal="Test", max_iterations=10)
        state.start_time = 0  # Long time ago

        with patch("time.time", return_value=400):  # 400 seconds elapsed
            should_continue, reason = agent.should_continue(state)

        assert not should_continue
        assert reason == "timeout"

    def test_should_continue_normal(self):
        """Test continuation in normal case."""
        agent = MockAgent(name="test", description="Test")
        state = AgentState(goal="Test", max_iterations=10)
        state.current_iteration = 2

        with patch("time.time", return_value=state.start_time + 10):
            should_continue, reason = agent.should_continue(state)

        assert should_continue
        assert reason is None

    @pytest.mark.asyncio
    async def test_run_concludes_immediately(self):
        """Test agent run that concludes immediately."""
        agent = MockAgent(name="test", description="Test")

        state = await agent.run(
            goal="Test goal",
            model_name="test-model",
            ollama_url="http://localhost:11434",
            max_iterations=10,
        )

        assert state.final_answer == "Mock final answer"
        assert state.termination_reason == "goal_satisfied"
        assert len(state.thinking_history) == 1


class TestAgentRegistry:
    """Test AgentRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        AgentRegistry._agents = {}

    def test_register_agent(self):
        """Test agent registration."""
        AgentRegistry.register("mock", MockAgent)

        assert AgentRegistry.is_registered("mock")
        assert "mock" in AgentRegistry._agents

    def test_get_agent(self):
        """Test retrieving agent by name."""
        AgentRegistry.register("mock", MockAgent)

        agent = AgentRegistry.get("mock")

        assert isinstance(agent, MockAgent)
        assert agent.name == "mock"

    def test_get_nonexistent_agent(self):
        """Test error on nonexistent agent."""
        with pytest.raises(KeyError, match="Agent 'nonexistent' not found"):
            AgentRegistry.get("nonexistent")

    def test_list_agents(self):
        """Test listing registered agents."""
        AgentRegistry.register("mock1", MockAgent)
        AgentRegistry.register("mock2", MockAgent)

        agents = AgentRegistry.list_agents()

        assert len(agents) == 2
        names = [name for name, _ in agents]
        assert "mock1" in names
        assert "mock2" in names

    def test_register_decorator(self):
        """Test @register_agent decorator."""

        @register_agent("decorated")
        class DecoratedAgent(MockAgent):
            pass

        assert AgentRegistry.is_registered("decorated")
        agent = AgentRegistry.get("decorated")
        assert isinstance(agent, DecoratedAgent)

    def test_register_non_agent_class(self):
        """Test error when registering non-BaseAgent class."""

        class NotAnAgent:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseAgent"):
            AgentRegistry.register("invalid", NotAnAgent)
