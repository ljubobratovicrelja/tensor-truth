"""Tests for RouterAgent base class."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from tensortruth.agents.config import AgentCallbacks, AgentResult
from tensortruth.agents.router.base import RouterAgent
from tensortruth.agents.router.state import RouterState


# Define test fixtures here instead of importing from test file
class TestPhase(Enum):
    """Test workflow phases."""

    INITIAL = "initial"
    WORKING = "working"
    COMPLETE = "complete"


@dataclass
class ConcreteRouterState(RouterState):
    """Concrete implementation for testing."""

    test_field: str = "default"

    def is_complete(self) -> bool:
        """Test implementation."""
        return self.phase == TestPhase.COMPLETE

    def to_dict(self) -> dict:
        """Test implementation."""
        return {
            "query": self.query,
            "phase": self.phase.value if hasattr(self.phase, "value") else self.phase,
            "actions_taken": self.actions_taken,
            "iteration_count": self.iteration_count,
            "test_field": self.test_field,
        }


class ConcreteRouterAgent(RouterAgent):
    """Concrete implementation for testing.

    Implements its own run loop (similar to BrowseAgent pattern).
    """

    def __init__(
        self,
        router_llm=None,
        synthesis_llm=None,
        tools=None,
        max_iterations: int = 10,
    ):
        """Initialize with optional components for testing."""
        self.router_llm = router_llm
        self.synthesis_llm = synthesis_llm
        self.tools = tools or {}
        self.max_iterations = max_iterations
        self.route_calls = []
        self.execute_calls = []
        self.synthesize_calls = []

    async def run(
        self, query: str, callbacks: Optional[AgentCallbacks] = None, **kwargs: Any
    ) -> AgentResult:
        """Implement run with iterative workflow for testing."""
        if callbacks is None:
            callbacks = AgentCallbacks()

        state = self._create_initial_state(query)

        # Router + executor loop
        while not state.is_complete() and state.iteration_count < self.max_iterations:
            action = await self.route(state)
            state = await self.execute(action, state, callbacks)
            state.iteration_count += 1

        # Synthesize final answer
        final_answer = await self._synthesize(state, callbacks)

        return self._build_result(state, final_answer)

    def get_metadata(self) -> Dict[str, Any]:
        """Return test metadata."""
        return {
            "name": "test_router",
            "agent_type": "test",
        }

    def _create_initial_state(self, query: str) -> RouterState:
        """Create initial test state."""
        return ConcreteRouterState(query=query, phase=TestPhase.INITIAL)

    async def route(self, state: RouterState) -> str:
        """Mock routing."""
        self.route_calls.append(state)
        if state.iteration_count == 0:
            return "action1"
        elif state.iteration_count == 1:
            return "action2"
        else:
            return "done"

    async def execute(
        self,
        action: str,
        state: RouterState,
        callbacks: Optional[AgentCallbacks] = None,
    ) -> RouterState:
        """Mock execution."""
        self.execute_calls.append((action, state))

        # Simulate tool call for callback testing
        if callbacks and callbacks.on_tool_call:
            callbacks.on_tool_call(action, {"state": state.query})

        state.actions_taken.append(action)

        if action == "done":
            state.phase = TestPhase.COMPLETE

        return state

    async def _synthesize(self, state: RouterState, callbacks) -> str:
        """Mock synthesis."""
        self.synthesize_calls.append(state)
        return f"Synthesized answer for: {state.query}"

    def _extract_urls(self, state: RouterState) -> List[str]:
        """Mock URL extraction."""
        return []

    def _build_result(self, state: RouterState, final_answer: str) -> AgentResult:
        """Build result from state."""
        return AgentResult(
            final_answer=final_answer,
            iterations=state.iteration_count,
            tools_called=state.actions_taken,
            urls_browsed=self._extract_urls(state),
        )


class TestRouterAgentAbstractMethods:
    """Test that RouterAgent requires abstract method implementation."""

    def test_router_agent_abstract_methods(self):
        """Should not allow instantiation without implementing abstract methods."""
        # This test verifies that RouterAgent is abstract
        with pytest.raises(TypeError):
            # Should fail because RouterAgent is abstract
            RouterAgent()  # type: ignore


class TestRouterAgentInitialization:
    """Test RouterAgent initialization."""

    def test_router_agent_initialization(self):
        """Should initialize with required components."""
        mock_router_llm = MagicMock()
        mock_synthesis_llm = MagicMock()
        mock_tools = {"tool1": MagicMock()}

        agent = ConcreteRouterAgent(
            mock_router_llm, mock_synthesis_llm, mock_tools, max_iterations=15
        )

        assert agent.router_llm == mock_router_llm
        assert agent.synthesis_llm == mock_synthesis_llm
        assert agent.tools == mock_tools
        assert agent.max_iterations == 15


@pytest.mark.asyncio
class TestRouterAgentRunWorkflow:
    """Test RouterAgent.run() workflow."""

    async def test_router_agent_run_workflow(self):
        """Should execute complete router workflow."""
        mock_router_llm = MagicMock()
        mock_synthesis_llm = MagicMock()

        agent = ConcreteRouterAgent(
            mock_router_llm, mock_synthesis_llm, {}, max_iterations=10
        )

        callbacks = AgentCallbacks()
        result = await agent.run("test query", callbacks)

        # Should have executed router + executor loop
        assert len(agent.route_calls) > 0
        assert len(agent.execute_calls) > 0

        # Should have synthesized
        assert len(agent.synthesize_calls) == 1

        # Should return AgentResult
        assert isinstance(result, AgentResult)
        assert result.final_answer == "Synthesized answer for: test query"
        assert result.iterations > 0

    async def test_router_agent_two_phase_execution(self):
        """Should route then execute in each iteration."""
        mock_router_llm = MagicMock()
        mock_synthesis_llm = MagicMock()

        agent = ConcreteRouterAgent(
            mock_router_llm, mock_synthesis_llm, {}, max_iterations=10
        )

        callbacks = AgentCallbacks()
        result = await agent.run("test query", callbacks)

        # Should have routed and executed multiple times
        assert len(agent.route_calls) >= 2
        assert len(agent.execute_calls) >= 2

        # Actions should be recorded
        assert "action1" in result.tools_called
        assert "action2" in result.tools_called

    async def test_router_agent_stops_when_complete(self):
        """Should stop when state.is_complete() returns True."""
        mock_router_llm = MagicMock()
        mock_synthesis_llm = MagicMock()

        agent = ConcreteRouterAgent(
            mock_router_llm, mock_synthesis_llm, {}, max_iterations=10
        )

        callbacks = AgentCallbacks()
        result = await agent.run("test query", callbacks)

        # Should stop before max_iterations (when done is executed)
        assert result.iterations < 10

    async def test_router_agent_respects_max_iterations(self):
        """Should stop at max_iterations even if not complete."""

        class NeverCompleteAgent(ConcreteRouterAgent):
            async def route(self, state):
                return "continue"

            async def execute(self, action, state, callbacks=None):
                state.actions_taken.append(action)
                # Never mark as complete
                return state

        mock_router_llm = MagicMock()
        mock_synthesis_llm = MagicMock()

        agent = NeverCompleteAgent(
            mock_router_llm, mock_synthesis_llm, {}, max_iterations=5
        )

        callbacks = AgentCallbacks()
        result = await agent.run("test query", callbacks)

        # Should stop at max_iterations
        assert result.iterations == 5

    async def test_router_agent_callbacks(self):
        """Should invoke callbacks during execution."""
        mock_router_llm = MagicMock()
        mock_synthesis_llm = MagicMock()

        agent = ConcreteRouterAgent(
            mock_router_llm, mock_synthesis_llm, {}, max_iterations=10
        )

        tool_call_records = []
        token_records = []

        def on_tool_call(name, args):
            tool_call_records.append((name, args))

        def on_token(token):
            token_records.append(token)

        callbacks = AgentCallbacks(on_tool_call=on_tool_call, on_token=on_token)
        await agent.run("test query", callbacks)

        # Should have called on_tool_call for each action
        assert len(tool_call_records) > 0

        # Note: on_token would only be called if synthesis actually streams tokens


class TestRouterAgentStateCreation:
    """Test state creation."""

    def test_create_initial_state(self):
        """Should create initial state for query."""
        mock_router_llm = MagicMock()
        mock_synthesis_llm = MagicMock()

        agent = ConcreteRouterAgent(mock_router_llm, mock_synthesis_llm, {})

        state = agent._create_initial_state("test query")

        assert isinstance(state, RouterState)
        assert state.query == "test query"
        assert state.iteration_count == 0
        assert state.actions_taken == []


class TestRouterAgentMetadata:
    """Test metadata methods."""

    def test_get_metadata(self):
        """Should return agent metadata."""
        agent = ConcreteRouterAgent()

        metadata = agent.get_metadata()

        assert "name" in metadata
        assert "agent_type" in metadata
        assert metadata["agent_type"] == "test"
