"""Tests for RouterState base class."""

from dataclasses import dataclass
from enum import Enum

from tensortruth.agents.router.state import RouterState


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


class TestRouterStateInitialization:
    """Test RouterState initialization."""

    def test_router_state_initialization(self):
        """Should initialize with required fields."""
        state = ConcreteRouterState(
            query="test query",
            phase=TestPhase.INITIAL,
        )

        assert state.query == "test query"
        assert state.phase == TestPhase.INITIAL
        assert state.actions_taken == []
        assert state.iteration_count == 0
        assert state.max_iterations == 10

    def test_router_state_with_custom_max_iterations(self):
        """Should allow custom max_iterations."""
        state = ConcreteRouterState(
            query="test query", phase=TestPhase.INITIAL, max_iterations=20
        )

        assert state.max_iterations == 20


class TestPhaseTracking:
    """Test phase tracking."""

    def test_phase_tracking(self):
        """Should track workflow phases."""
        state = ConcreteRouterState(query="test", phase=TestPhase.INITIAL)
        assert state.phase == TestPhase.INITIAL

        state.phase = TestPhase.WORKING
        assert state.phase == TestPhase.WORKING

        state.phase = TestPhase.COMPLETE
        assert state.phase == TestPhase.COMPLETE


class TestActionsTaken:
    """Test actions_taken tracking."""

    def test_actions_taken_tracking(self):
        """Should track actions taken."""
        state = ConcreteRouterState(query="test", phase=TestPhase.INITIAL)
        assert state.actions_taken == []

        state.actions_taken.append("search")
        assert state.actions_taken == ["search"]

        state.actions_taken.append("fetch")
        assert state.actions_taken == ["search", "fetch"]

    def test_actions_taken_initialized_empty(self):
        """Should initialize actions_taken as empty list."""
        state = ConcreteRouterState(query="test", phase=TestPhase.INITIAL)
        assert isinstance(state.actions_taken, list)
        assert len(state.actions_taken) == 0


class TestSubclassExtension:
    """Test that subclasses can add fields."""

    def test_subclass_can_add_fields(self):
        """Should allow subclasses to add custom fields."""
        state = ConcreteRouterState(
            query="test", phase=TestPhase.INITIAL, test_field="custom"
        )

        assert state.test_field == "custom"
        assert state.query == "test"

    def test_subclass_implements_abstract_methods(self):
        """Should require subclasses to implement abstract methods."""
        state = ConcreteRouterState(query="test", phase=TestPhase.COMPLETE)

        # is_complete is implemented
        assert state.is_complete() is True

        # to_dict is implemented
        result = state.to_dict()
        assert isinstance(result, dict)
        assert "query" in result


class TestIsComplete:
    """Test is_complete method."""

    def test_is_complete_logic(self):
        """Should check if workflow is complete."""
        state = ConcreteRouterState(query="test", phase=TestPhase.INITIAL)
        assert state.is_complete() is False

        state.phase = TestPhase.WORKING
        assert state.is_complete() is False

        state.phase = TestPhase.COMPLETE
        assert state.is_complete() is True


class TestToDict:
    """Test to_dict serialization."""

    def test_to_dict_serialization(self):
        """Should serialize state to dict."""
        state = ConcreteRouterState(
            query="test query",
            phase=TestPhase.WORKING,
            test_field="custom value",
        )
        state.actions_taken = ["action1", "action2"]
        state.iteration_count = 3

        result = state.to_dict()

        assert result["query"] == "test query"
        assert result["phase"] == "working"
        assert result["actions_taken"] == ["action1", "action2"]
        assert result["iteration_count"] == 3
        assert result["test_field"] == "custom value"
