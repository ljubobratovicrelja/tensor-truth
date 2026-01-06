"""Agent framework for autonomous task execution.

This module provides the base architecture for implementing autonomous agents
that can iteratively reason, take actions, and achieve user-specified goals.

Architecture Overview:
    - BaseAgent: Abstract class defining agent interface
    - AgentAction: Structured representation of agent actions
    - AgentState: Tracks agent execution state across iterations
    - AgentRegistry: Global registry for discovering available agents

Example Usage:
    # Register a new agent type
    @register_agent("browse")
    class BrowseAgent(BaseAgent):
        def reason_next_action(self, state):
            # Agent-specific reasoning logic
            pass

    # Execute an agent
    agent = AgentRegistry.get("browse")
    result = agent.execute(goal="Find Python docs", config={...})
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Core Data Structures
# =============================================================================


class AgentActionType(Enum):
    """Types of actions an agent can take."""

    SEARCH = "SEARCH"
    FETCH_PAGE = "FETCH_PAGE"
    CONCLUDE = "CONCLUDE"
    # Future action types can be added here:
    # CALCULATE = "CALCULATE"
    # EXECUTE_CODE = "EXECUTE_CODE"
    # ASK_USER = "ASK_USER"


@dataclass
class AgentAction:
    """Structured representation of an agent action.

    Attributes:
        type: Type of action to execute
        query: Search query (for SEARCH actions)
        url: URL to fetch (for FETCH_PAGE actions)
        title: Page title (for FETCH_PAGE actions)
        reasoning: Agent's reasoning for this action
    """

    type: AgentActionType
    query: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass
class AgentState:
    """Tracks agent execution state across iterations.

    This state object is passed to the agent at each iteration and accumulates
    observations from executed actions.

    Attributes:
        goal: User's original goal
        max_iterations: Maximum allowed iterations
        current_iteration: Current iteration count
        start_time: Timestamp when agent started
        searches_performed: List of (query, results) tuples
        pages_visited: List of (url, title, content_summary) tuples
        information_gathered: List of key insights extracted
        failed_fetches: List of (url, error_message) tuples
        termination_reason: Why agent stopped (goal_satisfied, max_iterations, etc.)
        final_answer: Agent's synthesized final answer
        thinking_history: List of thinking/reasoning from each iteration
    """

    goal: str
    max_iterations: int = 10
    current_iteration: int = 0
    start_time: float = field(default_factory=time.time)
    searches_performed: List[Tuple[str, List[Dict[str, str]]]] = field(
        default_factory=list
    )
    pages_visited: List[Tuple[str, str, str]] = field(default_factory=list)
    information_gathered: List[str] = field(default_factory=list)
    failed_fetches: List[Tuple[str, str]] = field(default_factory=list)
    termination_reason: Optional[str] = None
    final_answer: Optional[str] = None
    thinking_history: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Base Agent Class
# =============================================================================


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Subclasses must implement:
        - reason_next_action(): Decide what to do next given current state
        - execute_action(): Execute the chosen action and update state
        - synthesize_final_answer(): Create final response from gathered info

    Optional overrides:
        - get_max_time_seconds(): Override default timeout (300s)
        - get_default_config(): Override default configuration
        - should_continue(): Custom termination logic
    """

    def __init__(self, name: str, description: str):
        """Initialize agent.

        Args:
            name: Agent name (e.g., "browse", "research")
            description: Human-readable description of agent capabilities
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def reason_next_action(
        self,
        state: AgentState,
        model_name: str,
        ollama_url: str,
        context_window: int,
    ) -> Tuple[str, AgentAction]:
        """Determine the next action to take based on current state.

        Args:
            state: Current agent state
            model_name: Ollama model to use for reasoning
            ollama_url: Ollama API URL
            context_window: Model context window size

        Returns:
            Tuple of (thinking_text, next_action)
        """
        pass

    @abstractmethod
    async def execute_action(
        self,
        action: AgentAction,
        state: AgentState,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Execute an action and update state with observations.

        Args:
            action: Action to execute
            state: Agent state (modified in-place)
            progress_callback: Optional callback for progress updates
        """
        pass

    @abstractmethod
    async def synthesize_final_answer(
        self,
        state: AgentState,
        model_name: str,
        ollama_url: str,
        context_window: int,
        progress_callback: Optional[Callable[[str], None]] = None,
        synthesis_model: Optional[str] = None,
    ) -> str:
        """Synthesize final answer from accumulated state.

        Args:
            state: Final agent state with all observations
            model_name: Ollama model for synthesis (fallback if synthesis_model not provided)
            ollama_url: Ollama API URL
            context_window: Model context window size
            progress_callback: Optional callback for progress updates
            synthesis_model: Optional separate model for final synthesis

        Returns:
            Formatted final answer
        """
        pass

    def get_max_time_seconds(self) -> int:
        """Get maximum execution time in seconds.

        Override in subclasses for agent-specific timeouts.

        Returns:
            Timeout in seconds (default: 300 = 5 minutes)
        """
        return 300

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this agent.

        Override in subclasses to provide agent-specific defaults.

        Returns:
            Dictionary of default config values
        """
        return {
            "max_iterations": 10,
            "max_time_seconds": self.get_max_time_seconds(),
        }

    def should_continue(self, state: AgentState) -> Tuple[bool, Optional[str]]:
        """Determine if agent should continue iterating.

        Override for custom termination logic.

        Args:
            state: Current agent state

        Returns:
            Tuple of (should_continue, termination_reason)
        """
        # Check iteration budget
        if state.current_iteration >= state.max_iterations:
            return False, "max_iterations"

        # Check timeout
        elapsed = time.time() - state.start_time
        if elapsed > self.get_max_time_seconds():
            return False, "timeout"

        return True, None

    async def run(
        self,
        goal: str,
        model_name: str,
        ollama_url: str,
        max_iterations: int = 10,
        context_window: int = 16384,
        thinking_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        synthesis_model: Optional[str] = None,
    ) -> AgentState:
        """Execute the agent's main reasoning loop.

        This is the core execution method that orchestrates the agent's
        iterative reasoning and action cycle.

        Args:
            goal: User's goal to achieve
            model_name: Ollama model name for reasoning (fast model)
            ollama_url: Ollama API base URL
            max_iterations: Maximum iterations allowed
            context_window: Model context window size
            thinking_callback: Callback for streaming thinking updates
            progress_callback: Callback for progress updates
            synthesis_model: Optional separate model for final synthesis (quality model)

        Returns:
            Final agent state with answer and execution history
        """
        # Validate inputs
        if not goal or not goal.strip():
            raise ValueError("goal cannot be empty or whitespace-only")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        self.logger.info(f"Starting {self.name} agent for goal: {goal}")

        # Initialize state
        state = AgentState(goal=goal, max_iterations=max_iterations)

        try:
            # Main reasoning loop
            while True:
                # Check if should continue
                should_continue, reason = self.should_continue(state)
                if not should_continue:
                    state.termination_reason = reason
                    self.logger.info(f"Agent stopping: {reason}")
                    break

                # Reasoning phase: decide next action
                self.logger.debug(
                    f"Iteration {state.current_iteration + 1}/{max_iterations}"
                )
                thinking, next_action = await self.reason_next_action(
                    state, model_name, ollama_url, context_window
                )

                # Store thinking in history (bounded to prevent memory issues)
                MAX_THINKING_HISTORY = 20
                thinking_record = {
                    "iteration": state.current_iteration + 1,
                    "thinking": thinking,
                    "action": next_action.type.value,
                    "reasoning": next_action.reasoning or "",
                }
                state.thinking_history.append(thinking_record)

                # Keep only last N entries to prevent unbounded growth
                if len(state.thinking_history) > MAX_THINKING_HISTORY:
                    state.thinking_history = state.thinking_history[
                        -MAX_THINKING_HISTORY:
                    ]

                # Stream thinking to UI (if callback provided)
                if thinking_callback:
                    thinking_callback(
                        f"### Iteration {state.current_iteration + 1}\n\n"
                        f"**Thinking:**\n{thinking}\n\n"
                        f"**Action:** {next_action.type.value}\n\n"
                        f"**Reasoning:** {next_action.reasoning or 'N/A'}"
                    )

                # Show natural language progress update based on action type
                if progress_callback:
                    action_descriptions = {
                        AgentActionType.SEARCH: "searching for information",
                        AgentActionType.FETCH_PAGE: "retrieving a relevant page",
                    }
                    action_desc = action_descriptions.get(
                        next_action.type, "taking action"
                    )
                    progress_callback(f"🤔 **Agent is {action_desc}**")

                # Check if agent decided to conclude
                if next_action.type == AgentActionType.CONCLUDE:
                    state.termination_reason = "goal_satisfied"
                    self.logger.info("Agent concluded: goal satisfied")
                    break

                # Action phase: execute action
                await self.execute_action(next_action, state, progress_callback)

                state.current_iteration += 1

            # Final synthesis
            self.logger.info("Synthesizing final answer...")
            if progress_callback:
                progress_callback(
                    "🤖 **Synthesizing final answer from gathered sources**"
                )

            state.final_answer = await self.synthesize_final_answer(
                state,
                model_name,
                ollama_url,
                context_window,
                progress_callback,
                synthesis_model,
            )

        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}", exc_info=True)
            state.termination_reason = "error"
            state.final_answer = (
                f"❌ **Agent execution failed:** {str(e)}\n\n"
                f"Partial results may be available in thinking history."
            )

        return state


# =============================================================================
# Agent Registry
# =============================================================================


class AgentRegistry:
    """Global registry for agent discovery and instantiation.

    Usage:
        # Register an agent
        @register_agent("browse")
        class BrowseAgent(BaseAgent):
            pass

        # Get an agent instance
        agent = AgentRegistry.get("browse")

        # List available agents
        agents = AgentRegistry.list_agents()
    """

    _agents: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, agent_class: type) -> None:
        """Register an agent class.

        Args:
            name: Agent name (used for lookup)
            agent_class: Agent class (must inherit from BaseAgent)
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"{agent_class} must inherit from BaseAgent")

        cls._agents[name] = agent_class
        logger.info(f"Registered agent: {name}")

    @classmethod
    def get(cls, name: str) -> BaseAgent:
        """Get an agent instance by name.

        Args:
            name: Agent name

        Returns:
            Agent instance

        Raises:
            KeyError: If agent not found
        """
        if name not in cls._agents:
            available = ", ".join(cls._agents.keys())
            raise KeyError(f"Agent '{name}' not found. Available agents: {available}")

        agent_class = cls._agents[name]
        return agent_class(name=name, description=agent_class.__doc__ or "")

    @classmethod
    def list_agents(cls) -> List[Tuple[str, str]]:
        """List all registered agents.

        Returns:
            List of (name, description) tuples
        """
        return [
            (name, agent_class.__doc__ or "No description")
            for name, agent_class in cls._agents.items()
        ]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an agent is registered.

        Args:
            name: Agent name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._agents


# =============================================================================
# Decorator for Easy Registration
# =============================================================================


def register_agent(name: str):
    """Decorator to register an agent class.

    Usage:
        @register_agent("browse")
        class BrowseAgent(BaseAgent):
            pass
    """

    def decorator(agent_class: type) -> type:
        AgentRegistry.register(name, agent_class)
        return agent_class

    return decorator
