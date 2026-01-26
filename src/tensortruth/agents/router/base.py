"""Base class for router-based agents."""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.base import Agent
from tensortruth.agents.config import AgentCallbacks, AgentResult
from tensortruth.agents.router.state import RouterState

logger = logging.getLogger(__name__)


class RouterAgent(Agent):
    """Base class for router-based agents.

    Router agents use a two-phase architecture:
    1. Route: Small LLM decides next action based on state
    2. Execute: Action is executed, state is updated
    3. Repeat until complete
    4. Synthesize: Larger LLM generates final answer

    Subclasses implement domain-specific routing and execution logic.
    """

    def __init__(
        self,
        router_llm: Ollama,
        synthesis_llm: Ollama,
        tools: Dict[str, FunctionTool],
        max_iterations: int = 10,
    ):
        """Initialize router agent.

        Args:
            router_llm: Small, fast model for routing decisions
            synthesis_llm: Larger model for final synthesis
            tools: Dict of tool name -> FunctionTool
            max_iterations: Maximum iterations before stopping
        """
        self.router_llm = router_llm
        self.synthesis_llm = synthesis_llm
        self.tools = tools
        self.max_iterations = max_iterations

    @abstractmethod
    async def route(self, state: RouterState) -> str:
        """Route to next action based on current state.

        Args:
            state: Current workflow state

        Returns:
            Action name to execute (e.g., "search_web", "fetch_pages", "done")
        """
        pass

    @abstractmethod
    async def execute(
        self,
        action: str,
        state: RouterState,
        callbacks: Optional[AgentCallbacks] = None,
    ) -> RouterState:
        """Execute action and return updated state.

        Args:
            action: Action to execute
            state: Current state
            callbacks: Optional callbacks for tool calls

        Returns:
            Updated state after action execution
        """
        pass

    @abstractmethod
    def _create_initial_state(self, query: str) -> RouterState:
        """Create initial state for query.

        Args:
            query: User's query/request

        Returns:
            Initial RouterState subclass instance
        """
        pass

    @abstractmethod
    async def _synthesize(self, state: RouterState, callbacks: AgentCallbacks) -> str:
        """Synthesize final answer from workflow state.

        Args:
            state: Final workflow state
            callbacks: Callbacks for streaming tokens

        Returns:
            Final synthesized answer
        """
        pass

    @abstractmethod
    def _extract_urls(self, state: RouterState) -> List[str]:
        """Extract URLs browsed during workflow.

        Args:
            state: Final workflow state

        Returns:
            List of URLs
        """
        pass

    async def run(
        self, query: str, callbacks: AgentCallbacks, **kwargs: Any
    ) -> AgentResult:
        """Run complete agent workflow.

        This is the main entry point for all router agents. It:
        1. Creates initial state
        2. Loops: route → execute until complete or max_iterations
        3. Synthesizes final answer
        4. Returns AgentResult

        Args:
            query: User's query/request
            callbacks: Callbacks for progress updates

        Returns:
            AgentResult with final answer and metadata
        """
        # Initialize state
        state = self._create_initial_state(query)

        logger.info(
            f"Starting router agent workflow: query='{query}', "
            f"max_iterations={self.max_iterations}"
        )

        # Router + executor loop
        while not state.is_complete() and state.iteration_count < self.max_iterations:
            # Route: Decide next action
            action = await self.route(state)
            logger.debug(
                f"Iteration {state.iteration_count}: routed to action='{action}'"
            )

            # Callback for tool call (now handled in execute method with actual params)
            # if callbacks.on_tool_call:
            #     callbacks.on_tool_call(action, {})

            # Execute: Perform action and update state (callbacks passed through)
            state = await self.execute(action, state, callbacks)
            logger.debug(
                f"Iteration {state.iteration_count}: executed action='{action}', "
                f"state={state.to_dict()}"
            )

        logger.info(
            f"Router workflow complete: iterations={state.iteration_count}, "
            f"is_complete={state.is_complete()}"
        )

        # Synthesize final answer
        final_answer = await self._synthesize(state, callbacks)

        # Build result (subclasses can override to add more fields)
        return self._build_result(state, final_answer)

    def _build_result(self, state: RouterState, final_answer: str) -> AgentResult:
        """Build AgentResult from final state and answer.

        Subclasses can override to add additional fields like sources.

        Args:
            state: Final workflow state
            final_answer: Synthesized answer

        Returns:
            AgentResult
        """
        return AgentResult(
            final_answer=final_answer,
            iterations=state.iteration_count,
            tools_called=state.actions_taken,
            urls_browsed=self._extract_urls(state),
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata.

        Returns:
            Dict with agent metadata
        """
        return {
            "name": "router",
            "description": "Router-based agent",
            "agent_type": "router",
        }
