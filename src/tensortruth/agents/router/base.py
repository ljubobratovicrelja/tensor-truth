"""Base class for router-based agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from tensortruth.agents.base import Agent
from tensortruth.agents.config import AgentCallbacks, AgentResult


class RouterAgent(Agent, ABC):
    """Abstract base class for router-based agents.

    Subclasses implement their own routing logic and execution flow.
    Some agents (like BrowseAgent) iterate through multiple actions.
    Others (like ChatAgent) make a single routing decision.

    This is a pure abstract base class - subclasses define their own
    execution pattern and state management.
    """

    @abstractmethod
    async def run(
        self, query: str, callbacks: Optional[AgentCallbacks] = None, **kwargs: Any
    ) -> AgentResult:
        """Execute the agent's workflow.

        Subclasses define their own execution pattern:
        - BrowseAgent: iterative route → execute → synthesize loop
        - ChatAgent: single classification → delegate → forward

        Args:
            query: User's query/request
            callbacks: Optional callbacks for progress updates

        Returns:
            AgentResult with final answer and metadata
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return agent metadata.

        Returns:
            Dict with agent metadata including type and capabilities
        """
        pass
