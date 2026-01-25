"""Base interface for all TensorTruth agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from tensortruth.agents.config import AgentCallbacks, AgentResult


class Agent(ABC):
    """Base interface for all TensorTruth agents.

    All agents must implement this interface to work with AgentService.
    Supports both router-based and function-based agents.
    """

    @abstractmethod
    async def run(
        self, query: str, callbacks: AgentCallbacks, **kwargs: Any
    ) -> AgentResult:
        """Execute the agent with a query.

        Args:
            query: User's query/goal
            callbacks: Callbacks for progress, tool calls, streaming
            **kwargs: Additional agent-specific parameters

        Returns:
            AgentResult with final answer, iterations, tools, URLs
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata for introspection.

        Returns:
            Dict with: name, description, agent_type, capabilities
        """
        pass
