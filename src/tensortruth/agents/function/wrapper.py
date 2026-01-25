"""Wrapper for LlamaIndex FunctionAgent to implement Agent interface."""

from typing import Any, Dict

from llama_index.core.agent.workflow.function_agent import (
    FunctionAgent as LIFunctionAgent,
)

from tensortruth.agents.base import Agent
from tensortruth.agents.config import AgentCallbacks, AgentResult


class FunctionAgentWrapper(Agent):
    """Wraps LlamaIndex FunctionAgent to implement Agent interface.

    FunctionAgent uses native LLM tool-calling capabilities (single-shot).
    Works well with small models that have good JSON output support.
    """

    def __init__(self, function_agent: LIFunctionAgent, agent_name: str = "function"):
        """Initialize wrapper.

        Args:
            function_agent: LlamaIndex FunctionAgent instance
            agent_name: Name for metadata
        """
        self._agent = function_agent
        self._agent_name = agent_name

    async def run(
        self, query: str, callbacks: AgentCallbacks, **kwargs: Any
    ) -> AgentResult:
        """Execute FunctionAgent and return AgentResult.

        Args:
            query: User's query
            callbacks: Callbacks (note: FunctionAgent doesn't support streaming)
            **kwargs: Additional parameters

        Returns:
            AgentResult with final answer
        """
        # Progress callback
        if callbacks.on_progress:
            callbacks.on_progress(f"Starting {self._agent_name} agent...")

        # Execute FunctionAgent (uses LlamaIndex's run method)
        response = await self._agent.run(user_msg=query)

        # Convert to AgentResult
        return AgentResult(
            final_answer=str(response),
            iterations=0,  # FunctionAgent doesn't expose iteration count
            tools_called=[],  # Would need tracking
            urls_browsed=[],  # Would need tracking
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata.

        Returns:
            Dict with agent metadata
        """
        return {
            "name": self._agent_name,
            "description": "LlamaIndex FunctionAgent (native tool-calling)",
            "agent_type": "function",
            "capabilities": ["native_tool_calling", "single_shot_decisions"],
        }
