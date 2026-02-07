"""Wrapper for LlamaIndex FunctionAgent to implement Agent interface."""

from typing import Any, Dict

from llama_index.core.agent.workflow import AgentStream, ToolCall
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

        Streams events from the underlying workflow to fire callbacks
        for tool calls and token generation as they happen.

        Args:
            query: User's query
            callbacks: Callbacks for streaming progress
            **kwargs: Additional parameters

        Returns:
            AgentResult with final answer
        """
        if callbacks.on_progress:
            callbacks.on_progress(f"Starting {self._agent_name} agent...")

        tools_called: list[str] = []
        full_response = ""

        # Get handler (starts workflow) — don't await directly
        handler = self._agent.run(user_msg=query)

        # Intercept events as they're emitted
        async for event in handler.stream_events():
            if isinstance(event, ToolCall):
                tools_called.append(event.tool_name)
                if callbacks.on_tool_call:
                    callbacks.on_tool_call(event.tool_name, event.tool_kwargs)
                if callbacks.on_progress:
                    callbacks.on_progress(f"Calling {event.tool_name}...")
            elif isinstance(event, AgentStream):
                if event.delta and callbacks.on_token:
                    full_response += event.delta
                    callbacks.on_token(event.delta)

        # Get final result
        response = await handler
        final = full_response or str(response)

        return AgentResult(
            final_answer=final,
            tools_called=tools_called,
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
