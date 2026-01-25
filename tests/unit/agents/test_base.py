"""Tests for base Agent interface."""

import pytest

from tensortruth.agents.base import Agent
from tensortruth.agents.config import AgentCallbacks, AgentResult


def test_agent_is_abstract():
    """Test that Agent cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Agent()  # type: ignore


def test_agent_requires_run_method():
    """Test that Agent subclass must implement run()."""

    class IncompleteAgent(Agent):
        def get_metadata(self):
            return {}

    with pytest.raises(TypeError):
        IncompleteAgent()  # type: ignore


def test_agent_requires_get_metadata_method():
    """Test that Agent subclass must implement get_metadata()."""

    class IncompleteAgent(Agent):
        async def run(self, query, callbacks, **kwargs):
            return AgentResult(final_answer="")

    with pytest.raises(TypeError):
        IncompleteAgent()  # type: ignore


def test_agent_complete_implementation():
    """Test that Agent with both methods can be instantiated."""

    class CompleteAgent(Agent):
        async def run(self, query, callbacks, **kwargs):
            return AgentResult(final_answer="test")

        def get_metadata(self):
            return {"name": "test"}

    agent = CompleteAgent()
    assert agent is not None
    metadata = agent.get_metadata()
    assert metadata["name"] == "test"


@pytest.mark.asyncio
async def test_agent_run_returns_agent_result():
    """Test that run() returns AgentResult."""

    class TestAgent(Agent):
        async def run(self, query, callbacks, **kwargs):
            return AgentResult(final_answer=f"Answer to: {query}")

        def get_metadata(self):
            return {}

    agent = TestAgent()
    result = await agent.run("test query", AgentCallbacks())
    assert isinstance(result, AgentResult)
    assert result.final_answer == "Answer to: test query"
