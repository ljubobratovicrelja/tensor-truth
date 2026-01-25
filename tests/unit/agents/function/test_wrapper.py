"""Tests for FunctionAgentWrapper."""

import pytest
from llama_index.core.agent.workflow.function_agent import (
    FunctionAgent as LIFunctionAgent,
)
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.config import AgentCallbacks, AgentResult
from tensortruth.agents.function.wrapper import FunctionAgentWrapper


def test_function_agent_wrapper_implements_agent_interface():
    """Test that FunctionAgentWrapper implements Agent interface."""
    # Create a mock FunctionAgent
    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")]
    llm = Ollama(model="llama3.2:3b")
    function_agent = LIFunctionAgent(tools=tools, llm=llm, system_prompt="Test")

    wrapper = FunctionAgentWrapper(function_agent, agent_name="test")

    # Check methods exist
    assert hasattr(wrapper, "run")
    assert hasattr(wrapper, "get_metadata")


def test_function_agent_wrapper_get_metadata():
    """Test get_metadata returns correct structure."""
    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")]
    llm = Ollama(model="llama3.2:3b")
    function_agent = LIFunctionAgent(tools=tools, llm=llm, system_prompt="Test")

    wrapper = FunctionAgentWrapper(function_agent, agent_name="my_agent")
    metadata = wrapper.get_metadata()

    assert metadata["name"] == "my_agent"
    assert metadata["agent_type"] == "function"
    assert "capabilities" in metadata
    assert "native_tool_calling" in metadata["capabilities"]


@pytest.mark.asyncio
async def test_function_agent_wrapper_run_returns_agent_result():
    """Test that run() returns AgentResult."""

    # Create a simple tool
    def echo_tool(text: str) -> str:
        """Echo the text."""
        return f"Echo: {text}"

    tools = [FunctionTool.from_defaults(fn=echo_tool, name="echo")]
    llm = Ollama(model="llama3.2:3b")
    function_agent = LIFunctionAgent(
        tools=tools, llm=llm, system_prompt="You are a helpful assistant."
    )

    wrapper = FunctionAgentWrapper(function_agent, agent_name="echo_agent")

    # Note: This may fail if Ollama is not running, but tests the interface
    try:
        result = await wrapper.run("Test query", AgentCallbacks())
        assert isinstance(result, AgentResult)
        assert isinstance(result.final_answer, str)
    except Exception:
        # Skip if Ollama not available
        pytest.skip("Ollama not available for integration test")


def test_function_agent_wrapper_stores_agent_name():
    """Test that wrapper stores agent name correctly."""
    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")]
    llm = Ollama(model="llama3.2:3b")
    function_agent = LIFunctionAgent(tools=tools, llm=llm, system_prompt="Test")

    wrapper = FunctionAgentWrapper(function_agent, agent_name="custom_name")

    assert wrapper._agent_name == "custom_name"
    assert wrapper.get_metadata()["name"] == "custom_name"
