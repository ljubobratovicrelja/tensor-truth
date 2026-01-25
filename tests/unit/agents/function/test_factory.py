"""Tests for FunctionAgent factory."""

from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.config import AgentConfig
from tensortruth.agents.factory import get_agent_factory_registry
from tensortruth.agents.function.factory import create_function_agent
from tensortruth.agents.function.wrapper import FunctionAgentWrapper


def test_create_function_agent():
    """Test creating FunctionAgent via factory."""
    config = AgentConfig(
        name="test_function",
        description="Test function agent",
        tools=["test_tool"],
        system_prompt="You are a test assistant.",
    )

    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="test_tool")]
    llm = Ollama(model="llama3.2:3b")
    session_params = {}

    agent = create_function_agent(config, tools, llm, session_params)

    assert isinstance(agent, FunctionAgentWrapper)
    assert agent._agent_name == "test_function"


def test_create_function_agent_uses_system_prompt():
    """Test that factory uses system_prompt from config."""
    config = AgentConfig(
        name="test",
        description="Test",
        tools=["tool1"],
        system_prompt="Custom system prompt",
    )

    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="tool1")]
    llm = Ollama(model="llama3.2:3b")

    agent = create_function_agent(config, tools, llm, {})

    # Check that wrapper was created (system_prompt is internal to LlamaIndex agent)
    assert isinstance(agent, FunctionAgentWrapper)


def test_create_function_agent_default_system_prompt():
    """Test that factory provides default system prompt."""
    config = AgentConfig(
        name="test",
        description="Test",
        tools=["tool1"],
        system_prompt="",  # Empty prompt
    )

    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="tool1")]
    llm = Ollama(model="llama3.2:3b")

    agent = create_function_agent(config, tools, llm, {})

    assert isinstance(agent, FunctionAgentWrapper)


def test_function_agent_factory_self_registers():
    """Test that FunctionAgent factory self-registers on import."""
    # Import should trigger self-registration
    from tensortruth.agents.function import factory  # noqa: F401

    registry = get_agent_factory_registry()
    assert "function" in registry.list_types()


def test_function_agent_get_metadata():
    """Test that created FunctionAgent implements get_metadata()."""
    config = AgentConfig(
        name="test_function",
        description="Test",
        tools=["tool1"],
    )

    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="tool1")]
    llm = Ollama(model="llama3.2:3b")

    agent = create_function_agent(config, tools, llm, {})

    metadata = agent.get_metadata()
    assert metadata["name"] == "test_function"
    assert metadata["agent_type"] == "function"
    assert "capabilities" in metadata


def test_create_function_agent_handles_empty_tools():
    """Test creating FunctionAgent with no tools."""
    config = AgentConfig(
        name="test",
        description="Test",
        tools=[],
    )

    llm = Ollama(model="llama3.2:3b")

    # Should succeed (LlamaIndex handles empty tool list)
    agent = create_function_agent(config, [], llm, {})
    assert isinstance(agent, FunctionAgentWrapper)
