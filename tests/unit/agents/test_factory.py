"""Tests for agent factory registry."""

import pytest
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.base import Agent
from tensortruth.agents.config import AgentCallbacks, AgentConfig, AgentResult
from tensortruth.agents.factory import (
    AgentFactoryRegistry,
    get_agent_factory_registry,
    register_agent_factory,
)


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, name: str):
        self.name = name

    async def run(self, query, callbacks, **kwargs):
        return AgentResult(final_answer=f"Mock answer from {self.name}")

    def get_metadata(self):
        return {"name": self.name, "type": "mock"}


def test_registry_register_and_list():
    """Test registering factories and listing types."""
    registry = AgentFactoryRegistry()

    def mock_factory(config, tools, llm, params):
        return MockAgent("test")

    registry.register("test_type", mock_factory)
    assert "test_type" in registry.list_types()


def test_registry_duplicate_registration():
    """Test that duplicate registration raises ValueError."""
    registry = AgentFactoryRegistry()

    def factory1(config, tools, llm, params):
        return MockAgent("1")

    def factory2(config, tools, llm, params):
        return MockAgent("2")

    registry.register("duplicate", factory1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("duplicate", factory2)


def test_registry_create_agent():
    """Test creating agent via factory."""
    registry = AgentFactoryRegistry()

    def mock_factory(config, tools, llm, params):
        return MockAgent(config.name)

    registry.register("mock", mock_factory)

    config = AgentConfig(
        name="test_agent",
        description="Test",
        tools=[],
    )

    agent = registry.create("mock", config, [], None, {})  # type: ignore
    assert isinstance(agent, MockAgent)
    assert agent.name == "test_agent"


def test_registry_create_unknown_type():
    """Test creating unknown agent type raises ValueError."""
    registry = AgentFactoryRegistry()

    config = AgentConfig(
        name="test",
        description="Test",
        tools=[],
    )

    with pytest.raises(ValueError, match="Unknown agent type"):
        registry.create("nonexistent", config, [], None, {})  # type: ignore


def test_registry_factory_receives_parameters():
    """Test that factory receives all parameters correctly."""
    registry = AgentFactoryRegistry()

    received_params = {}

    def capturing_factory(config, tools, llm, params):
        received_params["config"] = config
        received_params["tools"] = tools
        received_params["llm"] = llm
        received_params["params"] = params
        return MockAgent("test")

    registry.register("capturing", capturing_factory)

    config = AgentConfig(name="test", description="Test", tools=["tool1"])
    tools = [FunctionTool.from_defaults(fn=lambda x: x, name="tool1")]
    llm = Ollama(model="test")
    params = {"key": "value"}

    _ = registry.create("capturing", config, tools, llm, params)

    assert received_params["config"] == config
    assert received_params["tools"] == tools
    assert received_params["llm"] == llm
    assert received_params["params"] == params


def test_global_registry_singleton():
    """Test that global registry is a singleton."""
    registry1 = get_agent_factory_registry()
    registry2 = get_agent_factory_registry()
    assert registry1 is registry2


def test_register_agent_factory_global():
    """Test global registration function."""

    def test_factory(config, tools, llm, params):
        return MockAgent("global")

    # Register via global function
    register_agent_factory("global_test", test_factory)

    # Verify it's in the global registry
    registry = get_agent_factory_registry()
    assert "global_test" in registry.list_types()


@pytest.mark.asyncio
async def test_created_agent_is_functional():
    """Test that agent created via factory is functional."""
    registry = AgentFactoryRegistry()

    def mock_factory(config, tools, llm, params):
        return MockAgent(config.name)

    registry.register("functional", mock_factory)

    config = AgentConfig(name="functional_agent", description="Test", tools=[])
    agent = registry.create("functional", config, [], None, {})  # type: ignore

    result = await agent.run("test query", AgentCallbacks())
    assert isinstance(result, AgentResult)
    assert "functional_agent" in result.final_answer
