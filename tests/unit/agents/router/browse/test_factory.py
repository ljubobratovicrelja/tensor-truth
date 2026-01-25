"""Tests for BrowseAgent factory."""

import pytest
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.config import AgentConfig
from tensortruth.agents.factory import get_agent_factory_registry
from tensortruth.agents.router.browse.agent import BrowseAgent
from tensortruth.agents.router.browse.factory import create_browse_agent


def test_create_browse_agent_with_required_params():
    """Test creating BrowseAgent with minimal required parameters."""
    config = AgentConfig(
        name="test_browse",
        description="Test",
        tools=["search_web", "fetch_pages_batch"],
    )

    # Create mock tools
    def mock_search(**kwargs):
        return {"results": []}

    def mock_fetch(**kwargs):
        return {"pages": []}

    tools = [
        FunctionTool.from_defaults(fn=mock_search, name="search_web"),
        FunctionTool.from_defaults(fn=mock_fetch, name="fetch_pages_batch"),
    ]

    llm = Ollama(model="llama3.1:8b")
    session_params = {
        "router_model": "llama3.2:3b",
        "context_window": 16384,
        "ollama_url": "http://localhost:11434",
    }

    agent = create_browse_agent(config, tools, llm, session_params)

    assert isinstance(agent, BrowseAgent)
    assert agent.min_pages_required == 5  # default
    assert agent.max_iterations == 10  # from config


def test_create_browse_agent_validates_required_tools():
    """Test that factory validates required tools."""
    config = AgentConfig(
        name="test_browse",
        description="Test",
        tools=["search_web"],  # Missing fetch_pages_batch
    )

    tools = [
        FunctionTool.from_defaults(fn=lambda: None, name="search_web"),
    ]

    llm = Ollama(model="llama3.1:8b")
    session_params = {}

    with pytest.raises(ValueError, match="Missing"):
        create_browse_agent(config, tools, llm, session_params)


def test_create_browse_agent_uses_factory_params():
    """Test that factory uses factory_params from config."""
    config = AgentConfig(
        name="test_browse",
        description="Test",
        tools=["search_web", "fetch_pages_batch"],
        factory_params={"min_pages_required": 5},
    )

    def mock_search(**kwargs):
        return {}

    def mock_fetch(**kwargs):
        return {}

    tools = [
        FunctionTool.from_defaults(fn=mock_search, name="search_web"),
        FunctionTool.from_defaults(fn=mock_fetch, name="fetch_pages_batch"),
    ]

    llm = Ollama(model="llama3.1:8b")
    session_params = {"context_window": 8192}

    agent = create_browse_agent(config, tools, llm, session_params)

    assert agent.min_pages_required == 5  # from factory_params
    assert agent.max_content_chars > 0  # calculated from context_window


def test_browse_agent_factory_self_registers():
    """Test that BrowseAgent factory self-registers on import."""
    # Import should trigger self-registration
    from tensortruth.agents.router.browse import factory  # noqa: F401

    registry = get_agent_factory_registry()
    assert "router" in registry.list_types()


def test_browse_agent_get_metadata():
    """Test that BrowseAgent implements get_metadata()."""
    config = AgentConfig(
        name="test_browse",
        description="Test",
        tools=["search_web", "fetch_pages_batch"],
    )

    tools = [
        FunctionTool.from_defaults(fn=lambda: None, name="search_web"),
        FunctionTool.from_defaults(fn=lambda: None, name="fetch_pages_batch"),
    ]

    llm = Ollama(model="llama3.1:8b")
    agent = create_browse_agent(config, tools, llm, {})

    metadata = agent.get_metadata()
    assert metadata["name"] == "browse"
    assert metadata["agent_type"] == "router"
    assert "capabilities" in metadata
