"""Unit tests for MCP browse agent."""

import pytest

from tensortruth.agents.config import AgentResult, MCPServerConfig, MCPServerType
from tensortruth.agents.server_registry import (
    MCPServerRegistry,
    create_default_registry,
)


class TestMCPServerConfig:
    """Tests for MCPServerConfig."""

    def test_stdio_config_valid(self):
        """Test valid stdio configuration."""
        config = MCPServerConfig(
            name="test",
            type=MCPServerType.STDIO,
            command="python",
            args=["-m", "test"],
        )
        assert config.name == "test"
        assert config.type == MCPServerType.STDIO

    def test_stdio_config_missing_command(self):
        """Test that stdio requires command."""
        with pytest.raises(ValueError, match="requires 'command'"):
            MCPServerConfig(
                name="test",
                type=MCPServerType.STDIO,
            )

    def test_sse_config_missing_url(self):
        """Test that sse requires url."""
        with pytest.raises(ValueError, match="requires 'url'"):
            MCPServerConfig(
                name="test",
                type=MCPServerType.SSE,
            )


class TestMCPServerRegistry:
    """Tests for MCPServerRegistry."""

    def test_register_server(self):
        """Test server registration."""
        registry = MCPServerRegistry()
        config = MCPServerConfig(
            name="test",
            type=MCPServerType.STDIO,
            command="python",
        )
        registry.register(config)
        assert "test" in registry.list_servers()

    def test_get_enabled_servers(self):
        """Test filtering enabled servers."""
        registry = MCPServerRegistry()
        registry.register(
            MCPServerConfig(name="enabled", type=MCPServerType.STDIO, command="python")
        )
        registry.register(
            MCPServerConfig(
                name="disabled",
                type=MCPServerType.STDIO,
                command="python",
                enabled=False,
            )
        )

        enabled = registry.get_enabled_servers()
        assert len(enabled) == 1
        assert enabled[0].name == "enabled"

    def test_create_default_registry(self):
        """Test default registry has web tools server."""
        registry = create_default_registry()
        servers = registry.list_servers()
        assert "tensor-truth-web" in servers


class TestAgentResult:
    """Tests for AgentResult."""

    def test_agent_result_creation(self):
        """Test AgentResult creation."""
        result = AgentResult(
            final_answer="Test answer",
            iterations=5,
            tools_called=["search_web", "fetch_page"],
        )
        assert result.final_answer == "Test answer"
        assert result.iterations == 5
        assert len(result.tools_called) == 2
        assert result.error is None

    def test_agent_result_with_error(self):
        """Test AgentResult with error."""
        result = AgentResult(
            final_answer="Error occurred",
            error="Connection failed",
        )
        assert result.error == "Connection failed"

    def test_agent_result_with_urls_browsed(self):
        """Test AgentResult with urls_browsed field."""
        result = AgentResult(
            final_answer="Research complete",
            iterations=3,
            tools_called=["search_web", "fetch_page", "fetch_page"],
            urls_browsed=["https://example.com", "https://docs.python.org"],
        )
        assert len(result.urls_browsed) == 2
        assert "https://example.com" in result.urls_browsed
        assert "https://docs.python.org" in result.urls_browsed
