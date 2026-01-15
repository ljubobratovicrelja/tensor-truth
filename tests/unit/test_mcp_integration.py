"""Integration tests for MCP-based agents."""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.agents.config import AgentResult, MCPServerConfig, MCPServerType
from tensortruth.agents.mcp_agent import MCPBrowseAgent, ToolTracker, browse_agent
from tensortruth.agents.server_registry import create_default_registry


class TestMCPIntegration:
    """Integration tests for MCP components."""

    def test_tool_tracker_initialization(self):
        """Test that tool tracker initializes correctly."""
        tracker = ToolTracker()
        assert tracker.tools_called == []
        assert tracker.urls_browsed == []
        assert tracker.search_queries == []

    def test_tool_tracker_with_progress_callback(self):
        """Test tool tracker with progress callback."""
        progress_messages = []

        def track_progress(msg: str):
            progress_messages.append(msg)

        tracker = ToolTracker(progress_callback=track_progress)
        tracker._report_progress("Test message")

        assert "Test message" in progress_messages

    def test_build_partial_response(self):
        """Test that partial response is built correctly when max iterations hit."""
        agent = MCPBrowseAgent(
            model_name="llama3.1:8b",
            ollama_url="http://localhost:11434",
        )
        # Set up tool tracker with some data
        agent._tool_tracker = ToolTracker()
        agent._tool_tracker.search_queries = ["AI news 2024", "machine learning trends"]
        agent._tool_tracker.urls_browsed = [
            "https://example.com/ai-news",
            "https://example.org/ml-trends",
        ]

        response = agent._build_partial_response("What are the latest AI trends?")

        assert "What are the latest AI trends?" in response
        assert "AI news 2024" in response
        assert "machine learning trends" in response
        assert "https://example.com/ai-news" in response
        assert "https://example.org/ml-trends" in response
        assert "iteration limit" in response

    @pytest.mark.asyncio
    async def test_mcp_browse_agent_initialization(self):
        """Test MCPBrowseAgent initialization."""
        agent = MCPBrowseAgent(
            model_name="llama3.1:8b",
            ollama_url="http://localhost:11434",
            max_iterations=5,
        )

        assert agent.model_name == "llama3.1:8b"
        assert agent.ollama_url == "http://localhost:11434"
        assert agent.max_iterations == 5
        assert agent._agent is None
        assert agent._tool_tracker is None

    @pytest.mark.asyncio
    async def test_mcp_browse_agent_with_custom_servers(self):
        """Test MCPBrowseAgent with custom MCP servers."""
        custom_server = MCPServerConfig(
            name="custom-server",
            type=MCPServerType.STDIO,
            command="python",
            args=["-m", "custom.module"],
            enabled=True,
        )

        agent = MCPBrowseAgent(
            model_name="llama3.1:8b",
            ollama_url="http://localhost:11434",
            mcp_servers=[custom_server],
        )

        # Verify custom server was registered
        servers = agent._registry.list_servers()
        assert "custom-server" in servers
        assert "tensor-truth-web" in servers  # Default server should also be there

    @patch("tensortruth.agents.mcp_agent.MCPBrowseAgent.run")
    def test_browse_agent_deprecation_warnings(self, mock_run):
        """Test that deprecation warnings are issued for unused parameters."""
        # Mock the run method to avoid actual MCP connections
        mock_run.return_value = AgentResult(
            final_answer="Test response", iterations=1, tools_called=[]
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call with deprecated parameters
            browse_agent(
                goal="Test",
                model_name="llama3.1:8b",
                ollama_url="http://localhost:11434",
                min_required_pages=10,  # Non-default value should trigger warning
                thinking_callback=lambda x: x,  # Should trigger warning
                max_iterations=1,
            )

            # Check that warnings were issued
            warning_messages = [str(warning.message) for warning in w]

            assert any("min_required_pages" in msg for msg in warning_messages)
            assert any("thinking_callback" in msg for msg in warning_messages)

    @pytest.mark.asyncio
    async def test_mcp_server_registry_connection_cleanup(self):
        """Test MCP server registry connection cleanup."""
        registry = create_default_registry()

        # Mock the clients
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        registry._clients = {
            "test-server": mock_client1,
            "another-server": mock_client2,
        }
        registry._tools = [MagicMock(), MagicMock()]

        # Call cleanup
        await registry.close_all_connections()

        # Verify clients and tools were cleared
        assert len(registry._clients) == 0
        assert len(registry._tools) == 0

    def test_agent_result_comprehensive(self):
        """Test AgentResult with all fields."""
        result = AgentResult(
            final_answer="Comprehensive test answer",
            iterations=3,
            tools_called=["search_web", "fetch_page", "search_web"],
            urls_browsed=["https://example.com", "https://docs.python.org"],
            error=None,
        )

        assert result.final_answer == "Comprehensive test answer"
        assert result.iterations == 3
        assert result.tools_called == ["search_web", "fetch_page", "search_web"]
        assert result.urls_browsed == ["https://example.com", "https://docs.python.org"]
        assert result.error is None

    def test_agent_result_with_error(self):
        """Test AgentResult with error."""
        result = AgentResult(final_answer="Error occurred", error="Connection timeout")

        assert result.final_answer == "Error occurred"
        assert result.error == "Connection timeout"
        assert result.iterations == 0
        assert result.tools_called == []


class TestMCPErrorHandling:
    """Tests for error handling in MCP components."""

    def test_mcp_server_config_validation(self):
        """Test MCP server config validation."""
        # Test STDIO without command
        with pytest.raises(ValueError, match="requires 'command'"):
            MCPServerConfig(name="invalid-stdio", type=MCPServerType.STDIO)

        # Test SSE without URL
        with pytest.raises(ValueError, match="requires 'url'"):
            MCPServerConfig(name="invalid-sse", type=MCPServerType.SSE)

    def test_agent_result_empty_goal(self):
        """Test AgentResult creation with empty goal."""
        result = AgentResult(
            final_answer="Error: Please provide a research goal.",
            error="Empty goal provided",
        )

        assert "Please provide a research goal" in result.final_answer
        assert result.error == "Empty goal provided"
