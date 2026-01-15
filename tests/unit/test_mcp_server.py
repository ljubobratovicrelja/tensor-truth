"""Unit tests for MCP web tools server."""

import pytest

from tensortruth.mcp_servers.web_tools_server import create_server


class TestWebToolsServer:
    """Tests for the web tools MCP server."""

    def test_create_server(self):
        """Test server creation."""
        server = create_server()
        assert server is not None
        assert server.name == "tensor-truth-web-tools"

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test that server has list_tools method."""
        server = create_server()

        # Verify the server has the expected methods
        assert hasattr(server, "list_tools")
        assert hasattr(server, "call_tool")
        assert callable(getattr(server, "list_tools"))
        assert callable(getattr(server, "call_tool"))
