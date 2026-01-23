"""Tests for ToolService.

ToolService manages tool sources (MCP servers) and provides FunctionTools to agents.
Uses LlamaIndex ToolSpec pattern - does NOT define custom tool interfaces.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensortruth.services.tool_service import ToolService


class TestToolServiceInit:
    """Test ToolService initialization."""

    def test_init_with_default_registry(self):
        """Should create with default MCP registry when none provided."""
        with patch(
            "tensortruth.services.tool_service.create_default_registry"
        ) as mock_create:
            mock_registry = MagicMock()
            mock_create.return_value = mock_registry

            service = ToolService()

            mock_create.assert_called_once()
            assert service._mcp_registry == mock_registry
            assert service._tools == []
            assert service._loaded is False

    def test_init_with_provided_registry(self):
        """Should use provided MCP registry."""
        mock_registry = MagicMock()

        service = ToolService(mcp_registry=mock_registry)

        assert service._mcp_registry == mock_registry


class TestToolServiceLoadTools:
    """Test ToolService.load_tools()."""

    @pytest.mark.asyncio
    async def test_load_tools_from_mcp_registry(self):
        """Should load tools from MCP registry."""
        mock_registry = MagicMock()
        mock_tool1 = MagicMock()
        mock_tool1.metadata.name = "search_web"
        mock_tool1.metadata.description = "Search the web"
        mock_tool2 = MagicMock()
        mock_tool2.metadata.name = "fetch_page"
        mock_tool2.metadata.description = "Fetch a web page"
        mock_registry.load_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        service = ToolService(mcp_registry=mock_registry)
        await service.load_tools()

        mock_registry.load_tools.assert_awaited_once()
        assert len(service.tools) == 2
        assert service._loaded is True

    @pytest.mark.asyncio
    async def test_load_tools_only_once(self):
        """Should only load tools once even if called multiple times."""
        mock_registry = MagicMock()
        mock_registry.load_tools = AsyncMock(return_value=[])

        service = ToolService(mcp_registry=mock_registry)
        await service.load_tools()
        await service.load_tools()  # Second call

        # Should only call load_tools once
        assert mock_registry.load_tools.await_count == 1

    @pytest.mark.asyncio
    async def test_load_tools_empty_registry(self):
        """Should handle empty tool list gracefully."""
        mock_registry = MagicMock()
        mock_registry.load_tools = AsyncMock(return_value=[])

        service = ToolService(mcp_registry=mock_registry)
        await service.load_tools()

        assert service.tools == []
        assert service._loaded is True


class TestToolServiceTools:
    """Test ToolService.tools property."""

    def test_tools_returns_loaded_tools(self):
        """Should return list of loaded FunctionTools."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)
        mock_tool = MagicMock()
        service._tools = [mock_tool]

        assert service.tools == [mock_tool]

    def test_tools_returns_empty_before_load(self):
        """Should return empty list before load_tools() called."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)

        assert service.tools == []


class TestToolServiceGetToolsByNames:
    """Test ToolService.get_tools_by_names()."""

    def test_get_tools_by_names_filters_correctly(self):
        """Should return only tools matching the requested names."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)

        # Create mock tools
        tool1 = MagicMock()
        tool1.metadata.name = "search_web"
        tool2 = MagicMock()
        tool2.metadata.name = "fetch_page"
        tool3 = MagicMock()
        tool3.metadata.name = "read_file"
        service._tools = [tool1, tool2, tool3]

        result = service.get_tools_by_names(["search_web", "fetch_page"])

        assert len(result) == 2
        assert tool1 in result
        assert tool2 in result
        assert tool3 not in result

    def test_get_tools_by_names_empty_list(self):
        """Should return empty list when no names requested."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)
        service._tools = [MagicMock()]

        result = service.get_tools_by_names([])

        assert result == []

    def test_get_tools_by_names_nonexistent_tool(self):
        """Should not include nonexistent tools in result."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)

        tool1 = MagicMock()
        tool1.metadata.name = "search_web"
        service._tools = [tool1]

        result = service.get_tools_by_names(["nonexistent"])

        assert result == []


class TestToolServiceListTools:
    """Test ToolService.list_tools()."""

    def test_list_tools_returns_metadata(self):
        """Should return tool metadata for API response."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)

        # Create mock tool with metadata
        tool = MagicMock()
        tool.metadata.name = "search_web"
        tool.metadata.description = "Search the web for information"
        tool.metadata.get_parameters_dict.return_value = {
            "query": {"type": "string", "description": "Search query"}
        }
        service._tools = [tool]

        result = service.list_tools()

        assert len(result) == 1
        assert result[0]["name"] == "search_web"
        assert result[0]["description"] == "Search the web for information"
        assert result[0]["parameters"] == {
            "query": {"type": "string", "description": "Search query"}
        }

    def test_list_tools_empty(self):
        """Should return empty list when no tools loaded."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)

        result = service.list_tools()

        assert result == []

    def test_list_tools_multiple(self):
        """Should return metadata for all tools."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)

        tool1 = MagicMock()
        tool1.metadata.name = "tool1"
        tool1.metadata.description = "First tool"
        tool1.metadata.get_parameters_dict.return_value = {}

        tool2 = MagicMock()
        tool2.metadata.name = "tool2"
        tool2.metadata.description = "Second tool"
        tool2.metadata.get_parameters_dict.return_value = {}

        service._tools = [tool1, tool2]

        result = service.list_tools()

        assert len(result) == 2
        assert {r["name"] for r in result} == {"tool1", "tool2"}


class TestToolServiceExecuteTool:
    """Test ToolService.execute_tool()."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Should execute tool and return success result."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)

        tool = MagicMock()
        tool.metadata.name = "search_web"
        tool.acall = AsyncMock(return_value={"results": ["result1", "result2"]})
        service._tools = [tool]

        result = await service.execute_tool("search_web", {"query": "test"})

        tool.acall.assert_awaited_once_with(query="test")
        assert result["success"] is True
        assert result["data"] == {"results": ["result1", "result2"]}

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_tool(self):
        """Should return error for unknown tool."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)
        service._tools = []

        result = await service.execute_tool("unknown_tool", {})

        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self):
        """Should return error when tool execution fails."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)

        tool = MagicMock()
        tool.metadata.name = "failing_tool"
        tool.acall = AsyncMock(side_effect=Exception("Tool failed"))
        service._tools = [tool]

        result = await service.execute_tool("failing_tool", {})

        assert result["success"] is False
        assert "Tool failed" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_with_multiple_params(self):
        """Should pass all parameters to tool."""
        mock_registry = MagicMock()
        service = ToolService(mcp_registry=mock_registry)

        tool = MagicMock()
        tool.metadata.name = "fetch_page"
        tool.acall = AsyncMock(return_value={"content": "page content"})
        service._tools = [tool]

        await service.execute_tool(
            "fetch_page", {"url": "https://example.com", "timeout": 30}
        )

        tool.acall.assert_awaited_once_with(url="https://example.com", timeout=30)
