"""Service for managing tools from MCP servers and other sources.

ToolService manages tool sources (MCP servers, future: user scripts) and provides
FunctionTools to agents. Uses LlamaIndex ToolSpec pattern - does NOT define
custom tool interfaces.
"""

import logging
from typing import Any, Dict, List, Optional

from llama_index.core.tools import FunctionTool

from tensortruth.agents.server_registry import (
    MCPServerRegistry,
    create_default_registry,
)

logger = logging.getLogger(__name__)


class ToolService:
    """Manages tool sources and provides FunctionTools to agents.

    Uses LlamaIndex ToolSpec pattern - does NOT define custom tool interfaces.
    Tools are loaded from MCP servers via the MCPServerRegistry.
    """

    def __init__(self, mcp_registry: Optional[MCPServerRegistry] = None):
        """Initialize ToolService.

        Args:
            mcp_registry: Optional MCPServerRegistry. If not provided,
                creates a default registry with built-in servers.
        """
        self._mcp_registry = mcp_registry or create_default_registry()
        self._tools: List[FunctionTool] = []
        self._loaded = False

    async def load_tools(self) -> None:
        """Load tools from all configured sources.

        Loads tools from MCP servers via the registry. This method is idempotent -
        calling it multiple times will only load tools once.
        """
        if self._loaded:
            logger.debug("Tools already loaded, skipping")
            return

        logger.info("Loading tools from MCP servers...")
        self._tools = await self._mcp_registry.load_tools()
        self._loaded = True
        logger.info(f"Loaded {len(self._tools)} tools")

    @property
    def tools(self) -> List[FunctionTool]:
        """Get all loaded FunctionTool instances.

        Returns:
            List of FunctionTool objects from all configured sources.
        """
        return self._tools

    def get_tools_by_names(self, names: List[str]) -> List[FunctionTool]:
        """Get specific tools by name for agent construction.

        Args:
            names: List of tool names to retrieve.

        Returns:
            List of FunctionTool objects matching the requested names.
        """
        return [t for t in self._tools if t.metadata.name in names]

    def list_tools(self) -> List[Dict[str, Any]]:
        """List tool metadata for API response.

        Returns:
            List of dictionaries with tool metadata (name, description, parameters).
        """
        return [
            {
                "name": t.metadata.name,
                "description": t.metadata.description,
                "parameters": t.metadata.get_parameters_dict(),
            }
            for t in self._tools
        ]

    async def execute_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool by name.

        Used for direct /tool commands to execute a specific tool.

        Args:
            name: Name of the tool to execute.
            params: Parameters to pass to the tool.

        Returns:
            Dictionary with success status and either data or error.
        """
        tool = next((t for t in self._tools if t.metadata.name == name), None)
        if not tool:
            return {"success": False, "error": f"Unknown tool: {name}"}
        try:
            result = await tool.acall(**params)
            return {"success": True, "data": result}
        except Exception as e:
            logger.error(f"Tool execution failed for {name}: {e}")
            return {"success": False, "error": str(e)}
