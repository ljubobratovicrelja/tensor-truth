"""Service for managing tools from MCP servers and other sources.

ToolService manages tool sources (MCP servers, future: user scripts) and provides
FunctionTools to agents. Uses LlamaIndex ToolSpec pattern - does NOT define
custom tool interfaces.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from llama_index.core.tools import FunctionTool

from tensortruth.agents.server_registry import (
    MCPServerRegistry,
    create_default_registry,
)
from tensortruth.services import builtin_tools

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

    async def _load_builtin_tools(self) -> List[FunctionTool]:
        """Load built-in tools for agents.

        Creates FunctionTool instances for built-in utilities like web search
        and page fetching. These tools are always available to agents.

        Returns:
            List of FunctionTool instances for built-in tools.
        """
        logger.info("Loading built-in tools...")

        # Import pydantic for explicit schemas
        from pydantic import BaseModel, Field

        # Define explicit parameter schemas for clarity
        class SearchWebInput(BaseModel):
            """Input for search_web tool - supports multi-query searches."""

            queries: Union[str, List[str]] = Field(
                description=(
                    "Single search query string OR list of diverse query strings. "
                    "Use multiple queries for comprehensive coverage: "
                    "['broad overview', 'technical details', 'recent 2025-2026 developments']"
                )
            )
            max_results_per_query: int = Field(
                default=5, description="Maximum results per query (default: 5)"
            )

        class FetchPageInput(BaseModel):
            """Input for fetch_page tool."""

            url: str = Field(description="The URL to fetch")
            timeout: int = Field(
                default=10, description="Timeout in seconds for the request"
            )

        class FetchPagesBatchInput(BaseModel):
            """Input for fetch_pages_batch tool - fetch multiple pages in parallel."""

            urls: List[str] = Field(
                description=(
                    "List of URLs to fetch in parallel (recommended: 3-5 URLs). "
                    "All pages are fetched simultaneously for efficiency."
                )
            )
            timeout: int = Field(
                default=10, description="Timeout in seconds per page (default: 10)"
            )

        class SearchFocusedInput(BaseModel):
            """Input for search_focused tool."""

            query: str = Field(description="The search query string")
            domain: str = Field(
                description="Domain to search within (e.g., 'stackoverflow.com')"
            )
            max_results: int = Field(
                default=5, description="Maximum number of results to return"
            )

        class SearchArxivInput(BaseModel):
            """Input for search_arxiv tool."""

            query: str = Field(
                description=(
                    "ArXiv search query. Supports field prefixes: "
                    "ti: (title), au: (author), abs: (abstract), cat: (category). "
                    "Boolean: AND, OR, ANDNOT. Use quotes for exact phrases. "
                    "Example: 'ti:\"gaussian splatting\" AND abs:face'"
                )
            )
            max_results: int = Field(
                default=5, description="Maximum papers to return (default: 5)"
            )
            sort_by: str = Field(
                default="relevance",
                description="Sort by: 'relevance', 'submitted', or 'updated'",
            )

        class GetArxivPaperInput(BaseModel):
            """Input for get_arxiv_paper tool."""

            paper_id: str = Field(
                description=(
                    "ArXiv paper ID (e.g. '2301.12345', 'hep-th/9901001', "
                    "or full URL)"
                )
            )

        tools = [
            FunctionTool.from_defaults(
                async_fn=builtin_tools.search_web,
                name="search_web",
                description=(
                    "Search the web using DuckDuckGo. "
                    "SUPPORTS MULTIPLE QUERIES: Pass a list of diverse queries "
                    "for comprehensive coverage. "
                    "Example: queries=['AI overview 2026', 'AI papers', 'AI news']. "
                    "Results are combined and deduplicated. "
                    "Required: queries (str or list[str]). "
                    "Optional: max_results_per_query (int, default=5)."
                ),
                fn_schema=SearchWebInput,
            ),
            FunctionTool.from_defaults(
                async_fn=builtin_tools.fetch_page,
                name="fetch_page",
                description=(
                    "Fetch a web page and convert it to clean markdown. "
                    "Uses domain-specific handlers for Wikipedia, GitHub, arXiv, YouTube, etc. "
                    "Returns markdown content on success or error message on failure. "
                    "Required parameter: url (str). Optional: timeout (int, default=10)."
                ),
                fn_schema=FetchPageInput,
            ),
            FunctionTool.from_defaults(
                async_fn=builtin_tools.fetch_pages_batch,
                name="fetch_pages_batch",
                description=(
                    "Fetch multiple web pages in parallel "
                    "(RECOMMENDED for research). "
                    "Much faster than calling fetch_page multiple times. "
                    "Returns JSON array with results for each URL. "
                    "Example: urls=['https://...', 'https://...', 'https://...']. "
                    "Required parameter: urls (list[str]). "
                    "Optional: timeout (int, default=10)."
                ),
                fn_schema=FetchPagesBatchInput,
            ),
            FunctionTool.from_defaults(
                async_fn=builtin_tools.search_focused,
                name="search_focused",
                description=(
                    "Search within a specific domain using DuckDuckGo site search. "
                    "Useful for finding content on specific websites like "
                    "Stack Overflow, GitHub, official documentation sites, etc. "
                    "Returns JSON results. "
                    "Required parameters: query (str), domain (str). "
                    "Optional: max_results (int, default=5)."
                ),
                fn_schema=SearchFocusedInput,
            ),
            FunctionTool.from_defaults(
                async_fn=builtin_tools.search_arxiv,
                name="search_arxiv",
                description=(
                    "Search arXiv for academic papers. "
                    "Supports arXiv query syntax for precise searches: "
                    "ti: (title), au: (author), abs: (abstract), cat: (category). "
                    "Use AND/OR/ANDNOT for boolean logic, quotes for exact phrases. "
                    'Example: \'ti:"gaussian splatting" AND ti:"face"\' or '
                    "'au:vaswani AND ti:attention'. "
                    "Common categories: cs.CV (vision), cs.CL (NLP), cs.LG (ML), "
                    "cs.AI, stat.ML, eess.IV (image/video). "
                    "Returns JSON array of papers with title, authors, abstract, "
                    "categories, and PDF URL. "
                    "For thorough research, call multiple times with varied queries. "
                    "Required: query (str). "
                    "Optional: max_results (int, default=5), "
                    "sort_by ('relevance'|'submitted'|'updated', default='relevance')."
                ),
                fn_schema=SearchArxivInput,
            ),
            FunctionTool.from_defaults(
                async_fn=builtin_tools.get_arxiv_paper,
                name="get_arxiv_paper",
                description=(
                    "Get detailed info about a specific arXiv paper by its ID. "
                    "Returns structured markdown with title, authors, dates, "
                    "categories, abstract, comments, DOI, and PDF URL. "
                    "Required: paper_id (str) — e.g. '2301.12345', "
                    "'hep-th/9901001', or full arXiv URL."
                ),
                fn_schema=GetArxivPaperInput,
            ),
        ]

        logger.info(f"Loaded {len(tools)} built-in tools")
        return tools

    async def load_tools(self) -> None:
        """Load tools from all configured sources.

        Loads built-in tools and tools from MCP servers via the registry.
        This method is idempotent - calling it multiple times will only load tools once.
        """
        if self._loaded:
            logger.debug("Tools already loaded, skipping")
            return

        # Load built-in tools
        builtin = await self._load_builtin_tools()

        # Load MCP tools
        logger.info("Loading tools from MCP servers...")
        mcp_tools = await self._mcp_registry.load_tools()

        # Combine all tools
        self._tools = builtin + mcp_tools
        self._loaded = True
        logger.info(
            f"Loaded {len(self._tools)} total tools "
            f"({len(builtin)} built-in, {len(mcp_tools)} MCP)"
        )

    def add_tool(self, tool: FunctionTool) -> None:
        """Add a tool after initial load (for user extensions).

        Args:
            tool: A FunctionTool instance to register.
        """
        self._tools.append(tool)

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
