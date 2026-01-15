"""MCP server exposing web search and page fetching tools.

Run standalone: python -m tensortruth.mcp_servers.web_tools_server
"""

import asyncio
import json
import logging
from typing import Any

import aiohttp
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Import existing web functions
from tensortruth.utils.web_search import fetch_page_as_markdown, search_duckduckgo

logger = logging.getLogger(__name__)


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("tensor-truth-web-tools")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="search_web",
                description=(
                    "Search DuckDuckGo for information on any topic. "
                    "Returns a list of results with url, title, and snippet."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="fetch_page",
                description=(
                    "Fetch a web page and convert it to clean markdown. "
                    "Automatically uses optimized handlers for Wikipedia, GitHub, "
                    "arXiv, and YouTube. Returns the page content as markdown text."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["url"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute a tool and return results."""
        logger.info(f"Calling tool: {name} with args: {arguments}")

        if name == "search_web":
            query = arguments["query"]
            max_results = arguments.get("max_results", 10)

            results = await search_duckduckgo(query, max_results)

            # Format results as JSON
            return [
                TextContent(
                    type="text",
                    text=json.dumps(results, indent=2),
                )
            ]

        elif name == "fetch_page":
            url = arguments["url"]
            timeout = arguments.get("timeout", 10)

            async with aiohttp.ClientSession() as session:
                content, status, error = await fetch_page_as_markdown(
                    url, session, timeout
                )

            if status == "success" and content:
                return [TextContent(type="text", text=content)]
            else:
                error_msg = error or f"Failed with status: {status}"
                return [TextContent(type="text", text=f"Error: {error_msg}")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def run_server():
    """Run the MCP server using stdio transport."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    # Configure logging when running standalone (MCP uses stdout for communication)
    from tensortruth.mcp_servers import configure_mcp_logging

    configure_mcp_logging()
    asyncio.run(run_server())
