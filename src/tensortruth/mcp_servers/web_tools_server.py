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
                    "SUPPORTS MULTIPLE QUERIES: Pass a single query string or list of diverse queries "
                    "for comprehensive coverage. Results are combined and deduplicated. "
                    "Returns a list of results with url, title, and snippet."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "queries": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            ],
                            "description": (
                                "Single search query string OR list of diverse query strings. "
                                "Use multiple queries for comprehensive coverage."
                            ),
                        },
                        "max_results_per_query": {
                            "type": "integer",
                            "description": "Maximum results per query (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["queries"],
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
            queries = arguments["queries"]
            max_results_per_query = arguments.get("max_results_per_query", 5)

            # Normalize to list
            query_list = queries if isinstance(queries, list) else [queries]

            # Execute all queries in parallel
            tasks = [
                search_duckduckgo(query, max_results=max_results_per_query)
                for query in query_list
            ]
            results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine and deduplicate by URL
            seen_urls = set()
            combined_results = []

            for query, results in zip(query_list, results_per_query):
                # Handle exceptions from individual queries
                if isinstance(results, Exception):
                    logger.warning(f"Query '{query}' failed: {results}")
                    continue

                # Add results, deduplicating by URL
                if not isinstance(results, list):
                    continue
                for result in results:
                    url = result.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        result["query"] = query  # Track which search returned this
                        combined_results.append(result)

            # Format results as JSON
            return [
                TextContent(
                    type="text",
                    text=json.dumps(combined_results, indent=2),
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
