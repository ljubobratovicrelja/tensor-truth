"""Entry point for running the MCP server as a module.

This configures logging to stderr BEFORE importing anything else,
preventing log messages from polluting MCP's stdout communication.

Usage: python -m tensortruth.mcp_servers
"""

# Configure logging BEFORE other imports (MCP uses stdout for communication)
from tensortruth.mcp_servers import configure_mcp_logging

configure_mcp_logging()

# Now import and run the server
import asyncio  # noqa: E402

from tensortruth.mcp_servers.web_tools_server import run_server  # noqa: E402

if __name__ == "__main__":
    asyncio.run(run_server())
