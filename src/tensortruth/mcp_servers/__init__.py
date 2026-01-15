"""MCP (Model Context Protocol) servers for TensorTruth."""

import logging
import sys


def configure_mcp_logging() -> None:
    """Configure logging for MCP server mode.

    MCP uses stdout for JSON-RPC communication, so all logging must go to stderr.
    This must be called BEFORE importing modules that configure logging.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s - %(name)s - %(message)s",
        force=True,
    )

    # Suppress noisy loggers
    logging.getLogger("tensortruth").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)


from .web_tools_server import create_server, run_server  # noqa: E402

__all__ = ["configure_mcp_logging", "create_server", "run_server"]
