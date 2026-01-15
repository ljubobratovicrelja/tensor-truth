"""Registry for managing multiple MCP server connections."""

import json
import logging
from pathlib import Path

from llama_index.core.tools import FunctionTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

from .config import MCPServerConfig, MCPServerType

logger = logging.getLogger(__name__)


class MCPServerRegistry:
    """Registry for managing multiple MCP server connections.

    This class handles:
    - Loading server configurations from JSON
    - Connecting to MCP servers
    - Aggregating tools from all connected servers
    """

    def __init__(self):
        self._servers: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, BasicMCPClient] = {}
        self._tools: list[FunctionTool] = []

    def register(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration.

        Args:
            config: Server configuration to register
        """
        self._servers[config.name] = config
        logger.info(f"Registered MCP server: {config.name}")

    def get_enabled_servers(self) -> list[MCPServerConfig]:
        """Get all enabled server configurations."""
        return [s for s in self._servers.values() if s.enabled]

    def load_from_json(self, config_path: str | Path) -> None:
        """Load server configurations from a JSON file.

        Args:
            config_path: Path to the JSON configuration file

        Expected JSON format:
        {
            "servers": [
                {
                    "name": "server-name",
                    "type": "stdio",
                    "command": "python",
                    "args": ["-m", "module.name"],
                    "enabled": true
                }
            ]
        }
        """
        path = Path(config_path)
        if not path.exists():
            logger.debug(f"MCP config file not found: {path}")
            return

        with open(path) as f:
            data = json.load(f)

        for server_data in data.get("servers", []):
            try:
                config = MCPServerConfig(
                    name=server_data["name"],
                    type=MCPServerType(server_data["type"]),
                    command=server_data.get("command"),
                    args=server_data.get("args", []),
                    url=server_data.get("url"),
                    description=server_data.get("description"),
                    enabled=server_data.get("enabled", True),
                )
                self.register(config)
            except (KeyError, ValueError) as e:
                logger.error(f"Invalid server config: {e}")

    async def load_tools(self) -> list[FunctionTool]:
        """Connect to all enabled servers and load their tools.

        Returns:
            List of LlamaIndex FunctionTool objects from all servers
        """
        all_tools = []

        for name, config in self._servers.items():
            if not config.enabled:
                continue

            try:
                logger.info(f"Connecting to MCP server: {name}")

                # Build connection parameters based on server type
                if config.type == MCPServerType.STDIO:
                    # For stdio, command is the executable, args are passed separately
                    client = BasicMCPClient(
                        command_or_url=config.command,
                        args=config.args or [],
                    )
                elif config.type == MCPServerType.SSE:
                    # For SSE, just pass the URL
                    client = BasicMCPClient(command_or_url=config.url)
                else:
                    logger.warning(f"Unknown server type for {name}: {config.type}")
                    continue

                # Create tool spec - BasicMCPClient manages connections per-operation
                mcp_spec = McpToolSpec(client=client)
                tools = await mcp_spec.to_tool_list_async()
                all_tools.extend(tools)
                self._clients[name] = client
                logger.info(f"Loaded {len(tools)} tools from {name}")

            except Exception as e:
                logger.error(f"Failed to connect to MCP server {name}: {e}")
                # Continue with other servers

        self._tools = all_tools
        return all_tools

    def list_servers(self) -> list[str]:
        """Get list of registered server names."""
        return list(self._servers.keys())

    async def close_all_connections(self) -> None:
        """Close all active MCP server connections.

        BasicMCPClient manages connections per-operation, so we just
        clear our reference to allow garbage collection.
        """
        self._clients.clear()
        self._tools.clear()
        logger.info("MCP server registry cleaned up")


# Default server configurations for TensorTruth
DEFAULT_SERVERS = [
    MCPServerConfig(
        name="tensor-truth-web",
        type=MCPServerType.STDIO,
        command="python",
        args=["-m", "tensortruth.mcp_servers"],  # Uses __main__.py entry point
        description="TensorTruth web search and page fetching tools",
        enabled=True,
    ),
]


def get_user_mcp_config_path() -> Path:
    """Get the path to user's MCP servers config file."""
    from tensortruth.app_utils.paths import get_user_data_dir

    return get_user_data_dir() / "mcp_servers.json"


def create_default_registry() -> MCPServerRegistry:
    """Create a registry with default TensorTruth servers and user-configured servers.

    Loads:
    1. Built-in default servers (web tools)
    2. User-configured servers from ~/.tensortruth/mcp_servers.json (if exists)

    Returns:
        MCPServerRegistry with all servers registered
    """
    registry = MCPServerRegistry()

    # Register built-in servers
    for server in DEFAULT_SERVERS:
        registry.register(server)

    # Load user-configured servers if config file exists
    user_config = get_user_mcp_config_path()
    if user_config.exists():
        logger.info(f"Loading user MCP servers from {user_config}")
        registry.load_from_json(user_config)

    return registry
