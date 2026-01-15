"""Configuration types for MCP agents."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MCPServerType(str, Enum):
    """Supported MCP server transport types."""

    STDIO = "stdio"
    SSE = "sse"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection.

    Attributes:
        name: Unique identifier for this server
        type: Transport type (stdio or sse)
        command: Command to run (for stdio type)
        args: Command arguments (for stdio type)
        url: Server URL (for sse type)
        description: Human-readable description
        enabled: Whether this server is active
    """

    name: str
    type: MCPServerType
    command: Optional[str] = None
    args: Optional[list[str]] = field(default_factory=list)
    url: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.type == MCPServerType.STDIO and not self.command:
            raise ValueError(
                f"Server '{self.name}' with type 'stdio' requires 'command'"
            )
        if self.type == MCPServerType.SSE and not self.url:
            raise ValueError(f"Server '{self.name}' with type 'sse' requires 'url'")


@dataclass
class AgentResult:
    """Result from agent execution.

    Attributes:
        final_answer: The synthesized response text
        iterations: Number of reasoning iterations performed
        tools_called: List of tool names that were invoked
        urls_browsed: List of URLs that were fetched during execution
        error: Error message if execution failed
    """

    final_answer: str
    iterations: int = 0
    tools_called: list[str] = field(default_factory=list)
    urls_browsed: list[str] = field(default_factory=list)
    error: Optional[str] = None
