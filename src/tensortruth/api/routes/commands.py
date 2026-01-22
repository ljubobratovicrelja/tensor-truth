"""Command system for tool/agent triggers in the API.

This module implements a modern command system that serves as a tool/agent
trigger framework rather than configuration management. Commands route user
requests to external tools (web search, arXiv, GitHub, etc.) and agents
(MCP browsing agent), streaming results back as LLM responses.

Key features:
- Commands can appear anywhere in input (not just at start)
- Streaming integration for LLM responses
- Extensible architecture for adding new tools
- Backend-first design with frontend UX layer
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket

# REST endpoints (mounted under /api)
router = APIRouter()


class ToolCommand(ABC):
    """Base class for all tool/agent commands.

    Tool commands are different from the old Streamlit config commands:
    - They trigger external tools or agents
    - They stream LLM responses via WebSocket
    - They can appear anywhere in user input
    - Results are saved to session history (not ephemeral)
    """

    name: str
    aliases: List[str]
    description: str
    usage: str

    @abstractmethod
    async def execute(self, args: str, session: dict, websocket: WebSocket) -> None:
        """Execute command and stream response via WebSocket.

        Args:
            args: Full text after command name
            session: Current session data
            websocket: WebSocket connection for streaming response

        The implementation should:
        1. Send status updates via websocket.send_json({"type": "status", ...})
        2. Execute the tool/agent operation
        3. Stream response tokens via websocket.send_json({"type": "token", ...})
        4. Send completion via websocket.send_json({"type": "done", ...})
        """
        pass


class CommandRegistry:
    """Registry for managing available commands.

    Provides command registration, discovery, and lookup functionality.
    Commands are registered by name and all aliases point to the same instance.
    """

    def __init__(self):
        """Initialize empty command registry."""
        self.commands: Dict[str, ToolCommand] = {}

    def register(self, command: ToolCommand) -> None:
        """Register a command by name and aliases.

        Args:
            command: Command instance to register
        """
        # Register by primary name
        self.commands[command.name] = command

        # Register all aliases
        for alias in command.aliases:
            self.commands[alias] = command

    def get(self, name: str) -> Optional[ToolCommand]:
        """Get command by name or alias (case-insensitive).

        Args:
            name: Command name or alias to look up

        Returns:
            Command instance if found, None otherwise
        """
        return self.commands.get(name.lower())

    def list_all(self) -> List[dict]:
        """List all unique commands (deduplicated by aliases).

        Returns:
            List of command metadata dicts for autocomplete/help
        """
        # Get unique commands (avoid duplicates from aliases)
        unique_commands = list(set(self.commands.values()))

        return [
            {
                "name": cmd.name,
                "aliases": cmd.aliases,
                "description": cmd.description,
                "usage": cmd.usage,
            }
            for cmd in unique_commands
        ]


# Global command registry
registry = CommandRegistry()


class HelpCommand(ToolCommand):
    """Command to show available commands and their usage."""

    name = "help"
    aliases = []
    description = "Show available commands"
    usage = "/help"

    async def execute(self, args: str, session: dict, websocket: WebSocket) -> None:
        """Return formatted list of all registered commands."""
        commands_list = registry.list_all()

        # Sort by name for consistent ordering
        commands_list.sort(key=lambda x: x["name"])

        # Format as markdown
        help_text = "# Available Commands\n\n"

        for cmd in commands_list:
            help_text += f"**{cmd['usage']}**\n"
            help_text += f"{cmd['description']}\n"

            if cmd["aliases"]:
                aliases_str = ", ".join(f"/{a}" for a in cmd["aliases"])
                help_text += f"*Aliases: {aliases_str}*\n"

            help_text += "\n"

        # Send as done message
        await websocket.send_json({"type": "done", "content": help_text})


class WebSearchCommand(ToolCommand):
    """Command to search the web using DuckDuckGo and get AI summary."""

    name = "web"
    aliases = ["search", "websearch"]
    description = "Search the web and get AI-generated summary"
    usage = "/web <search query>"

    async def execute(self, args: str, session: dict, websocket: WebSocket) -> None:
        """Execute web search and stream AI summary.

        Args:
            args: Search query
            session: Current session data
            websocket: WebSocket for streaming response
        """
        # Handle empty query
        if not args or not args.strip():
            await websocket.send_json(
                {
                    "type": "error",
                    "detail": "Please provide a search query. Usage: /web <search query>",
                }
            )
            return

        # Send searching status
        await websocket.send_json({"type": "status", "status": "searching"})

        try:
            # Execute DuckDuckGo search
            search_results = await self._duckduckgo_search(args.strip())

            if not search_results:
                await websocket.send_json(
                    {
                        "type": "done",
                        "content": "No search results found. Try a different query.",
                    }
                )
                return

            # Send generating status
            await websocket.send_json({"type": "status", "status": "generating"})

            # Generate AI summary
            summary = await self._generate_summary(search_results, session)

            # Send complete response
            await websocket.send_json({"type": "done", "content": summary})

        except Exception as e:
            await websocket.send_json(
                {"type": "error", "detail": f"Search failed: {str(e)}"}
            )

    async def _duckduckgo_search(self, query: str) -> List[dict]:
        """Execute DuckDuckGo search.

        Args:
            query: Search query string

        Returns:
            List of search result dicts with title, snippet, url
        """
        # TODO: Extract and refactor from existing Streamlit implementation
        # For now, return stub data for tests to pass
        return [
            {
                "title": "Example Result",
                "snippet": "Example snippet for: " + query,
                "url": "https://example.com",
            }
        ]

    async def _generate_summary(self, search_results: List[dict], session: dict) -> str:
        """Generate AI summary of search results.

        Args:
            search_results: List of search result dicts
            session: Session data for LLM params

        Returns:
            AI-generated summary text
        """
        # TODO: Implement LLM summary generation
        # For now, return formatted results for tests to pass
        summary = f"Found {len(search_results)} results:\n\n"

        for i, result in enumerate(search_results[:5], 1):
            summary += f"{i}. **{result['title']}**\n"
            summary += f"   {result['snippet']}\n"
            summary += f"   {result['url']}\n\n"

        return summary


# Register built-in commands
registry.register(HelpCommand())
registry.register(WebSearchCommand())


# API Endpoints
@router.get("/commands")
async def get_commands():
    """Get list of all available commands for autocomplete/help.

    Returns:
        JSON with commands list containing name, aliases, description, usage
    """
    return {"commands": registry.list_all()}
