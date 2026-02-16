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

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket

from tensortruth.core.source_converter import SourceConverter
from tensortruth.services.config_service import ConfigService
from tensortruth.utils.web_search import web_search_stream

logger = logging.getLogger(__name__)

# REST endpoints (mounted under /api)
router = APIRouter()


class ToolCommand(ABC):
    """Base class for all tool/agent commands.

    Tool commands trigger external tools or agents, stream LLM responses
    via WebSocket, can appear anywhere in user input, and results are
    saved to session history.
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
    """Search the web and get AI-generated summary."""

    name = "web"
    aliases = ["search", "websearch"]
    description = "Search the web and get AI-generated summary"
    usage = "/web <query>[;instructions]"

    async def execute(self, args: str, session: dict, websocket: WebSocket) -> None:
        """Execute web search and stream AI summary.

        Uses the streaming web_search_stream() generator to emit tokens
        incrementally, matching the same pattern as RAG streaming.

        Args:
            args: Full text after command name (query[;instructions])
            session: Current session data
            websocket: WebSocket for streaming response
        """
        if not args or not args.strip():
            await websocket.send_json(
                {
                    "type": "error",
                    "detail": "Usage: /web <query>[;instructions for summary]",
                }
            )
            return

        # Parse query and optional instructions
        full_text = args.strip()
        query = full_text
        custom_instructions = None

        # Semicolon takes precedence over comma
        for separator in [";", ","]:
            if separator in full_text:
                parts = full_text.split(separator, 1)
                query = parts[0].strip()
                custom_instructions = parts[1].strip() if len(parts) > 1 else None
                break

        # Load configuration
        config_service = ConfigService()
        config = config_service.load()
        ws_config = config.web_search

        # Extract params from session with config defaults
        params = session.get("params", {})
        model_name = params.get("model", config.llm.default_model)
        ollama_url = params.get("ollama_url", config.ollama.base_url)
        context_window = params.get("context_window", config.llm.default_context_window)
        # Reranking params - uses session's reranker model if configured
        reranker_model = params.get("reranker_model")
        reranker_device = params.get("rag_device")

        try:
            full_response = ""
            sources_for_session = []

            # Use streaming generator for real-time token output
            async for chunk in web_search_stream(
                query=query,
                model_name=model_name,
                ollama_url=ollama_url,
                max_results=ws_config.ddg_max_results,
                max_pages=ws_config.max_pages_to_fetch,
                context_window=context_window,
                custom_instructions=custom_instructions,
                reranker_model=reranker_model,
                reranker_device=reranker_device,
                rerank_title_threshold=ws_config.rerank_title_threshold,
                rerank_content_threshold=ws_config.rerank_content_threshold,
                max_source_context_pct=ws_config.max_source_context_pct,
                input_context_pct=ws_config.input_context_pct,
            ):
                if chunk.agent_progress:
                    # Send agent progress for search/fetch phases
                    await websocket.send_json(
                        {"type": "agent_progress", **chunk.agent_progress}
                    )

                elif chunk.status:
                    # Send pipeline status (e.g., "generating")
                    await websocket.send_json(
                        {"type": "status", "status": chunk.status}
                    )

                elif chunk.token:
                    # Stream token to client
                    full_response += chunk.token
                    await websocket.send_json({"type": "token", "content": chunk.token})

                elif chunk.sources is not None:
                    # Store sources for session saving
                    sources_for_session = chunk.sources
                    # Convert to API format via SourceNode
                    if chunk.sources:
                        source_nodes = [
                            SourceConverter.to_api_schema(
                                SourceConverter.from_web_search_source(s)
                            )
                            for s in chunk.sources
                        ]
                        await websocket.send_json(
                            {"type": "sources", "data": source_nodes}
                        )

            # Check if this is first message (for title generation)
            messages = session.get("messages", [])
            is_first = len(messages) <= 1

            # Send final done message
            await websocket.send_json(
                {
                    "type": "done",
                    "content": full_response,
                    "confidence_level": "web_search",
                    "title_pending": is_first,
                    # Include sources in done for session saving
                    "sources": (
                        [
                            SourceConverter.to_api_schema(
                                SourceConverter.from_web_search_source(s)
                            )
                            for s in sources_for_session
                        ]
                        if sources_for_session
                        else None
                    ),
                }
            )

        except Exception as e:
            await websocket.send_json(
                {"type": "error", "detail": f"Web search failed: {str(e)}"}
            )


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
