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

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket

from tensortruth.api.deps import get_agent_service
from tensortruth.core.source import SourceNode
from tensortruth.core.source_converter import SourceConverter
from tensortruth.services.agent_service import AgentCallbacks
from tensortruth.services.config_service import ConfigService
from tensortruth.utils.web_search import web_search_stream

logger = logging.getLogger(__name__)

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
        model_name = params.get("model", config.models.default_agent_reasoning_model)
        ollama_url = params.get("ollama_url", config.ollama.base_url)
        context_window = params.get("context_window", config.ui.default_context_window)
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


class BrowseCommand(ToolCommand):
    """Execute the browse agent for autonomous web research."""

    name = "browse"
    aliases = ["research"]
    description = "Autonomous web research agent for complex queries"
    usage = "/browse <query>"

    async def execute(self, args: str, session: dict, websocket: WebSocket) -> None:
        """Execute browse agent and stream response.

        Args:
            args: Research query/goal
            session: Current session data
            websocket: WebSocket connection for streaming
        """
        # 1. Validate input
        if not args or not args.strip():
            logger.warning("Browse command called with empty query")
            await websocket.send_json(
                {"type": "error", "detail": "Usage: /browse <query>"}
            )
            return

        query = args.strip()
        logger.info(f"Browse command started with query: {query}")

        # 2. Load config and extract session parameters
        config_service = ConfigService()
        config = config_service.load()
        params = session.get("params", {})

        # 3. Build conversation history using ChatHistoryService
        from tensortruth.services.chat_history import ChatHistoryService

        chat_history_service = ChatHistoryService(config)
        session_messages = session.get("messages", [])
        conversation_history = chat_history_service.build_history(
            session_messages,
            max_turns=5,  # Last 5 turns for context
            apply_cleaning=True,
        )

        # Build session_params with config defaults
        session_params = {
            "model": params.get("model", config.models.default_agent_reasoning_model),
            "ollama_url": params.get("ollama_url", config.ollama.base_url),
            "context_window": params.get(
                "context_window", config.ui.default_context_window
            ),
            "reranker_model": params.get("reranker_model"),  # Already in session params
            "rag_device": params.get("rag_device"),  # Already in session params
            "router_model": params.get("router_model"),  # Already in session params
            "conversation_history": conversation_history,  # NEW: For context-aware queries
        }

        logger.info(
            f"Session params: model={session_params['model']}, "
            f"context_window={session_params['context_window']}, "
            f"ollama_url={session_params['ollama_url']}, "
            f"reranker_model={session_params['reranker_model']}, "
            f"rag_device={session_params['rag_device']}, "
            f"router_model={session_params['router_model']}"
        )
        logger.info(f"Raw params from session: {params}")

        # 3. Create WebSocket streaming callbacks with enhanced progress
        full_response = ""
        pages_fetched = []

        def sync_on_progress(msg: str):
            """Send progress messages to UI."""
            logger.info(f"Agent progress: {msg}")
            # Parse phase and clean message content (handles "phase:message" format)
            phase = _parse_phase(msg)
            clean_message = _parse_message_content(msg)
            asyncio.create_task(
                websocket.send_json(
                    {
                        "type": "agent_progress",
                        "agent": "browse",
                        "phase": phase,
                        "message": clean_message,
                    }
                )
            )

        def sync_on_tool_call(tool_name: str, tool_params: dict):
            """Translate tool calls into user-friendly progress messages."""
            # Log tool call
            logger.info(f"Tool call: {tool_name} with params: {tool_params}")

            # Create user-friendly message based on tool
            if tool_name == "search_web":
                # Extract queries (could be str or list)
                queries = tool_params.get("queries", [])
                if isinstance(queries, str):
                    queries = [queries]

                # Show each query being searched
                queries_str = ", ".join(f'"{q}"' for q in queries)
                message = f"Searching DDG with {len(queries)} queries: {queries_str}"
                phase = "searching"

                # Send detailed agent_progress for search
                asyncio.create_task(
                    websocket.send_json(
                        {
                            "type": "agent_progress",
                            "agent": "browse",
                            "phase": phase,
                            "message": message,
                            "details": {
                                "queries": queries,
                                "query_count": len(queries),
                            },
                        }
                    )
                )

            elif tool_name == "fetch_page":
                url = tool_params.get("url", "")
                pages_fetched.append(url)
                count = len(pages_fetched)

                # Extract domain for cleaner display
                from urllib.parse import urlparse

                domain = urlparse(url).netloc

                message = f"Fetching page {count}: {domain}"
                phase = "fetching"

                # Send detailed agent_progress for fetch
                asyncio.create_task(
                    websocket.send_json(
                        {
                            "type": "agent_progress",
                            "agent": "browse",
                            "phase": phase,
                            "message": message,
                            "details": {
                                "url": url,
                                "page_number": count,
                                "domain": domain,
                            },
                        }
                    )
                )

            elif tool_name == "fetch_pages_batch":
                urls = tool_params.get("urls", [])
                pages_fetched.extend(urls)

                # Extract domains for cleaner display
                from urllib.parse import urlparse

                domains = [urlparse(url).netloc for url in urls]
                domains_str = ", ".join(domains[:3])  # Show first 3
                if len(domains) > 3:
                    domains_str += f" (+{len(domains) - 3} more)"

                message = f"Fetching {len(urls)} pages in parallel: {domains_str}"
                phase = "fetching"

                # Send detailed agent_progress for batch fetch
                asyncio.create_task(
                    websocket.send_json(
                        {
                            "type": "agent_progress",
                            "agent": "browse",
                            "phase": phase,
                            "message": message,
                            "details": {
                                "urls": urls,
                                "page_count": len(urls),
                                "domains": domains,
                            },
                        }
                    )
                )

            elif tool_name == "search_focused":
                search_query = tool_params.get("query", "")
                domain = tool_params.get("domain", "")
                message = f"Searching {domain}: {search_query}"
                phase = "searching"

                # Send detailed agent_progress for focused search
                asyncio.create_task(
                    websocket.send_json(
                        {
                            "type": "agent_progress",
                            "agent": "browse",
                            "phase": phase,
                            "message": message,
                        }
                    )
                )

            # Also send raw tool_progress for debugging
            asyncio.create_task(
                websocket.send_json(
                    {
                        "type": "tool_progress",
                        "tool": tool_name,
                        "action": "calling",
                        "params": tool_params,
                    }
                )
            )

        def sync_on_token(token: str):
            """Stream synthesis tokens."""
            nonlocal full_response
            full_response += token
            asyncio.create_task(
                websocket.send_json(
                    {
                        "type": "token",
                        "content": token,
                    }
                )
            )

        callbacks = AgentCallbacks(
            on_progress=sync_on_progress,
            on_tool_call=sync_on_tool_call,
            on_token=sync_on_token,
        )

        try:
            # 4. Execute agent
            logger.info("Retrieving AgentService instance")
            agent_service = get_agent_service()

            if agent_service is None:
                raise RuntimeError("AgentService is not initialized")

            logger.info(
                f"Calling AgentService.run() with agent_name='browse', goal='{query}'"
            )

            result = await agent_service.run(
                agent_name="browse",
                goal=query,
                callbacks=callbacks,
                session_params=session_params,
            )

            logger.info(
                f"Agent completed: iterations={result.iterations}, "
                f"tools_called={result.tools_called}, "
                f"urls_browsed={len(result.urls_browsed)}, "
                f"error={result.error}"
            )

            # 5. Handle errors
            if result.error:
                logger.error(f"Browse agent failed: {result.error}")
                await websocket.send_json(
                    {"type": "error", "detail": f"Browse agent failed: {result.error}"}
                )
                return

            # Check if agent didn't find any sources
            if not result.urls_browsed:
                logger.warning("Browse agent completed but found no sources")

            # 6. Convert SourceNode objects to frontend format
            source_nodes = []
            for source in result.sources:
                if isinstance(source, SourceNode):
                    # Use SourceConverter for consistent API format
                    api_source = SourceConverter.to_api_schema(source)
                    # Add browse_agent marker for backward compat
                    api_source["metadata"]["source_type"] = "browse_agent"
                    source_nodes.append(api_source)
                else:
                    # Fallback for backward compatibility (shouldn't happen)
                    logger.warning(f"Unexpected source type: {type(source)}")
                    url = (
                        str(source)
                        if not isinstance(source, dict)
                        else source.get("url", str(source))
                    )
                    source_nodes.append(
                        {
                            "text": url,
                            "score": 1.0,
                            "metadata": {
                                "url": url,
                                "source_type": "browse_agent",
                            },
                        }
                    )

            # 7. Send sources
            if source_nodes:
                logger.info(f"Sending {len(source_nodes)} sources to client")
                await websocket.send_json(
                    {
                        "type": "sources",
                        "data": source_nodes,
                    }
                )

            # 8. Send done message
            messages = session.get("messages", [])
            is_first = len(messages) <= 1

            logger.info(
                f"Sending done message (response length: {len(result.final_answer)})"
            )
            await websocket.send_json(
                {
                    "type": "done",
                    "content": result.final_answer,
                    "confidence_level": "browse_agent",
                    "title_pending": is_first,
                    "sources": source_nodes if source_nodes else None,
                }
            )

            logger.info("Browse command completed successfully")

        except Exception as e:
            logger.error(f"Browse command failed with exception: {e}", exc_info=True)
            try:
                await websocket.send_json(
                    {"type": "error", "detail": f"Browse command failed: {str(e)}"}
                )
            except Exception:
                logger.debug("Could not send error to client (WebSocket closed)")


# Pipeline phases from SourceFetchPipeline
PIPELINE_PHASES = {
    "loading_model",
    "fetching",
    "ranking_titles",
    "ranking_content",
    "fitting",
}


def _parse_phase(status_msg: str) -> str:
    """Parse phase from AgentService progress message.

    Maps status messages to phase names for frontend display.
    Handles both plain messages and phase-prefixed messages (e.g., "fetching:Fetching batch 1...")
    """
    # Check for phase prefix (e.g., "fetching:Fetching batch 1...")
    if ":" in status_msg:
        potential_phase = status_msg.split(":", 1)[0].lower()
        if potential_phase in PIPELINE_PHASES:
            return potential_phase

    # Fallback to keyword matching
    msg_lower = status_msg.lower()
    if "starting" in msg_lower:
        return "starting"
    elif "loading" in msg_lower and "model" in msg_lower:
        return "loading_model"
    elif "ranking" in msg_lower and "title" in msg_lower:
        return "ranking_titles"
    elif "ranking" in msg_lower and "content" in msg_lower:
        return "ranking_content"
    elif "ranking" in msg_lower:
        return "ranking_content"  # Default ranking to content
    elif "searching" in msg_lower or "search" in msg_lower:
        return "searching"
    elif "fetching" in msg_lower or "fetch" in msg_lower:
        return "fetching"
    elif "fitting" in msg_lower or "fitted" in msg_lower:
        return "fitting"
    elif "synthesizing" in msg_lower or "synthesis" in msg_lower:
        return "summarizing"
    else:
        return "processing"


def _parse_message_content(status_msg: str) -> str:
    """Extract message content, stripping phase prefix if present.

    Args:
        status_msg: Raw status message, possibly with phase prefix

    Returns:
        Clean message for display
    """
    if ":" in status_msg:
        potential_phase = status_msg.split(":", 1)[0].lower()
        if potential_phase in PIPELINE_PHASES:
            return status_msg.split(":", 1)[1].strip()
    return status_msg


# Register built-in commands
registry.register(HelpCommand())
registry.register(WebSearchCommand())
registry.register(BrowseCommand())


# API Endpoints
@router.get("/commands")
async def get_commands():
    """Get list of all available commands for autocomplete/help.

    Returns:
        JSON with commands list containing name, aliases, description, usage
    """
    return {"commands": registry.list_all()}
