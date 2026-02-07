"""Context7 command — Python version (reference implementation).

This is the Python equivalent of context7.yaml, included as a reference
for users who want to write Python extensions instead of YAML.

To use:
    cp extension_library/commands/context7.py ~/.tensortruth/commands/
    (Make sure context7.yaml is NOT also present, or one will overwrite the other.)
"""

import re

from fastapi import WebSocket

from tensortruth.api.routes.commands import ToolCommand
from tensortruth.extensions.yaml_command import _extract_string


class Context7Command(ToolCommand):
    name = "context7"
    aliases = ["c7"]
    description = "Look up library docs from Context7"
    usage = "/context7 <library> <topic>"

    async def execute(self, args: str, session: dict, websocket: WebSocket) -> None:
        from tensortruth.api.deps import get_tool_service

        if not args or not args.strip():
            await websocket.send_json(
                {"type": "error", "detail": "Usage: /context7 <library> <topic>"}
            )
            return

        parts = args.strip().split(maxsplit=1)
        library_name = parts[0]
        topic = parts[1] if len(parts) > 1 else ""

        tool_service = get_tool_service()

        # Step 1: Resolve library ID
        await websocket.send_json(
            {
                "type": "agent_progress",
                "agent": "context7",
                "phase": "tool_call",
                "message": f"Resolving library: {library_name}...",
            }
        )

        resolve_result = await tool_service.execute_tool(
            "resolve-library-id",
            {"libraryName": library_name, "query": args.strip()},
        )

        if not resolve_result.get("success"):
            await websocket.send_json(
                {
                    "type": "error",
                    "detail": f"Failed to resolve library: {resolve_result.get('error')}",
                }
            )
            return

        # Parse library ID from result text
        text = _extract_string(resolve_result.get("data", ""))
        match = re.search(
            r"Context7-compatible library ID:\s*(\S+)", text, re.IGNORECASE
        )
        if not match:
            await websocket.send_json(
                {"type": "error", "detail": "Could not find library ID in response"}
            )
            return
        library_id = match.group(1)

        # Step 2: Fetch docs
        await websocket.send_json(
            {
                "type": "agent_progress",
                "agent": "context7",
                "phase": "tool_call",
                "message": f"Fetching docs for {library_id}...",
            }
        )

        docs_result = await tool_service.execute_tool(
            "query-docs",
            {"libraryId": library_id, "query": topic},
        )

        if not docs_result.get("success"):
            await websocket.send_json(
                {
                    "type": "error",
                    "detail": f"Failed to fetch docs: {docs_result.get('error')}",
                }
            )
            return

        content = _extract_string(docs_result.get("data", ""))
        await websocket.send_json({"type": "done", "content": content})


def register(command_registry, agent_service, tool_service):
    """Required entry point for Python extensions."""
    command_registry.register(Context7Command())
