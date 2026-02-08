"""YAML-driven command implementations.

Provides ``YamlCommand`` (tool-pipeline commands) and ``YamlAgentCommand``
(agent-delegating commands), both subclassing the existing ``ToolCommand`` ABC.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List

from fastapi import WebSocket

from tensortruth.api.routes.commands import ToolCommand
from tensortruth.extensions.errors import TemplateResolutionError
from tensortruth.extensions.schema import CommandSpec, StepSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


def resolve_template(template: str, context: Dict[str, Any]) -> str:
    """Replace ``{{var}}`` and ``{{var.field}}`` with values from *context*.

    Supports dot-path walking for nested dicts (e.g. ``{{resolved.libraryID}}``).
    Non-string values are JSON-serialised so they can be embedded in tool params.

    Raises:
        TemplateResolutionError: If a referenced variable cannot be found.
    """

    def _replacer(match: re.Match) -> str:
        path = match.group(1).strip()
        value = _resolve_path(context, path)
        if isinstance(value, str):
            return value
        return json.dumps(value)

    return re.sub(r"\{\{(.+?)\}\}", _replacer, template)


def _resolve_path(context: Dict[str, Any], path: str) -> Any:
    """Walk a dot-separated *path* through *context*.

    Tries the full dot-path key first (``args.rest`` might be a direct key)
    then falls back to walking segment by segment.
    """
    # Fast path: exact key match (handles "args.0", "args.rest" etc.)
    if path in context:
        return context[path]

    parts = path.split(".")
    current: Any = context

    for i, part in enumerate(parts):
        # Try dict lookup first
        if isinstance(current, dict):
            if part in current:
                current = current[part]
                continue
            # Try integer index for list-like access
            try:
                idx = int(part)
                # Not applicable for dicts
            except ValueError:
                pass
            remaining = ".".join(parts[: i + 1])
            raise TemplateResolutionError(remaining)
        # Try list index
        if isinstance(current, (list, tuple)):
            try:
                idx = int(part)
                current = current[idx]
                continue
            except (ValueError, IndexError):
                remaining = ".".join(parts[: i + 1])
                raise TemplateResolutionError(remaining)
        remaining = ".".join(parts[: i + 1])
        raise TemplateResolutionError(remaining)

    return current


def _resolve_params(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolve templates in a parameter dict."""
    resolved: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            resolved[key] = resolve_template(value, context)
        elif isinstance(value, dict):
            resolved[key] = _resolve_params(value, context)
        elif isinstance(value, list):
            resolved[key] = [
                resolve_template(v, context) if isinstance(v, str) else v for v in value
            ]
        else:
            resolved[key] = value
    return resolved


def _build_args_context(args: str) -> Dict[str, Any]:
    """Build the initial template context from the raw *args* string."""
    parts = args.split() if args else []
    ctx: Dict[str, Any] = {"args": args}
    for i, part in enumerate(parts):
        ctx[f"args.{i}"] = part
    ctx["args.rest"] = " ".join(parts[1:]) if len(parts) > 1 else ""
    return ctx


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------


def _extract_string(raw_data: Any) -> str:
    """Get a clean string from a tool result.

    Handles LlamaIndex ``ToolOutput`` objects whose ``.content`` may be
    an MCP ``CallToolResult`` containing ``TextContent`` items, as well
    as plain strings and other types.
    """
    text = _dig_for_text(raw_data)
    if text is not None:
        return text
    return str(raw_data)


def _dig_for_text(obj: Any, depth: int = 0) -> str | None:
    """Recursively walk through ToolOutput / CallToolResult / TextContent."""
    if depth > 5:
        return None
    if isinstance(obj, str):
        return obj
    # List of content blocks (e.g. [TextContent(...), ...])
    if isinstance(obj, list):
        texts = [
            item.text
            for item in obj
            if hasattr(item, "text") and isinstance(item.text, str)
        ]
        if texts:
            return "\n".join(texts)
    # ToolOutput.raw_output often has the actual MCP result object
    # (while .content may already be str(result) — a lossy conversion)
    if hasattr(obj, "raw_output"):
        result = _dig_for_text(obj.raw_output, depth + 1)
        if result is not None:
            return result
    # Object with .content (ToolOutput, CallToolResult)
    if hasattr(obj, "content"):
        return _dig_for_text(obj.content, depth + 1)
    # Single TextContent-like object with .text
    if hasattr(obj, "text") and isinstance(obj.text, str):
        return obj.text
    return None


def _flatten_json_into_context(
    var_name: str, text: str, context: Dict[str, Any]
) -> None:
    """Try to parse *text* as structured data and flatten keys into *context*.

    Strategy (in order):
    1. JSON dict  → flatten keys directly
    2. JSON array → flatten first element's keys
    3. ``key: value`` lines in plain text → flatten as keys
    """
    # 1. Try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                context[f"{var_name}.{k}"] = v
            return
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            for k, v in parsed[0].items():
                context[f"{var_name}.{k}"] = v
            return
    except (json.JSONDecodeError, TypeError):
        pass


def _apply_result_extract(
    var_name: str, pattern: str, text: str, context: Dict[str, Any]
) -> None:
    """Apply a regex *pattern* to *text* and store matches in *context*.

    Named groups become ``{var_name}.{group_name}``.
    If there are no named groups, the whole match replaces ``{var_name}``.
    """
    match = re.search(pattern, text)
    if not match:
        logger.warning(f"result_extract pattern did not match: {pattern!r}")
        return
    groups = match.groupdict()
    if groups:
        for key, value in groups.items():
            context[f"{var_name}.{key}"] = value
    else:
        context[var_name] = match.group(0)


# ---------------------------------------------------------------------------
# YamlCommand – tool pipeline
# ---------------------------------------------------------------------------


class YamlCommand(ToolCommand):
    """A command defined by a YAML file with a ``steps`` tool pipeline."""

    def __init__(self, spec: CommandSpec):
        self.name = spec.name
        self.aliases = list(spec.aliases)
        self.description = spec.description
        self.usage = spec.usage or f"/{spec.name} <args>"
        self._steps: List[StepSpec] = spec.steps or []
        self._response_template = spec.response
        self._requires_mcp = spec.requires_mcp

    async def execute(self, args: str, session: dict, websocket: WebSocket) -> None:
        from tensortruth.api.deps import get_tool_service

        tool_service = get_tool_service()
        context = _build_args_context(args)
        last_result: Any = None

        for i, step in enumerate(self._steps):
            # Progress
            await websocket.send_json(
                {
                    "type": "agent_progress",
                    "agent": self.name,
                    "phase": "tool_call",
                    "message": f"Calling {step.tool}...",
                }
            )

            # Resolve params
            try:
                resolved_params = _resolve_params(step.params, context)
            except TemplateResolutionError as e:
                await websocket.send_json(
                    {"type": "error", "detail": f"Template error: {e}"}
                )
                return

            # Execute
            result = await tool_service.execute_tool(step.tool, resolved_params)

            if not result.get("success"):
                await websocket.send_json(
                    {
                        "type": "error",
                        "detail": f"Tool '{step.tool}' failed: {result.get('error', 'unknown')}",
                    }
                )
                return

            # Store result — handle LlamaIndex ToolOutput and raw strings
            raw_data = result.get("data", "")
            str_data = _extract_string(raw_data)

            logger.debug(
                f"Step '{step.tool}' result type={type(raw_data).__name__}, "
                f"str_data={str_data[:200]!r}"
            )

            last_result = str_data
            context["_last_result"] = str_data

            # Try to parse as JSON for dot-path access
            if step.result_var:
                context[step.result_var] = str_data
                _flatten_json_into_context(step.result_var, str_data, context)

                # Apply regex extraction if specified
                if step.result_extract:
                    _apply_result_extract(
                        step.result_var, step.result_extract, str_data, context
                    )

        # Resolve final response template
        try:
            final = resolve_template(self._response_template, context)
        except TemplateResolutionError as e:
            final = str(last_result) if last_result is not None else ""
            logger.warning(f"Response template error ({e}), using raw last result")

        await websocket.send_json({"type": "done", "content": final})


# ---------------------------------------------------------------------------
# YamlAgentCommand – delegates to a registered agent
# ---------------------------------------------------------------------------


class YamlAgentCommand(ToolCommand):
    """A command defined by a YAML file that delegates to a named agent."""

    def __init__(self, spec: CommandSpec):
        self.name = spec.name
        self.aliases = list(spec.aliases)
        self.description = spec.description
        self.usage = spec.usage or f"/{spec.name} <args>"
        self._agent_name = spec.agent or ""

    async def execute(self, args: str, session: dict, websocket: WebSocket) -> None:
        from tensortruth.api.deps import get_agent_service
        from tensortruth.services.agent_service import AgentCallbacks
        from tensortruth.services.config_service import ConfigService

        if not args or not args.strip():
            await websocket.send_json(
                {"type": "error", "detail": f"Usage: {self.usage}"}
            )
            return

        query = args.strip()

        # Build session params (same pattern as BrowseCommand)
        config_service = ConfigService()
        config = config_service.load()
        params = session.get("params", {})
        session_params = {
            "model": params.get("model", config.models.default_agent_reasoning_model),
            "ollama_url": params.get("ollama_url", config.ollama.base_url),
            "context_window": params.get(
                "context_window", config.ui.default_context_window
            ),
            "router_model": params.get("router_model"),
            "function_agent_model": params.get("function_agent_model"),
        }

        # Streaming callbacks
        full_response = ""
        tool_steps_data: list[dict] = []

        def on_progress(msg: str) -> None:
            asyncio.create_task(
                websocket.send_json(
                    {
                        "type": "agent_progress",
                        "agent": self._agent_name,
                        "phase": "processing",
                        "message": msg,
                    }
                )
            )

        def on_tool_call(tool_name: str, tool_params: dict) -> None:
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

        def on_tool_call_result(
            tool_name: str, tool_params: dict, output: str, is_error: bool
        ) -> None:
            step = {
                "tool": tool_name,
                "params": tool_params,
                "output": output[:500],
                "is_error": is_error,
            }
            tool_steps_data.append(step)
            asyncio.create_task(
                websocket.send_json(
                    {
                        "type": "tool_progress",
                        "tool": tool_name,
                        "action": "failed" if is_error else "completed",
                        "params": tool_params,
                        "output": output[:500],
                        "is_error": is_error,
                    }
                )
            )

        def on_token(token: str) -> None:
            nonlocal full_response
            full_response += token
            asyncio.create_task(
                websocket.send_json({"type": "token", "content": token})
            )

        callbacks = AgentCallbacks(
            on_progress=on_progress,
            on_tool_call=on_tool_call,
            on_tool_call_result=on_tool_call_result,
            on_token=on_token,
        )

        try:
            agent_service = get_agent_service()
            result = await agent_service.run(
                agent_name=self._agent_name,
                goal=query,
                callbacks=callbacks,
                session_params=session_params,
            )

            if result.error:
                await websocket.send_json(
                    {"type": "error", "detail": f"Agent failed: {result.error}"}
                )
                return

            done_msg: dict = {"type": "done", "content": result.final_answer}
            if tool_steps_data:
                done_msg["tool_steps"] = tool_steps_data
            await websocket.send_json(done_msg)

        except Exception as e:
            logger.error(f"YamlAgentCommand failed: {e}", exc_info=True)
            try:
                await websocket.send_json(
                    {"type": "error", "detail": f"Agent command failed: {e}"}
                )
            except Exception:
                pass
