"""Chat endpoints including WebSocket streaming."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from tensortruth.api.deps import (
    ChatServiceDep,
    IntentServiceDep,
    ProjectServiceDep,
    SessionServiceDep,
    get_chat_service,
    get_config_service,
    get_document_service,
    get_project_service,
    get_rag_service,
    get_session_service,
    get_tool_service,
)
from tensortruth.api.routes.commands import registry as command_registry
from tensortruth.api.schemas import (
    ChatRequest,
    ChatResponse,
    IntentRequest,
    IntentResponse,
    SourceNode,
)
from tensortruth.app_utils.paths import get_project_index_dir
from tensortruth.app_utils.title_generation import generate_smart_title_async
from tensortruth.core.ollama import check_tool_call_support, get_ollama_url
from tensortruth.services import ProjectService, SessionService
from tensortruth.services.models import ToolProgress
from tensortruth.services.orchestrator_service import (
    OrchestratorService,
    load_module_descriptions,
)
from tensortruth.services.orchestrator_stream import OrchestratorStreamTranslator

logger = logging.getLogger(__name__)

# REST endpoints (mounted under /api)
rest_router = APIRouter()
# WebSocket endpoint (mounted at root, not under /api)
ws_router = APIRouter()
# Legacy alias for backwards compatibility
router = rest_router


@dataclass
class ChatContext:
    """Context for a chat request.

    Encapsulates all the information needed to execute a chat query,
    including session configuration and message history.
    """

    session_id: str
    prompt: str
    modules: List[str]
    params: Dict[str, Any]
    session_messages: List[Dict[str, Any]]
    additional_index_paths: List[str] = field(default_factory=list)

    @classmethod
    def from_session(
        cls,
        session_id: str,
        prompt: str,
        session: Dict[str, Any],
        pdf_service: Any,
        project_service: Optional[ProjectService] = None,
    ) -> "ChatContext":
        """Create ChatContext from session data and PDF service.

        When the session belongs to a project, merges project-level catalog
        modules and collects project/session index paths.

        Args:
            session_id: The session identifier.
            prompt: The user's current prompt.
            session: Session dict containing modules, params, messages.
            pdf_service: PDFService instance to check for PDF index.
            project_service: Optional ProjectService for project context resolution.

        Returns:
            ChatContext instance with all fields populated.
        """
        session_modules = session.get("modules") or []
        additional_index_paths: List[str] = []

        # Resolve project context if session belongs to a project
        project_id = session.get("project_id")
        project_modules: List[str] = []
        composed_system_prompt: Optional[str] = None
        project: Optional[Dict[str, Any]] = None
        if project_id and project_service:
            project_data = project_service.load()
            project = project_service.get_project(project_id, project_data)
            if project:
                # Extract indexed catalog modules from project
                # catalog_modules is a dict: {"module_name": {"status": "indexed"}}
                catalog = project.get("catalog_modules", {})
                project_modules = [
                    module_name
                    for module_name, mod_info in catalog.items()
                    if isinstance(mod_info, dict)
                    and mod_info.get("status") == "indexed"
                ]

                # Check for project index directory on disk
                project_index_dir = get_project_index_dir(project_id)
                chroma_db_path = project_index_dir / "chroma.sqlite3"
                if chroma_db_path.exists():
                    additional_index_paths.append(str(project_index_dir))

                # Compose project context into system_prompt from
                # project name, description, and custom instructions
                parts: List[str] = []
                project_name = (project.get("name") or "").strip()
                project_desc = (project.get("description") or "").strip()
                project_instructions = (
                    (project.get("config") or {}).get("system_prompt") or ""
                ).strip()

                if project_name:
                    parts.append(f"Project: {project_name}")
                if project_desc:
                    parts.append(project_desc)
                if project_instructions:
                    parts.append(project_instructions)

                if parts:
                    composed_system_prompt = "\n\n".join(parts)
            else:
                logger.warning(
                    f"Project {project_id} not found for session {session_id}, "
                    "falling back to session-only context"
                )

        # Merge modules: project modules first, then session, deduplicated
        merged_modules = list(dict.fromkeys(project_modules + session_modules))

        # Add session PDF index if it exists
        session_index_path = pdf_service.get_index_path()
        if session_index_path:
            additional_index_paths.append(str(session_index_path))

        # Build params, overriding system_prompt for project sessions
        params = dict(session.get("params", {}))
        if project and composed_system_prompt:
            params["system_prompt"] = composed_system_prompt

        return cls(
            session_id=session_id,
            prompt=prompt,
            modules=merged_modules,
            params=params,
            session_messages=session.get("messages", []),
            additional_index_paths=additional_index_paths,
        )


def _save_assistant_message(
    session_service: SessionService,
    session_id: str,
    content: str,
    sources: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]],
) -> None:
    """Save assistant message with sources and metrics to session.

    Args:
        session_service: The session service instance.
        session_id: The session identifier.
        content: The assistant's response content.
        sources: List of source dicts (may be empty).
        metrics: Optional retrieval metrics.
    """
    assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
    if sources:
        assistant_message["sources"] = sources
    if metrics:
        assistant_message["metrics"] = metrics

    data = session_service.load()  # Reload in case of concurrent updates
    data = session_service.add_message(session_id, assistant_message, data)
    session_service.save(data)


def _is_orchestrator_enabled(
    session: Dict[str, Any],
    model_name: Optional[str],
) -> bool:
    """Check if the orchestrator should be used for this session.

    The orchestrator is enabled when both conditions are met:
    1. Config check: session params has ``orchestrator_enabled=True`` (default True).
    2. Model capability: the active model supports native tool-calling.

    Args:
        session: Session dict with params.
        model_name: The Ollama model name for the session. If None, the
            orchestrator is disabled.

    Returns:
        True if the orchestrator should be used, False otherwise.
    """
    if not model_name:
        return False

    params = session.get("params", {})
    config_enabled = params.get("orchestrator_enabled", True)
    if not config_enabled:
        logger.debug("Orchestrator disabled by session config")
        return False

    try:
        has_tools = check_tool_call_support(model_name)
    except Exception:
        logger.warning(
            "Failed to check tool-call support for model '%s', "
            "disabling orchestrator",
            model_name,
        )
        return False

    if not has_tools:
        logger.debug(
            "Model '%s' does not support tool-calling, " "disabling orchestrator",
            model_name,
        )
        return False

    return True


async def _run_orchestrator_path(
    websocket: WebSocket,
    context: "ChatContext",
    session: Dict[str, Any],
    chat_service: Any,
    session_service: SessionService,
    needs_title: bool,
) -> None:
    """Execute the orchestrator path for a user prompt.

    Creates an OrchestratorService, streams events via the
    OrchestratorStreamTranslator, sends sources and done messages,
    saves the assistant message, and generates a title if needed.

    On orchestrator failure, falls back to the direct ChatService path.

    Args:
        websocket: The active WebSocket connection.
        context: ChatContext with prompt, modules, params, etc.
        session: Raw session dict.
        chat_service: ChatService instance (for source extraction fallback).
        session_service: SessionService for persisting messages.
        needs_title: Whether a title generation is pending.
    """
    params = context.params
    model_name = params.get("model")
    base_url = params.get("ollama_url") or get_ollama_url()
    context_window = params.get("context_window", 16384)

    # Load config for module descriptions
    config_service = get_config_service()
    config = config_service.load()

    # Load module descriptions for the orchestrator system prompt
    module_descriptions = load_module_descriptions(context.modules, config)

    # Extract custom instructions and project metadata from params
    custom_instructions = params.get("system_prompt")
    # Project metadata is the composed system_prompt for project sessions;
    # for non-project sessions custom_instructions and project_metadata
    # may be the same. The orchestrator handles deduplication internally.
    project_metadata = None
    project_id = session.get("project_id")
    if project_id and custom_instructions:
        # For project sessions, the system_prompt was composed from project
        # metadata. Pass it as project_metadata to the orchestrator.
        project_metadata = custom_instructions
        custom_instructions = None

    # Get services for orchestrator
    tool_service = get_tool_service()
    rag_service = get_rag_service()

    # Ensure RAG engine is loaded for the current modules/params if needed
    if context.modules or context.additional_index_paths:
        if rag_service.needs_reload(
            context.modules, params, context.additional_index_paths
        ):
            await websocket.send_json(
                {
                    "type": "tool_phase",
                    "tool_id": "rag",
                    "phase": "loading_models",
                    "message": "Loading models...",
                    "metadata": {},
                }
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                rag_service.load_engine,
                context.modules,
                params,
                context.additional_index_paths,
            )

    # Create OrchestratorService
    orchestrator = OrchestratorService(
        tool_service=tool_service,
        rag_service=rag_service,
        model=model_name,
        base_url=base_url,
        context_window=context_window,
        session_params=params,
        session_messages=context.session_messages,
        module_descriptions=module_descriptions,
        custom_instructions=custom_instructions,
        project_metadata=project_metadata,
    )

    # Create stream translator for event -> WebSocket message conversion
    translator = OrchestratorStreamTranslator(chat_service=chat_service)

    # Define progress emitter that sends StreamToolPhase messages via WebSocket.
    # This is captured by tool wrappers at construction time.
    async def _ws_progress_emitter(tp: ToolProgress) -> None:
        try:
            await websocket.send_json(
                {
                    "type": "tool_phase",
                    "tool_id": tp.tool_id,
                    "phase": tp.phase,
                    "message": tp.message,
                    "metadata": tp.metadata,
                }
            )
        except Exception:
            pass  # WebSocket may have closed

    # The OrchestratorService's progress_emitter is synchronous (called from
    # sync tool wrappers). We wrap the async emitter to schedule it on the
    # event loop without awaiting (fire-and-forget from the sync context).
    loop = asyncio.get_event_loop()

    def _sync_progress_emitter(tp: ToolProgress) -> None:
        try:
            loop.call_soon_threadsafe(asyncio.ensure_future, _ws_progress_emitter(tp))
        except Exception:
            pass

    # Stream orchestrator events
    async for event in orchestrator.execute(
        prompt=context.prompt,
        chat_history=context.session_messages,
        progress_emitter=_sync_progress_emitter,
    ):
        msg = translator.process_event(event)
        if msg is not None:
            try:
                await websocket.send_json(msg)
            except Exception:
                break

    # Inject RAG retrieval result for proper source extraction
    if orchestrator.last_rag_result:
        translator.set_rag_retrieval_result(orchestrator.last_rag_result)

    # Send sources message
    sources_msg = translator.build_sources_message()
    if sources_msg is not None:
        await websocket.send_json(sources_msg)

    # Send done message
    done_msg = translator.build_done_message(title_pending=needs_title)
    await websocket.send_json(done_msg)

    # Finalize to get accumulated result data for saving
    result = translator.finalize()

    # Save assistant message with tool steps
    assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": result.full_response,
    }
    if result.full_thinking:
        assistant_message["thinking"] = result.full_thinking
    if result.sources:
        assistant_message["sources"] = result.sources
    if result.metrics:
        assistant_message["metrics"] = result.metrics
    if result.confidence_level != "normal":
        assistant_message["confidence_level"] = result.confidence_level
    if result.tool_steps:
        assistant_message["tool_steps"] = result.tool_steps

    data = session_service.load()
    data = session_service.add_message(context.session_id, assistant_message, data)
    session_service.save(data)

    # Generate title if needed
    if needs_title and result.full_response:
        try:
            title = await generate_smart_title_async(result.full_response)
            fresh_data = session_service.load()
            fresh_data = session_service.update_title(
                context.session_id, title, fresh_data
            )
            session_service.save(fresh_data)
            await websocket.send_json({"type": "title", "title": title})
        except Exception:
            pass


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    session_service: SessionServiceDep,
    chat_service: ChatServiceDep,
    project_service: ProjectServiceDep,
) -> ChatResponse:
    """Non-streaming chat endpoint."""
    # 1. Validate session
    data = session_service.load()
    session = session_service.get_session(session_id, data)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # 2. Build context
    with get_document_service(session_id, "session") as pdf_service:
        context = ChatContext.from_session(
            session_id,
            body.prompt,
            session,
            pdf_service,
            project_service=project_service,
        )

    # 3. Save user message
    data = session_service.add_message(
        session_id, {"role": "user", "content": body.prompt}, data
    )
    session_service.save(data)

    # 4. Execute query via ChatService
    result = chat_service.execute(
        prompt=context.prompt,
        modules=context.modules,
        params=context.params,
        session_messages=context.session_messages,
        additional_index_paths=context.additional_index_paths,
    )

    # 5. Save assistant message
    _save_assistant_message(
        session_service, session_id, result.response, result.sources, result.metrics
    )

    # Convert source dicts to SourceNode models for response
    sources = [SourceNode(**s) for s in result.sources]

    return ChatResponse(
        content=result.response,
        sources=sources,
        confidence_level="normal",
        metrics=result.metrics,
    )


@ws_router.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    session_service=Depends(get_session_service),
    chat_service=Depends(get_chat_service),
    project_service=Depends(get_project_service),
) -> None:
    """WebSocket endpoint for streaming chat.

    Protocol:
    - Client sends: {"prompt": "user question"}
    - Server sends: {"type": "token", "content": "partial"}
    - Server sends: {"type": "sources", "data": [...]}
    - Server sends: {"type": "done", "content": "full response", "confidence_level": "normal"}
    """
    await websocket.accept()

    try:
        data = session_service.load()
        session = session_service.get_session(session_id, data)
        if session is None:
            await websocket.send_json({"type": "error", "detail": "Session not found"})
            await websocket.close(code=1008)
            return

        while True:
            # Receive message
            message = await websocket.receive_text()
            try:
                request = json.loads(message)
                prompt = request.get("prompt", "")
            except json.JSONDecodeError:
                prompt = message

            if not prompt:
                await websocket.send_json(
                    {
                        "type": "error",
                        "detail": "Empty prompt",
                    }
                )
                continue

            # Detect command anywhere in prompt
            command_match = re.search(r"/(\w+)(?:\s+(.+))?", prompt)

            if command_match:
                cmd_name = command_match.group(1)
                cmd_args = command_match.group(2) or ""

                command = command_registry.get(cmd_name)

                if command:
                    # Save user message
                    data = session_service.load()
                    data = session_service.add_message(
                        session_id, {"role": "user", "content": prompt}, data
                    )
                    session_service.save(data)

                    # Check if we need to generate a title (first message in session)
                    needs_title = session_service.needs_title_update(session_id, data)

                    # Execute command (streams response via websocket)
                    # Create a wrapper websocket that captures response and sources
                    full_response = ""
                    captured_sources = None
                    captured_tool_steps = None
                    title_pending = False

                    class ResponseCapturingWebSocket:
                        """Wrapper that captures command response, sources, and tool steps."""

                        def __init__(self, ws: WebSocket):
                            self.ws = ws

                        async def send_json(self, data: dict) -> None:
                            nonlocal full_response, captured_sources
                            nonlocal captured_tool_steps, title_pending
                            # Capture response, sources, and tool_steps from done message
                            if data.get("type") == "done":
                                if "content" in data:
                                    full_response = data["content"]
                                if "sources" in data:
                                    captured_sources = data["sources"]
                                if "tool_steps" in data:
                                    captured_tool_steps = data["tool_steps"]
                                if data.get("title_pending"):
                                    title_pending = True
                            # Forward all messages to real websocket
                            await self.ws.send_json(data)

                        # Forward other WebSocket methods to underlying ws
                        def __getattr__(self, name):
                            return getattr(self.ws, name)

                    wrapper = ResponseCapturingWebSocket(websocket)
                    # ResponseCapturingWebSocket duck-types WebSocket (safe)
                    await command.execute(cmd_args, session, wrapper)  # type: ignore

                    # Save assistant response to session (with sources if available)
                    if full_response:
                        cmd_response: dict = {
                            "role": "assistant",
                            "content": full_response,
                        }
                        if captured_sources:
                            cmd_response["sources"] = captured_sources
                        if captured_tool_steps:
                            cmd_response["tool_steps"] = captured_tool_steps

                        data = session_service.load()
                        data = session_service.add_message(
                            session_id,
                            cmd_response,
                            data,
                        )
                        session_service.save(data)

                    # Generate title for first message (same as regular chat)
                    if needs_title and full_response:
                        try:
                            title = await generate_smart_title_async(full_response)
                            fresh_data = session_service.load()
                            fresh_data = session_service.update_title(
                                session_id, title, fresh_data
                            )
                            session_service.save(fresh_data)
                            await websocket.send_json({"type": "title", "title": title})
                        except Exception:
                            # Title generation failure is non-critical
                            pass

                    # Continue to next iteration to wait for next message
                    continue
                else:
                    # Unknown command - send error
                    error_msg = (
                        f"Unknown command: /{cmd_name}. "
                        "Type /help for available commands."
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "detail": error_msg,
                        }
                    )
                    continue

            # Reload session data to get updated modules/params (may have changed mid-session)
            data = session_service.load()
            session = session_service.get_session(session_id, data)
            if session is None:
                await websocket.send_json(
                    {"type": "error", "detail": "Session not found"}
                )
                break

            # Build context (resolves project modules + index paths)
            with get_document_service(session_id, "session") as pdf_service:
                context = ChatContext.from_session(
                    session_id,
                    prompt,
                    session,
                    pdf_service,
                    project_service=project_service,
                )

            # Add user message
            data = session_service.load()
            data = session_service.add_message(
                session_id, {"role": "user", "content": prompt}, data
            )
            session_service.save(data)

            # Check if we need to generate a title (first message in session)
            needs_title = session_service.needs_title_update(session_id, data)

            # Determine model name from session params
            model_name = context.params.get("model")

            # Route through orchestrator if enabled, otherwise fallback
            use_orchestrator = _is_orchestrator_enabled(session, model_name)

            if use_orchestrator:
                # --- Orchestrator path ---
                try:
                    await _run_orchestrator_path(
                        websocket=websocket,
                        context=context,
                        session=session,
                        chat_service=chat_service,
                        session_service=session_service,
                        needs_title=needs_title,
                    )
                except Exception as orch_err:
                    logger.error(
                        "Orchestrator path failed, falling back to direct "
                        "ChatService: %s",
                        orch_err,
                        exc_info=True,
                    )
                    # Fall through to the direct ChatService path below
                    use_orchestrator = False

            if not use_orchestrator:
                # --- Direct ChatService path (fallback) ---
                # Stream response with status and thinking support
                full_response = ""
                full_thinking = ""
                sources: List[Dict[str, Any]] = []
                metrics_dict = None
                confidence_level = "normal"

                # Use ChatService for query routing (handles engine loading internally)
                query_generator = chat_service.query(
                    prompt=context.prompt,
                    modules=context.modules,
                    params=context.params,
                    session_messages=context.session_messages,
                    additional_index_paths=context.additional_index_paths,
                )

                # Iterate generator in thread pool to prevent blocking event loop
                loop = asyncio.get_event_loop()

                while True:
                    try:
                        # Get next chunk from generator in executor (non-blocking)
                        chunk = await loop.run_in_executor(None, next, query_generator)

                        if chunk.is_complete:
                            # Use ChatService to extract sources
                            sources = chat_service.extract_sources(chunk.source_nodes)
                            metrics_dict = chunk.metrics
                            confidence_level = chunk.confidence_level
                            break
                        elif chunk.status:
                            # Send legacy pipeline status update
                            await websocket.send_json(
                                {
                                    "type": "status",
                                    "status": chunk.status,
                                }
                            )
                            # Send tool_phase alongside if available
                            if chunk.progress:
                                await websocket.send_json(
                                    {
                                        "type": "tool_phase",
                                        "tool_id": chunk.progress.tool_id,
                                        "phase": chunk.progress.phase,
                                        "message": chunk.progress.message,
                                        "metadata": chunk.progress.metadata,
                                    }
                                )
                        elif chunk.thinking:
                            full_thinking += chunk.thinking
                            await websocket.send_json(
                                {
                                    "type": "thinking",
                                    "content": chunk.thinking,
                                }
                            )
                        elif chunk.text:
                            full_response += chunk.text
                            await websocket.send_json(
                                {
                                    "type": "token",
                                    "content": chunk.text,
                                }
                            )
                    except StopIteration:
                        break

                # Send sources with metrics (only in RAG mode)
                if sources:
                    await websocket.send_json(
                        {
                            "type": "sources",
                            "data": sources,
                            "metrics": metrics_dict,
                        }
                    )

                # Send completion
                await websocket.send_json(
                    {
                        "type": "done",
                        "content": full_response,
                        "confidence_level": confidence_level,
                        "title_pending": needs_title,
                    }
                )

                # Save assistant response
                assistant_message: dict = {
                    "role": "assistant",
                    "content": full_response,
                }
                if full_thinking:
                    assistant_message["thinking"] = full_thinking
                if sources:
                    assistant_message["sources"] = sources
                if metrics_dict:
                    assistant_message["metrics"] = metrics_dict
                if confidence_level != "normal":
                    assistant_message["confidence_level"] = confidence_level
                data = session_service.load()
                data = session_service.add_message(session_id, assistant_message, data)
                session_service.save(data)

                # Generate title from the response
                if needs_title:
                    try:
                        title = await generate_smart_title_async(full_response)
                        fresh_data = session_service.load()
                        fresh_data = session_service.update_title(
                            session_id, title, fresh_data
                        )
                        session_service.save(fresh_data)
                        await websocket.send_json({"type": "title", "title": title})
                    except Exception:
                        pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass


@router.post("/sessions/{session_id}/intent", response_model=IntentResponse)
async def classify_intent(
    session_id: str,
    body: IntentRequest,
    session_service: SessionServiceDep,
    intent_service: IntentServiceDep,
) -> IntentResponse:
    """Classify the intent of a message."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    result = intent_service.classify(body.message, body.recent_messages)

    return IntentResponse(
        intent=result.intent,
        query=result.query,
        reason=result.reason,
    )
