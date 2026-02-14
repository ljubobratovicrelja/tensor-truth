"""Chat endpoints including WebSocket streaming."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from tensortruth.api.deps import (
    ChatServiceDep,
    IntentServiceDep,
    SessionServiceDep,
    get_chat_service,
    get_pdf_service,
    get_session_service,
)
from tensortruth.api.routes.commands import registry as command_registry
from tensortruth.api.schemas import (
    ChatRequest,
    ChatResponse,
    IntentRequest,
    IntentResponse,
    SourceNode,
)
from tensortruth.app_utils.title_generation import generate_smart_title_async
from tensortruth.services import SessionService

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
    session_index_path: Optional[str]

    @classmethod
    def from_session(
        cls,
        session_id: str,
        prompt: str,
        session: Dict[str, Any],
        pdf_service: Any,
    ) -> "ChatContext":
        """Create ChatContext from session data and PDF service.

        Args:
            session_id: The session identifier.
            prompt: The user's current prompt.
            session: Session dict containing modules, params, messages.
            pdf_service: PDFService instance to check for PDF index.

        Returns:
            ChatContext instance with all fields populated.
        """
        index_path = pdf_service.get_index_path()
        return cls(
            session_id=session_id,
            prompt=prompt,
            modules=session.get("modules") or [],
            params=session.get("params", {}),
            session_messages=session.get("messages", []),
            session_index_path=str(index_path) if index_path else None,
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


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    session_service: SessionServiceDep,
    chat_service: ChatServiceDep,
) -> ChatResponse:
    """Non-streaming chat endpoint."""
    # 1. Validate session
    data = session_service.load()
    session = session_service.get_session(session_id, data)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # 2. Build context
    with get_pdf_service(session_id) as pdf_service:
        context = ChatContext.from_session(
            session_id, body.prompt, session, pdf_service
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
        session_index_path=context.session_index_path,
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

            modules = session.get("modules") or []
            params = session.get("params", {})

            # Check for session PDF index
            with get_pdf_service(session_id) as pdf_service:
                index_path = pdf_service.get_index_path()
                session_index_path = str(index_path) if index_path else None

            # Get session messages for history BEFORE adding the new user message
            session_messages = session.get("messages", [])

            # Add user message
            data = session_service.load()
            data = session_service.add_message(
                session_id, {"role": "user", "content": prompt}, data
            )
            session_service.save(data)

            # Check if we need to generate a title (first message in session)
            needs_title = session_service.needs_title_update(session_id, data)

            # Stream response with status and thinking support
            full_response = ""
            full_thinking = ""
            sources: List[Dict[str, Any]] = []
            metrics_dict = None

            # Use ChatService for query routing (handles engine loading internally)
            query_generator = chat_service.query(
                prompt=prompt,
                modules=modules,
                params=params,
                session_messages=session_messages,
                session_index_path=session_index_path,
            )

            # Iterate generator in thread pool to prevent blocking event loop
            loop = asyncio.get_event_loop()

            while True:
                try:
                    # Get next chunk from generator in executor (non-blocking)
                    chunk = await loop.run_in_executor(None, next, query_generator)

                    if chunk.is_complete:
                        # Use ChatService to extract sources (contains foreign type handling)
                        sources = chat_service.extract_sources(chunk.source_nodes)
                        metrics_dict = chunk.metrics
                        break
                    elif chunk.status:
                        # Send pipeline status update immediately
                        await websocket.send_json(
                            {
                                "type": "status",
                                "status": chunk.status,
                            }
                        )
                    elif chunk.thinking:
                        # Send thinking token and accumulate
                        full_thinking += chunk.thinking
                        await websocket.send_json(
                            {
                                "type": "thinking",
                                "content": chunk.thinking,
                            }
                        )
                    elif chunk.text:
                        # Send content token
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
                    "confidence_level": "normal",
                    "title_pending": needs_title,
                }
            )

            # Save assistant response (include thinking for UI display, but it won't
            # be included in LLM chat history - that's handled by ChatHistoryService)
            assistant_message: dict = {"role": "assistant", "content": full_response}
            if full_thinking:
                assistant_message["thinking"] = full_thinking
            if sources:
                assistant_message["sources"] = sources
            if metrics_dict:
                assistant_message["metrics"] = metrics_dict
            data = session_service.load()
            data = session_service.add_message(session_id, assistant_message, data)
            session_service.save(data)

            # Generate title from the response (not the prompt) for better context
            # This runs after streaming completes but doesn't block the UI
            if needs_title:
                try:
                    title = await generate_smart_title_async(full_response)
                    # Update session with new title
                    fresh_data = session_service.load()
                    fresh_data = session_service.update_title(
                        session_id, title, fresh_data
                    )
                    session_service.save(fresh_data)
                    # Send title to client
                    await websocket.send_json({"type": "title", "title": title})
                except Exception:
                    # Title generation failure is non-critical
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
