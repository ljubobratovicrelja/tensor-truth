"""Chat endpoints including WebSocket streaming."""

import asyncio
import json
import re
from typing import List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from tensortruth.api.deps import (
    ConfigServiceDep,
    IntentServiceDep,
    RAGServiceDep,
    SessionServiceDep,
    get_config_service,
    get_pdf_service,
    get_rag_service,
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

# REST endpoints (mounted under /api)
rest_router = APIRouter()
# WebSocket endpoint (mounted at root, not under /api)
ws_router = APIRouter()
# Legacy alias for backwards compatibility
router = rest_router


def _extract_sources(source_nodes: List) -> List[SourceNode]:
    """Extract source information from RAG source nodes.

    Uses get_content() to get the full merged content from AutoMergingRetriever,
    which may be larger than the leaf node text if nodes were merged to parent.
    """
    sources = []
    for node in source_nodes:
        try:
            # For NodeWithScore, access the inner node for get_content()
            inner_node = getattr(node, "node", node)

            # Prefer get_content() which returns merged parent content
            # when AutoMergingRetriever has merged leaf nodes
            if hasattr(inner_node, "get_content"):
                text = inner_node.get_content()
            elif hasattr(node, "text"):
                text = node.text
            else:
                text = str(node)

            score = node.score if hasattr(node, "score") else None
            metadata = node.metadata if hasattr(node, "metadata") else {}
            sources.append(SourceNode(text=text, score=score, metadata=metadata))
        except Exception:
            continue
    return sources


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    session_service: SessionServiceDep,
    config_service: ConfigServiceDep,
    rag_service: RAGServiceDep,
) -> ChatResponse:
    """Non-streaming chat endpoint."""
    data = session_service.load()
    session = session_service.get_session(session_id, data)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    modules = session.get("modules") or []
    params = session.get("params", {})

    # Check for session PDF index
    with get_pdf_service(session_id) as pdf_service:
        index_path = pdf_service.get_index_path()
        session_index_path = str(index_path) if index_path else None

    # Determine if we're in LLM-only mode (no modules or PDFs)
    llm_only_mode = not modules and not session_index_path

    # Get session messages for history BEFORE adding the new user message
    session_messages = session.get("messages", [])

    # Add user message to session
    data = session_service.add_message(
        session_id, {"role": "user", "content": body.prompt}, data
    )
    session_service.save(data)

    # Query engine (collect full response)
    full_response = ""
    sources: List[SourceNode] = []
    metrics_dict = None

    if llm_only_mode:
        # LLM-only mode: direct LLM query without RAG
        for chunk in rag_service.query_llm_only(
            body.prompt, params, session_messages=session_messages
        ):
            if chunk.is_complete:
                sources = []
                metrics_dict = chunk.metrics
            elif chunk.text:
                full_response += chunk.text
    else:
        # RAG mode: load engine if needed and query with retrieval
        if rag_service.needs_reload(modules, params, session_index_path):
            rag_service.load_engine(
                modules=modules,
                params=params,
                session_index_path=session_index_path,
            )

        for chunk in rag_service.query(body.prompt, session_messages=session_messages):
            if chunk.is_complete:
                sources = _extract_sources(chunk.source_nodes)
                metrics_dict = chunk.metrics
            elif chunk.text:
                full_response += chunk.text

    # Add assistant response to session
    assistant_message: dict = {"role": "assistant", "content": full_response}
    if sources:
        assistant_message["sources"] = [s.model_dump() for s in sources]
    if metrics_dict:
        assistant_message["metrics"] = metrics_dict
    data = session_service.load()  # Reload in case of concurrent updates
    data = session_service.add_message(session_id, assistant_message, data)
    session_service.save(data)

    return ChatResponse(
        content=full_response,
        sources=sources,
        confidence_level="llm_only" if llm_only_mode else "normal",
        metrics=metrics_dict,
    )


@ws_router.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    session_service=Depends(get_session_service),
    config_service=Depends(get_config_service),
    rag_service=Depends(get_rag_service),
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
                    title_pending = False

                    class ResponseCapturingWebSocket:
                        """Wrapper that captures command response and sources."""

                        def __init__(self, ws: WebSocket):
                            self.ws = ws

                        async def send_json(self, data: dict) -> None:
                            nonlocal full_response, captured_sources, title_pending
                            # Capture response and sources from done message
                            if data.get("type") == "done":
                                if "content" in data:
                                    full_response = data["content"]
                                if "sources" in data:
                                    captured_sources = data["sources"]
                                if data.get("title_pending"):
                                    title_pending = True
                            # Forward all messages to real websocket
                            await self.ws.send_json(data)

                        # Forward other WebSocket methods to underlying ws
                        def __getattr__(self, name):
                            return getattr(self.ws, name)

                    wrapper = ResponseCapturingWebSocket(websocket)
                    # Type ignore because wrapper is duck-typed compatible
                    await command.execute(cmd_args, session, wrapper)  # type: ignore

                    # Save assistant response to session (with sources if available)
                    if full_response:
                        assistant_message: dict = {
                            "role": "assistant",
                            "content": full_response,
                        }
                        if captured_sources:
                            assistant_message["sources"] = captured_sources

                        data = session_service.load()
                        data = session_service.add_message(
                            session_id,
                            assistant_message,
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

            # Determine if we're in LLM-only mode (no modules or PDFs)
            llm_only_mode = not modules and not session_index_path

            # Get session messages for history BEFORE adding the new user message
            session_messages = session.get("messages", [])

            # Load RAG engine if needed (send status to keep connection alive)
            if not llm_only_mode and rag_service.needs_reload(
                modules, params, session_index_path
            ):
                # Send loading status to prevent WebSocket timeout
                await websocket.send_json(
                    {
                        "type": "status",
                        "status": "loading_models",
                    }
                )

                # Run blocking load_engine in thread pool to keep event loop responsive
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    rag_service.load_engine,
                    modules,
                    params,
                    session_index_path,
                )

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
            sources = []
            metrics_dict = None

            # Choose query method based on mode
            # Pass session_messages for history context
            if llm_only_mode:
                query_generator = rag_service.query_llm_only(
                    prompt, params, session_messages=session_messages
                )
            else:
                query_generator = rag_service.query(
                    prompt, session_messages=session_messages
                )

            # Iterate generator in thread pool to prevent blocking event loop
            loop = asyncio.get_event_loop()

            while True:
                try:
                    # Get next chunk from generator in executor (non-blocking)
                    chunk = await loop.run_in_executor(None, next, query_generator)

                    if chunk.is_complete:
                        sources = _extract_sources(chunk.source_nodes)
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
                        "data": [s.model_dump() for s in sources],
                        "metrics": metrics_dict,
                    }
                )

            # Send completion
            await websocket.send_json(
                {
                    "type": "done",
                    "content": full_response,
                    "confidence_level": "llm_only" if llm_only_mode else "normal",
                    "title_pending": needs_title,
                }
            )

            # Save assistant response (include thinking for UI display, but it won't
            # be included in LLM chat history - that's handled by ChatHistoryService)
            assistant_message: dict = {"role": "assistant", "content": full_response}
            if full_thinking:
                assistant_message["thinking"] = full_thinking
            if sources:
                assistant_message["sources"] = [s.model_dump() for s in sources]
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
