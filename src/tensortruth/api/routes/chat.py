"""Chat endpoints including WebSocket streaming."""

import json
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
from tensortruth.api.schemas import (
    ChatRequest,
    ChatResponse,
    IntentRequest,
    IntentResponse,
    SourceNode,
)

# REST endpoints (mounted under /api)
rest_router = APIRouter()
# WebSocket endpoint (mounted at root, not under /api)
ws_router = APIRouter()
# Legacy alias for backwards compatibility
router = rest_router


def _extract_sources(source_nodes: List) -> List[SourceNode]:
    """Extract source information from RAG source nodes."""
    sources = []
    for node in source_nodes:
        try:
            text = node.text if hasattr(node, "text") else str(node)
            score = node.score if hasattr(node, "score") else None
            metadata = node.metadata if hasattr(node, "metadata") else {}
            sources.append(SourceNode(text=text[:500], score=score, metadata=metadata))
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

    # Load engine if needed
    if not modules and not session_index_path:
        raise HTTPException(
            status_code=400,
            detail="Session has no modules or PDFs. Add modules or upload PDFs first.",
        )

    if rag_service.needs_reload(modules, params, session_index_path):
        chat_history = rag_service.get_chat_history()
        rag_service.load_engine(
            modules=modules,
            params=params,
            session_index_path=session_index_path,
            chat_history=chat_history,
        )

    # Add user message to session
    data = session_service.add_message(
        session_id, {"role": "user", "content": body.prompt}, data
    )
    session_service.save(data)

    # Query RAG engine (collect full response)
    full_response = ""
    sources = []
    for chunk in rag_service.query(body.prompt):
        if chunk.is_complete:
            sources = _extract_sources(chunk.source_nodes)
        else:
            full_response += chunk.text

    # Add assistant response to session
    data = session_service.load()  # Reload in case of concurrent updates
    data = session_service.add_message(
        session_id, {"role": "assistant", "content": full_response}, data
    )
    session_service.save(data)

    return ChatResponse(
        content=full_response,
        sources=sources,
        confidence_level="normal",
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

        modules = session.get("modules") or []
        params = session.get("params", {})

        # Check for session PDF index
        with get_pdf_service(session_id) as pdf_service:
            index_path = pdf_service.get_index_path()
            session_index_path = str(index_path) if index_path else None

        if not modules and not session_index_path:
            await websocket.send_json(
                {
                    "type": "error",
                    "detail": "Session has no modules or PDFs",
                }
            )
            await websocket.close(code=1008)
            return

        # Load engine if needed
        if rag_service.needs_reload(modules, params, session_index_path):
            chat_history = rag_service.get_chat_history()
            rag_service.load_engine(
                modules=modules,
                params=params,
                session_index_path=session_index_path,
                chat_history=chat_history,
            )

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

            # Add user message
            data = session_service.load()
            data = session_service.add_message(
                session_id, {"role": "user", "content": prompt}, data
            )
            session_service.save(data)

            # Stream response
            full_response = ""
            sources = []

            for chunk in rag_service.query(prompt):
                if chunk.is_complete:
                    sources = _extract_sources(chunk.source_nodes)
                else:
                    full_response += chunk.text
                    await websocket.send_json(
                        {
                            "type": "token",
                            "content": chunk.text,
                        }
                    )

            # Send sources
            if sources:
                await websocket.send_json(
                    {
                        "type": "sources",
                        "data": [s.model_dump() for s in sources],
                    }
                )

            # Send completion
            await websocket.send_json(
                {
                    "type": "done",
                    "content": full_response,
                    "confidence_level": "normal",
                }
            )

            # Save assistant response
            data = session_service.load()
            data = session_service.add_message(
                session_id, {"role": "assistant", "content": full_response}, data
            )
            session_service.save(data)

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
