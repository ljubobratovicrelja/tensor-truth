"""Session management endpoints."""

from fastapi import APIRouter, HTTPException

from tensortruth.api.deps import SessionServiceDep
from tensortruth.api.schemas import (
    MessageCreate,
    MessageResponse,
    MessagesResponse,
    SessionCreate,
    SessionListResponse,
    SessionResponse,
    SessionUpdate,
)
from tensortruth.app_utils.paths import get_session_dir

router = APIRouter()


def _session_to_response(session_id: str, session: dict) -> SessionResponse:
    """Convert internal session dict to response schema."""
    return SessionResponse(
        session_id=session_id,
        title=session.get("title", "New Session"),
        created_at=session.get("created_at", ""),
        modules=session.get("modules"),
        params=session.get("params", {}),
        message_count=len(session.get("messages", [])),
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions(session_service: SessionServiceDep) -> SessionListResponse:
    """List all chat sessions."""
    data = session_service.load()
    sessions = [_session_to_response(sid, sess) for sid, sess in data.sessions.items()]
    # Sort by created_at descending (newest first)
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    return SessionListResponse(sessions=sessions, current_id=data.current_id)


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreate, session_service: SessionServiceDep
) -> SessionResponse:
    """Create a new chat session."""
    data = session_service.load()
    new_id, new_data = session_service.create(
        modules=body.modules, params=body.params, data=data
    )
    session_service.save(new_data)
    session = new_data.sessions[new_id]
    return _session_to_response(new_id, session)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str, session_service: SessionServiceDep
) -> SessionResponse:
    """Get a session by ID."""
    data = session_service.load()
    session = session_service.get_session(session_id, data)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_to_response(session_id, session)


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str, body: SessionUpdate, session_service: SessionServiceDep
) -> SessionResponse:
    """Update a session (title, params)."""
    data = session_service.load()
    session = session_service.get_session(session_id, data)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Update title if provided
    if body.title is not None:
        data = session_service.update_title(session_id, body.title, data)

    # Update params if provided
    if body.params is not None:
        new_sessions = dict(data.sessions)
        new_sessions[session_id] = dict(new_sessions[session_id])
        new_sessions[session_id]["params"] = body.params
        from tensortruth.services import SessionData

        data = SessionData(current_id=data.current_id, sessions=new_sessions)

    session_service.save(data)
    return _session_to_response(session_id, data.sessions[session_id])


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, session_service: SessionServiceDep) -> None:
    """Delete a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = get_session_dir(session_id)
    new_data = session_service.delete(session_id, data, session_dir=session_dir)
    session_service.save(new_data)


@router.get("/{session_id}/messages", response_model=MessagesResponse)
async def get_messages(
    session_id: str, session_service: SessionServiceDep
) -> MessagesResponse:
    """Get all messages for a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = session_service.get_messages(session_id, data)
    return MessagesResponse(
        messages=[
            MessageResponse(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                sources=msg.get("sources"),
                thinking=msg.get("thinking"),
            )
            for msg in messages
        ]
    )


@router.post("/{session_id}/messages", response_model=MessageResponse, status_code=201)
async def add_message(
    session_id: str, body: MessageCreate, session_service: SessionServiceDep
) -> MessageResponse:
    """Add a message to a session."""
    data = session_service.load()
    if session_id not in data.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    message = {"role": body.role, "content": body.content}
    new_data = session_service.add_message(session_id, message, data)
    session_service.save(new_data)

    return MessageResponse(role=body.role, content=body.content)
