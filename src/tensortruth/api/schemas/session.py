"""Session-related schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """Request body for creating a new session."""

    modules: Optional[List[str]] = Field(
        default=None, description="List of knowledge modules to enable"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session parameters (model, temperature, etc.)",
    )


class SessionResponse(BaseModel):
    """Response for a single session."""

    session_id: str
    title: str
    created_at: str
    modules: Optional[List[str]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0


class SessionListResponse(BaseModel):
    """Response for listing all sessions."""

    sessions: List[SessionResponse]
    current_id: Optional[str] = None


class SessionUpdate(BaseModel):
    """Request body for updating a session."""

    title: Optional[str] = None
    modules: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None


class MessageCreate(BaseModel):
    """Request body for adding a message."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class MessageResponse(BaseModel):
    """Response for a single message."""

    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class MessagesResponse(BaseModel):
    """Response for listing session messages."""

    messages: List[MessageResponse]


class SessionStatsResponse(BaseModel):
    """Response schema for session statistics."""

    # Total session history (all messages stored)
    history_messages: int
    history_chars: int
    history_tokens_estimate: int = 0

    # Compiled history (what's actually sent to LLM per config)
    compiled_history_messages: int = 0
    compiled_history_chars: int = 0
    compiled_history_tokens_estimate: int = 0
    max_history_messages: int = 0  # Config limit

    model_name: Optional[str] = None
    context_length: int = 0
