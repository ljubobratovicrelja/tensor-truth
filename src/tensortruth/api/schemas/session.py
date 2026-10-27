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
    title: Optional[str] = Field(
        default=None,
        description=(
            "Title chosen by the client (e.g. previewed while the user was "
            "typing). When set, no title is regenerated after the first turn."
        ),
    )


class TitlePreviewRequest(BaseModel):
    """Request body for previewing a session title from a drafted prompt."""

    prompt: str = Field(..., description="The prompt the user is drafting")
    model: Optional[str] = None
    provider_id: Optional[str] = None
    context_window: Optional[int] = None


class TitlePreviewResponse(BaseModel):
    """Response for a previewed session title."""

    title: str


class SessionResponse(BaseModel):
    """Response for a single session."""

    session_id: str
    title: str
    created_at: str
    modules: Optional[List[str]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    project_id: Optional[str] = None


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


class ImageRef(BaseModel):
    """Reference to a stored chat image."""

    id: str
    mimetype: str
    filename: str


class MessageResponse(BaseModel):
    """Response for a single message."""

    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None
    narration_segments: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None
    tool_steps: Optional[List[Dict[str, Any]]] = None
    confidence_level: Optional[str] = None
    images: Optional[List[Dict[str, Any]]] = None


class MessagesResponse(BaseModel):
    """Response for listing session messages."""

    messages: List[MessageResponse]


class ContextCategory(BaseModel):
    """One slice of the context window."""

    id: str
    label: str
    hint: str = ""
    chars: int = 0
    tokens: int = 0
    share: float = 0.0


class SessionContextResponse(BaseModel):
    """Per-category breakdown of the context sent to the LLM."""

    measured: bool = False
    categories: List[ContextCategory] = []
    total_chars: int = 0
    total_tokens: int = 0
    context_window: int = 0
    used_percent: float = 0.0
    remaining_tokens: Optional[int] = None
    model_name: Optional[str] = None
    messages_included: int = 0
    messages_total: int = 0
    measured_at: Optional[float] = None


class CompactRequest(BaseModel):
    """Request body for /compact."""

    keep_last: int = 6
    target_tokens: Optional[int] = None


class CompactResponse(BaseModel):
    """Result of compacting a conversation."""

    compacted: bool
    message: str = ""
    messages_before: int = 0
    messages_after: int = 0
    messages_compacted: int = 0
    chars_before: int = 0
    chars_after: int = 0
    tokens_saved: int = 0
    summary_chars: int = 0
    backup_path: Optional[str] = None
    context: Optional[SessionContextResponse] = None


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

    # Config limit: max conversation turns (1 turn = user query + assistant response)
    max_history_turns: int = 0

    model_name: Optional[str] = None
    context_length: int = 0
