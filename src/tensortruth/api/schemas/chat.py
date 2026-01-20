"""Chat-related schemas."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    prompt: str = Field(..., min_length=1)


class SourceNode(BaseModel):
    """A source reference from RAG."""

    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Response for non-streaming chat."""

    content: str
    sources: List[SourceNode] = Field(default_factory=list)
    confidence_level: str = "normal"


class IntentRequest(BaseModel):
    """Request body for intent classification."""

    message: str = Field(..., min_length=1)
    recent_messages: List[Dict[str, str]] = Field(
        default_factory=list, description="Recent conversation history"
    )


class IntentResponse(BaseModel):
    """Response for intent classification."""

    intent: Literal["chat", "browse", "search"]
    query: Optional[str] = None
    reason: str


# WebSocket message schemas
class StreamToken(BaseModel):
    """WebSocket message for streaming token."""

    type: Literal["token"] = "token"
    content: str


class StreamSources(BaseModel):
    """WebSocket message for sources."""

    type: Literal["sources"] = "sources"
    data: List[SourceNode]


class StreamDone(BaseModel):
    """WebSocket message for completion."""

    type: Literal["done"] = "done"
    content: str
    confidence_level: str = "normal"


class StreamThinking(BaseModel):
    """WebSocket message for thinking/reasoning tokens."""

    type: Literal["thinking"] = "thinking"
    content: str


class StreamStatus(BaseModel):
    """WebSocket message for pipeline status updates."""

    type: Literal["status"] = "status"
    status: Literal["retrieving", "thinking", "generating"]
