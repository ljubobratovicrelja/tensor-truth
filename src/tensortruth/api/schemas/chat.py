"""Chat-related schemas."""

from enum import Enum
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
    metrics: Optional[Dict[str, Any]] = None


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
    metrics: Optional[Dict[str, Any]] = None
    source_types: Optional[List[str]] = None
    rag_count: Optional[int] = None
    web_count: Optional[int] = None


class StreamDone(BaseModel):
    """WebSocket message for completion."""

    type: Literal["done"] = "done"
    content: str
    confidence_level: str = "normal"


class StreamThinking(BaseModel):
    """WebSocket message for thinking/reasoning tokens."""

    type: Literal["thinking"] = "thinking"
    content: str


class StreamReasoning(BaseModel):
    """WebSocket message for agent intermediate reasoning deltas.

    Emitted between tool calls when the FunctionAgent's LLM is reasoning
    about results. Ephemeral — not accumulated or stored.
    """

    type: Literal["reasoning"] = "reasoning"
    content: str


class StreamStatus(BaseModel):
    """WebSocket message for pipeline status updates."""

    type: Literal["status"] = "status"
    status: Literal["retrieving", "thinking", "generating"]


class StreamToolProgress(BaseModel):
    """WebSocket message for tool execution progress."""

    type: Literal["tool_progress"] = "tool_progress"
    tool: str
    action: Literal["calling", "completed", "failed"]
    params: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[str] = None
    is_error: Optional[bool] = None


class StreamToolPhase(BaseModel):
    """WebSocket message for real-time tool phase progress.

    Separate from StreamToolProgress which tracks tool execution steps
    (calling/completed/failed). This reports phase-level progress within
    a tool's execution (e.g., "Searching knowledge base...", "Ranking results...").
    Each tool defines its own phases and messages.
    """

    type: Literal["tool_phase"] = "tool_phase"
    tool_id: str
    phase: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentPhase(str, Enum):
    """Phases of agent execution."""

    SEARCHING = "searching"
    FETCHING = "fetching"
    SUMMARIZING = "summarizing"
    COMPLETE = "complete"


class StreamAgentProgress(BaseModel):
    """WebSocket message for agent execution progress."""

    type: Literal["agent_progress"] = "agent_progress"
    agent: str  # "web_search", future agents
    phase: str  # AgentPhase value
    message: str  # Human-readable status (no emoji)

    # Phase-specific data (all optional)
    search_query: Optional[str] = None
    search_hits: Optional[int] = None
    pages_target: Optional[int] = None
    pages_fetched: Optional[int] = None
    pages_failed: Optional[int] = None
    current_page: Optional[Dict[str, Any]] = None  # {url, title, status, error}
    model_name: Optional[str] = None


class WebSearchSource(BaseModel):
    """A single web search source result."""

    url: str
    title: str
    status: Literal["success", "failed", "skipped"]
    error: Optional[str] = None
    snippet: Optional[str] = None


class StreamWebSearchSources(BaseModel):
    """WebSocket message for web search sources."""

    type: Literal["web_sources"] = "web_sources"
    sources: List[WebSearchSource]
