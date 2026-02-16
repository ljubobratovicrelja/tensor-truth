"""Dataclasses for service layer models.

These models are used for data transfer between services and the UI layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


@dataclass
class SessionData:
    """Immutable session state container.

    Represents the complete state of all chat sessions, designed to be
    passed between service methods without relying on global state.
    """

    current_id: Optional[str]
    sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "current_id": self.current_id,
            "sessions": self.sessions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionData":
        """Create from dictionary (loaded from JSON)."""
        return cls(
            current_id=data.get("current_id"),
            sessions=data.get("sessions", {}),
        )


@dataclass
class SessionInfo:
    """Information about a single chat session."""

    session_id: str
    title: str
    created_at: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    modules: Optional[List[str]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    title_needs_update: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "title": self.title,
            "created_at": self.created_at,
            "messages": self.messages,
            "modules": self.modules,
            "params": self.params,
            "title_needs_update": self.title_needs_update,
        }

    @classmethod
    def from_dict(cls, session_id: str, data: Dict[str, Any]) -> "SessionInfo":
        """Create from dictionary."""
        return cls(
            session_id=session_id,
            title=data.get("title", "New Session"),
            created_at=data.get("created_at", str(datetime.now())),
            messages=data.get("messages", []),
            modules=data.get("modules"),
            params=data.get("params", {}),
            title_needs_update=data.get("title_needs_update", False),
        )


@dataclass
class ProjectData:
    """Immutable project state container.

    Represents the complete state of all projects, designed to be
    passed between service methods without relying on global state.
    """

    projects: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "projects": self.projects,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectData":
        """Create from dictionary (loaded from JSON)."""
        return cls(
            projects=data.get("projects", {}),
        )


@dataclass
class IntentResult:
    """Result of intent classification.

    Used to route user messages to appropriate handlers (chat, browse, search).
    """

    intent: Literal["chat", "browse", "search"]
    query: Optional[str]
    reason: str


@dataclass
class PDFMetadata:
    """Metadata for an uploaded PDF file."""

    pdf_id: str
    filename: str
    path: str
    file_size: int
    page_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.pdf_id,
            "filename": self.filename,
            "path": self.path,
            "file_size": self.file_size,
            "page_count": self.page_count,
        }


@dataclass
class ToolProgress:
    """Progress report from a tool during execution.

    Each tool defines its own phases and messages. This replaces hardcoded
    pipeline status strings with an extensible, tool-aware progress API.

    Attributes:
        tool_id: Identifier for the tool ("rag", "web_search", "orchestrator", etc.).
        phase: Current execution phase ("retrieving", "searching", "fetching", etc.).
        message: Human-readable status message ("Searching your documents...").
        metadata: Phase-specific data (e.g., {"pages_fetched": 3, "pages_target": 5}).
    """

    tool_id: str
    phase: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGChunk:
    """A chunk of RAG response with source information.

    Attributes:
        text: Content text delta (actual response).
        thinking: Thinking/reasoning text delta (if model supports thinking).
        source_nodes: Retrieved source documents (only set when is_complete=True).
        is_complete: Whether this is the final chunk with sources.
        status: Pipeline status indicator ('loading_models', 'retrieving',
            'reranking', 'thinking', 'generating', None).
            Kept for backward compatibility; prefer using `progress`.
        progress: Structured tool progress report (new extensible API).
            When present, `status` is also set for backward compat.
        metrics: Retrieval quality metrics (only set when is_complete=True).
    """

    text: str = ""
    thinking: Optional[str] = None
    source_nodes: List[Any] = field(default_factory=list)
    is_complete: bool = False
    status: Optional[
        Literal["loading_models", "retrieving", "reranking", "thinking", "generating"]
    ] = None
    progress: Optional[ToolProgress] = None
    metrics: Optional[Dict[str, Any]] = None
    confidence_level: str = "normal"


@dataclass
class RAGRetrievalResult:
    """Result of RAG retrieval without LLM synthesis.

    Contains retrieved sources, confidence scoring, and metrics from the
    retrieval + reranking pipeline. Used by the orchestrator's rag_query tool
    to get retrieval results without generating a response.

    Attributes:
        source_nodes: Retrieved and reranked source documents (LlamaIndex NodeWithScore).
        confidence_level: Confidence assessment ("normal", "low", "none").
        metrics: Retrieval quality metrics dict (score distribution, diversity, etc.).
        condensed_query: The query after history condensation (may differ from original).
        num_sources: Number of sources returned after reranking.
    """

    source_nodes: List[Any] = field(default_factory=list)
    confidence_level: str = "normal"
    metrics: Optional[Dict[str, Any]] = None
    condensed_query: str = ""
    num_sources: int = 0


@dataclass
class RAGResponse:
    """Complete RAG response with all sources."""

    text: str
    source_nodes: List[Any] = field(default_factory=list)
    confidence_level: str = "normal"  # "normal", "low", "none"
    metrics: Optional[Dict[str, Any]] = None
