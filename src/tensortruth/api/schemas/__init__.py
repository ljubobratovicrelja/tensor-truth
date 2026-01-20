"""Pydantic schemas for API request/response models."""

from .chat import (
    ChatRequest,
    ChatResponse,
    IntentRequest,
    IntentResponse,
    SourceNode,
    StreamDone,
    StreamSources,
    StreamToken,
)
from .common import ErrorResponse, HealthResponse
from .config import (
    AgentConfigSchema,
    ConfigResponse,
    ConfigUpdateRequest,
    ModelsConfigSchema,
    OllamaConfigSchema,
    RAGConfigSchema,
    UIConfigSchema,
)
from .pdf import PDFListResponse, PDFMetadataResponse, ReindexResponse
from .session import (
    MessageCreate,
    MessageResponse,
    MessagesResponse,
    SessionCreate,
    SessionListResponse,
    SessionResponse,
    SessionUpdate,
)

__all__ = [
    # Common
    "ErrorResponse",
    "HealthResponse",
    # Session
    "SessionCreate",
    "SessionResponse",
    "SessionListResponse",
    "SessionUpdate",
    "MessageCreate",
    "MessageResponse",
    "MessagesResponse",
    # Config
    "ConfigResponse",
    "ConfigUpdateRequest",
    "OllamaConfigSchema",
    "UIConfigSchema",
    "RAGConfigSchema",
    "ModelsConfigSchema",
    "AgentConfigSchema",
    # Chat
    "ChatRequest",
    "ChatResponse",
    "IntentRequest",
    "IntentResponse",
    "SourceNode",
    "StreamToken",
    "StreamSources",
    "StreamDone",
    # PDF
    "PDFMetadataResponse",
    "PDFListResponse",
    "ReindexResponse",
]
