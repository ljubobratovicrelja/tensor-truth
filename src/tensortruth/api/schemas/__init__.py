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
    HistoryCleaningConfigSchema,
    ModelsConfigSchema,
    OllamaConfigSchema,
    RAGConfigSchema,
    UIConfigSchema,
    WebSearchConfigSchema,
)
from .document import (
    ArxivLookupResponse,
    ArxivUploadRequest,
    CatalogModuleAddRequest,
    CatalogModuleAddResponse,
    CatalogModuleRemoveResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
    FileUrlInfoResponse,
    FileUrlUploadRequest,
    IndexingConfig,
    IndexingConfigUpdate,
    TextUploadRequest,
    UrlUploadRequest,
)
from .pdf import (
    PDFListResponse,
    PDFMetadataResponse,
    ReindexResponse,
    ReindexTaskResponse,
)
from .project import (
    CatalogModuleStatus,
    DocumentInfo,
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectSessionCreate,
    ProjectUpdate,
)
from .session import (
    MessageCreate,
    MessageResponse,
    MessagesResponse,
    SessionCreate,
    SessionListResponse,
    SessionResponse,
    SessionStatsResponse,
    SessionUpdate,
)
from .startup import (
    IndexDownloadRequest,
    IndexDownloadResponse,
    IndexesStatusSchema,
    ModelPullRequest,
    ModelPullResponse,
    ModelsStatusSchema,
    ReinitializeIndexesResponse,
    StartupStatusResponse,
)
from .task import TaskListResponse, TaskResponse, TaskSubmitResponse

__all__ = [
    # Common
    "ErrorResponse",
    "HealthResponse",
    # Session
    "SessionCreate",
    "SessionResponse",
    "SessionListResponse",
    "SessionStatsResponse",
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
    "HistoryCleaningConfigSchema",
    "WebSearchConfigSchema",
    # Chat
    "ChatRequest",
    "ChatResponse",
    "IntentRequest",
    "IntentResponse",
    "SourceNode",
    "StreamToken",
    "StreamSources",
    "StreamDone",
    # Project
    "CatalogModuleStatus",
    "DocumentInfo",
    "ProjectCreate",
    "ProjectUpdate",
    "ProjectResponse",
    "ProjectListResponse",
    "ProjectSessionCreate",
    # PDF
    "PDFMetadataResponse",
    "PDFListResponse",
    "ReindexResponse",
    "ReindexTaskResponse",
    # Document
    "ArxivLookupResponse",
    "ArxivUploadRequest",
    "DocumentUploadResponse",
    "DocumentListItem",
    "DocumentListResponse",
    "UrlUploadRequest",
    "TextUploadRequest",
    "CatalogModuleAddRequest",
    "CatalogModuleAddResponse",
    "CatalogModuleRemoveResponse",
    "FileUrlInfoResponse",
    "FileUrlUploadRequest",
    "IndexingConfig",
    "IndexingConfigUpdate",
    # Startup
    "StartupStatusResponse",
    "IndexesStatusSchema",
    "ModelsStatusSchema",
    "IndexDownloadRequest",
    "IndexDownloadResponse",
    "ModelPullRequest",
    "ModelPullResponse",
    "ReinitializeIndexesResponse",
    # Task
    "TaskResponse",
    "TaskListResponse",
    "TaskSubmitResponse",
]
