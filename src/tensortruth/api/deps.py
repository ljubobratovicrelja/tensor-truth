"""Dependency injection for FastAPI routes.

Provides singleton services and per-request dependencies.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from tensortruth.app_utils.paths import (
    get_indexes_dir,
    get_project_dir,
    get_projects_data_dir,
    get_session_dir,
    get_sessions_data_dir,
    get_sessions_file,
)
from tensortruth.services import (
    AgentService,
    ChatHistoryService,
    ChatService,
    ConfigService,
    DocumentService,
    IntentService,
    ProjectService,
    RAGService,
    SessionService,
    TaskRunner,
    ToolService,
)
from tensortruth.services.startup_service import StartupService


@lru_cache
def get_session_service() -> SessionService:
    """Get the singleton SessionService instance."""
    return SessionService(
        sessions_file=get_sessions_file(),  # Legacy path for migration
        sessions_dir=get_sessions_data_dir(),  # New per-session storage
    )


@lru_cache
def get_project_service() -> ProjectService:
    """Get the singleton ProjectService instance."""
    return ProjectService(projects_dir=get_projects_data_dir())


@lru_cache
def get_config_service() -> ConfigService:
    """Get the singleton ConfigService instance."""
    return ConfigService()


@lru_cache
def get_startup_service() -> StartupService:
    """Get the singleton StartupService instance."""
    config_service = get_config_service()
    return StartupService(config_service)


@lru_cache
def get_chat_history_service() -> ChatHistoryService:
    """Get the singleton ChatHistoryService instance.

    ChatHistoryService handles chat history operations (building, cleaning, formatting).
    """
    config_service = get_config_service()
    config = config_service.load()
    return ChatHistoryService(config)


@lru_cache
def get_rag_service() -> RAGService:
    """Get the singleton RAGService instance.

    RAGService manages GPU resources and engine state. It must be a singleton
    to preserve the loaded engine across requests.
    """
    config_service = get_config_service()
    config = config_service.load()
    chat_history_service = get_chat_history_service()
    return RAGService(
        config=config,
        indexes_dir=get_indexes_dir(),
        chat_history_service=chat_history_service,
    )


@lru_cache
def get_chat_service() -> ChatService:
    """Get the singleton ChatService instance.

    ChatService routes queries to appropriate backend (LLM-only or RAG).
    """
    return ChatService(rag_service=get_rag_service())


def get_intent_service() -> IntentService:
    """Get IntentService instance."""
    config_service = get_config_service()
    return IntentService(
        ollama_url=config_service.get_ollama_url(),
        classifier_model=config_service.get_intent_classifier_model(),
    )


def get_document_service(scope_id: str, scope_type: str = "session") -> DocumentService:
    """Get DocumentService for a scope (session or project).

    Args:
        scope_id: Session or project identifier.
        scope_type: Either "session" or "project".

    Returns:
        Configured DocumentService instance.
    """
    if scope_type == "project":
        return DocumentService(
            scope_id=scope_id,
            scope_dir=get_project_dir(scope_id),
            scope_type="project",
        )
    return DocumentService(
        scope_id=scope_id,
        scope_dir=get_session_dir(scope_id),
        scope_type="session",
    )


def get_pdf_service(session_id: str) -> DocumentService:
    """Get DocumentService for a session (backwards-compatible alias)."""
    return get_document_service(session_id, "session")


# ToolService singleton - loaded at startup
_tool_service: ToolService | None = None


def get_tool_service() -> ToolService:
    """Get the singleton ToolService instance.

    Note: Tools are loaded asynchronously at startup in the lifespan handler.
    """
    global _tool_service
    if _tool_service is None:
        _tool_service = ToolService()
    return _tool_service


# AgentService singleton
_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    """Get the singleton AgentService instance."""
    global _agent_service
    if _agent_service is None:
        config_service = get_config_service()
        config = config_service.load().to_dict()
        _agent_service = AgentService(
            get_tool_service(), config, config_service=config_service
        )
    return _agent_service


# Type aliases for FastAPI dependency injection
SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
ConfigServiceDep = Annotated[ConfigService, Depends(get_config_service)]
StartupServiceDep = Annotated[StartupService, Depends(get_startup_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
IntentServiceDep = Annotated[IntentService, Depends(get_intent_service)]
ChatHistoryServiceDep = Annotated[ChatHistoryService, Depends(get_chat_history_service)]
ToolServiceDep = Annotated[ToolService, Depends(get_tool_service)]
# TaskRunner singleton - needs async start()/stop() lifecycle
_task_runner: TaskRunner | None = None


def get_task_runner() -> TaskRunner:
    """Get the singleton TaskRunner instance."""
    global _task_runner
    if _task_runner is None:
        _task_runner = TaskRunner()
    return _task_runner


AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
TaskRunnerDep = Annotated[TaskRunner, Depends(get_task_runner)]
