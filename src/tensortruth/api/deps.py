"""Dependency injection for FastAPI routes.

Provides singleton services and per-request dependencies.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from tensortruth.app_utils.paths import (
    get_indexes_dir,
    get_session_dir,
    get_sessions_file,
)
from tensortruth.services import (
    ConfigService,
    IntentService,
    PDFService,
    RAGService,
    SessionService,
)
from tensortruth.services.startup_service import StartupService


@lru_cache
def get_session_service() -> SessionService:
    """Get the singleton SessionService instance."""
    return SessionService(sessions_file=get_sessions_file())


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
def get_rag_service() -> RAGService:
    """Get the singleton RAGService instance.

    RAGService manages GPU resources and engine state. It must be a singleton
    to preserve the loaded engine across requests.
    """
    config_service = get_config_service()
    config = config_service.load()
    return RAGService(config=config, indexes_dir=get_indexes_dir())


def get_intent_service() -> IntentService:
    """Get IntentService instance."""
    config_service = get_config_service()
    return IntentService(
        ollama_url=config_service.get_ollama_url(),
        classifier_model=config_service.get_intent_classifier_model(),
    )


def get_pdf_service(session_id: str) -> PDFService:
    """Get PDFService for a specific session."""
    return PDFService(
        session_id=session_id,
        session_dir=get_session_dir(session_id),
    )


# Type aliases for FastAPI dependency injection
SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]
ConfigServiceDep = Annotated[ConfigService, Depends(get_config_service)]
StartupServiceDep = Annotated[StartupService, Depends(get_startup_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
IntentServiceDep = Annotated[IntentService, Depends(get_intent_service)]
