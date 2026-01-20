"""FastAPI application factory and entry point."""

import argparse
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tensortruth import __version__
from tensortruth.api.routes import chat, config, modules, pdfs, sessions
from tensortruth.api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    yield
    # Shutdown - cleanup resources
    from tensortruth.api.deps import get_config_service, get_session_service

    # Clear LRU caches on shutdown
    get_session_service.cache_clear()
    get_config_service.cache_clear()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="TensorTruth API",
        description="REST + WebSocket API for TensorTruth RAG pipeline",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware - allow all origins in development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers under /api prefix
    app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
    app.include_router(chat.rest_router, prefix="/api", tags=["chat"])
    app.include_router(config.router, prefix="/api/config", tags=["config"])
    app.include_router(modules.router, prefix="/api", tags=["modules"])
    app.include_router(pdfs.router, prefix="/api", tags=["pdfs"])

    # WebSocket router at /ws (not under /api)
    app.include_router(chat.ws_router, tags=["websocket"])

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="ok", version=__version__)

    return app


# Create default app instance
app = create_app()


def run() -> None:
    """Run the API server (CLI entry point)."""
    parser = argparse.ArgumentParser(description="TensorTruth API Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    uvicorn.run(
        "tensortruth.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    run()
