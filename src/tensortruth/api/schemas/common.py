"""Common schemas shared across API endpoints."""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
