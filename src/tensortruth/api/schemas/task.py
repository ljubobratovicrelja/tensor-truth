"""Task-related schemas for background job tracking."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TaskResponse(BaseModel):
    """Response for a single background task."""

    task_id: str
    task_type: str
    status: Literal["pending", "running", "completed", "error"]
    created_at: str
    updated_at: str
    progress: int = Field(ge=0, le=100)
    stage: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskListResponse(BaseModel):
    """Response for listing background tasks."""

    tasks: List[TaskResponse]


class TaskSubmitResponse(BaseModel):
    """Response returned when a task is submitted (used by Story 6+)."""

    task_id: str
