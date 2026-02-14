"""Task polling endpoints for background job status."""

from typing import Optional

from fastapi import APIRouter, HTTPException

from tensortruth.api.deps import TaskRunnerDep
from tensortruth.api.schemas import TaskListResponse, TaskResponse
from tensortruth.services.task_runner import TaskInfo, TaskStatus

router = APIRouter()


def _task_to_response(info: TaskInfo) -> TaskResponse:
    """Convert internal TaskInfo to API response schema."""
    return TaskResponse(
        task_id=info.task_id,
        task_type=info.task_type,
        status=info.status.value,
        created_at=info.created_at,
        updated_at=info.updated_at,
        progress=info.progress,
        stage=info.stage,
        result=info.result,
        error=info.error,
        metadata=info.metadata,
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, task_runner: TaskRunnerDep) -> TaskResponse:
    """Get the status of a background task."""
    info = task_runner.get_task(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_response(info)


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    task_runner: TaskRunnerDep,
    task_type: Optional[str] = None,
    status: Optional[str] = None,
) -> TaskListResponse:
    """List background tasks, optionally filtered by type and status."""
    status_enum = None
    if status is not None:
        try:
            status_enum = TaskStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    tasks = task_runner.list_tasks(task_type=task_type, status=status_enum)
    return TaskListResponse(tasks=[_task_to_response(t) for t in tasks])
