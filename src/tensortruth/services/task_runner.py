"""Async task runner for long-running background operations.

Provides a singleton TaskRunner that executes sync callables in a thread pool,
with per-task progress tracking and optional on_complete callbacks.
"""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TaskInfo:
    """State of a background task."""

    task_id: str
    task_type: str
    status: TaskStatus
    created_at: str
    updated_at: str
    progress: int = 0
    stage: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskRunner:
    """Runs sync callables in a background thread with progress tracking.

    One task runs at a time (serial queue). Progress is reported via a callback
    that the callable receives. Task state is stored in memory only.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskInfo] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._pending_fns: Dict[str, tuple[Callable, Optional[Callable]]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background worker loop."""
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("TaskRunner started")

    async def stop(self) -> None:
        """Cancel the worker and shut down the executor."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        self._executor.shutdown(wait=False)
        logger.info("TaskRunner stopped")

    async def submit(
        self,
        task_type: str,
        fn: Callable,
        *,
        on_complete: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a callable for background execution.

        Args:
            task_type: Label for the kind of work (e.g. "build_module").
            fn: Sync callable receiving a ``progress_callback(stage, current, total)``.
            on_complete: Optional callback invoked after the task finishes (success or error).
                Receives the TaskInfo. May be sync or async.
            metadata: Arbitrary dict stored on the TaskInfo.

        Returns:
            The new task's ID.
        """
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        self._tasks[task_id] = info
        self._pending_fns[task_id] = (fn, on_complete)
        await self._queue.put(task_id)
        logger.info("Task %s (%s) submitted", task_id, task_type)
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Return task info or None if not found."""
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        task_type: Optional[str] = None,
        status: Optional[TaskStatus] = None,
    ) -> List[TaskInfo]:
        """Return tasks filtered by type and/or status, newest first."""
        tasks = list(self._tasks.values())
        if task_type is not None:
            tasks = [t for t in tasks if t.task_type == task_type]
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    async def _worker_loop(self) -> None:
        """Process tasks from the queue one at a time."""
        loop = asyncio.get_running_loop()
        while True:
            task_id = await self._queue.get()
            entry = self._pending_fns.pop(task_id, None)
            if entry is None:
                continue
            fn, on_complete = entry
            info = self._tasks[task_id]

            info.status = TaskStatus.RUNNING
            info.updated_at = datetime.now(timezone.utc).isoformat()

            def _make_progress_cb(t: TaskInfo) -> Callable:
                def _cb(stage: str, current: int, total: int) -> None:
                    t.stage = stage
                    t.progress = min(int(current / total * 100), 100) if total else 0
                    t.updated_at = datetime.now(timezone.utc).isoformat()

                return _cb

            progress_cb = _make_progress_cb(info)
            try:
                raw_result = await loop.run_in_executor(self._executor, fn, progress_cb)
                info.status = TaskStatus.COMPLETED
                info.progress = 100
                if isinstance(raw_result, dict):
                    info.result = raw_result
                elif raw_result is not None:
                    info.result = {"success": raw_result}
                logger.info("Task %s completed", task_id)
            except Exception as exc:
                info.status = TaskStatus.ERROR
                info.error = str(exc)
                logger.error("Task %s failed: %s", task_id, exc)
            finally:
                info.updated_at = datetime.now(timezone.utc).isoformat()

            if on_complete is not None:
                try:
                    ret = on_complete(info)
                    if asyncio.iscoroutine(ret):
                        await ret
                except Exception as exc:
                    logger.error(
                        "on_complete callback for task %s failed: %s",
                        task_id,
                        exc,
                    )
