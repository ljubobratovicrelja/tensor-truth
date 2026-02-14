"""Integration tests for tasks API endpoints."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

import tensortruth.api.deps as deps_module
from tensortruth.api.main import create_app
from tensortruth.services.task_runner import TaskRunner, TaskStatus


@pytest.fixture(autouse=True)
def _reset_task_runner_singleton():
    """Reset the global TaskRunner singleton between tests."""
    deps_module._task_runner = None
    yield
    deps_module._task_runner = None


@pytest.fixture
def app():
    """Create test application."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
async def runner():
    """Get a fresh TaskRunner, start it, yield, stop it."""
    r = deps_module.get_task_runner()
    await r.start()
    yield r
    await r.stop()


async def _wait_for_task(runner: TaskRunner, task_id: str, timeout: float = 5.0):
    """Poll until a task reaches a terminal state."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        info = runner.get_task(task_id)
        if info and info.status in (TaskStatus.COMPLETED, TaskStatus.ERROR):
            return info
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Task {task_id} did not finish within {timeout}s")


class TestGetTask:
    """Tests for GET /api/tasks/{task_id}."""

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, client, runner):
        response = await client.get("/api/tasks/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_task_completed(self, client, runner):
        tid = await runner.submit("test_type", lambda cb: cb("done", 1, 1))
        await _wait_for_task(runner, tid)

        response = await client.get(f"/api/tasks/{tid}")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == tid
        assert data["task_type"] == "test_type"
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert data["error"] is None
        assert "created_at" in data
        assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_get_task_error(self, client, runner):
        def failing(cb):
            raise RuntimeError("something broke")

        tid = await runner.submit("test_type", failing)
        await _wait_for_task(runner, tid)

        response = await client.get(f"/api/tasks/{tid}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["error"] == "something broke"


class TestListTasks:
    """Tests for GET /api/tasks."""

    @pytest.mark.asyncio
    async def test_list_tasks_empty(self, client, runner):
        response = await client.get("/api/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["tasks"] == []

    @pytest.mark.asyncio
    async def test_list_tasks_with_results(self, client, runner):
        t1 = await runner.submit("build", lambda cb: None)
        t2 = await runner.submit("fetch", lambda cb: None)
        await _wait_for_task(runner, t1)
        await _wait_for_task(runner, t2)

        response = await client.get("/api/tasks")
        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 2

        # Filter by type
        response = await client.get("/api/tasks?task_type=build")
        assert len(response.json()["tasks"]) == 1

        # Filter by status
        response = await client.get("/api/tasks?status=completed")
        assert len(response.json()["tasks"]) == 2
