"""Unit tests for TaskRunner service."""

import asyncio

import pytest

from tensortruth.services.task_runner import TaskInfo, TaskRunner, TaskStatus


@pytest.fixture
async def runner():
    """Create, start, yield, and stop a TaskRunner."""
    r = TaskRunner()
    await r.start()
    yield r
    await r.stop()


async def wait_for_task(
    runner: TaskRunner, task_id: str, timeout: float = 5.0
) -> TaskInfo:
    """Poll until a task reaches a terminal state."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        info = runner.get_task(task_id)
        if info and info.status in (TaskStatus.COMPLETED, TaskStatus.ERROR):
            return info
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Task {task_id} did not finish within {timeout}s")


class TestTaskSubmission:
    """Tests for submit() and initial task state."""

    @pytest.mark.asyncio
    async def test_submit_returns_uuid(self, runner: TaskRunner):
        tid = await runner.submit("test", lambda cb: None)
        assert isinstance(tid, str)
        assert len(tid) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_submit_creates_pending_task(self, runner: TaskRunner):
        tid = await runner.submit("build", lambda cb: None, metadata={"key": "val"})
        info = runner.get_task(tid)
        assert info is not None
        assert info.task_type == "build"
        assert info.metadata == {"key": "val"}

    @pytest.mark.asyncio
    async def test_submit_stores_task_type(self, runner: TaskRunner):
        tid = await runner.submit("index_build", lambda cb: None)
        info = runner.get_task(tid)
        assert info.task_type == "index_build"


class TestTaskExecution:
    """Tests for task execution, progress, and callbacks."""

    @pytest.mark.asyncio
    async def test_runs_to_completion(self, runner: TaskRunner):
        def work(cb):
            cb("done", 1, 1)

        tid = await runner.submit("test", work)
        info = await wait_for_task(runner, tid)
        assert info.status == TaskStatus.COMPLETED
        assert info.progress == 100

    @pytest.mark.asyncio
    async def test_error_sets_status_and_message(self, runner: TaskRunner):
        def failing(cb):
            raise ValueError("boom")

        tid = await runner.submit("test", failing)
        info = await wait_for_task(runner, tid)
        assert info.status == TaskStatus.ERROR
        assert info.error == "boom"

    @pytest.mark.asyncio
    async def test_progress_callback_updates_stage_and_progress(
        self, runner: TaskRunner
    ):
        stages_seen = []

        def work(cb):
            cb("downloading", 1, 4)
            stages_seen.append(True)
            cb("indexing", 3, 4)
            stages_seen.append(True)
            cb("done", 4, 4)

        tid = await runner.submit("test", work)
        info = await wait_for_task(runner, tid)
        assert info.status == TaskStatus.COMPLETED
        assert info.stage == "done"
        assert info.progress == 100

    @pytest.mark.asyncio
    async def test_serial_execution_order(self, runner: TaskRunner):
        order = []

        def make_fn(label):
            def work(cb):
                order.append(label)

            return work

        await runner.submit("test", make_fn("first"))
        t2 = await runner.submit("test", make_fn("second"))
        await wait_for_task(runner, t2)
        assert order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_on_complete_called_on_success(self, runner: TaskRunner):
        called_with = []

        def on_done(info: TaskInfo):
            called_with.append(info.status)

        tid = await runner.submit("test", lambda cb: None, on_complete=on_done)
        await wait_for_task(runner, tid)
        assert called_with == [TaskStatus.COMPLETED]

    @pytest.mark.asyncio
    async def test_on_complete_called_on_error(self, runner: TaskRunner):
        called_with = []

        def on_done(info: TaskInfo):
            called_with.append(info.status)

        def failing(cb):
            raise RuntimeError("fail")

        tid = await runner.submit("test", failing, on_complete=on_done)
        await wait_for_task(runner, tid)
        assert called_with == [TaskStatus.ERROR]

    @pytest.mark.asyncio
    async def test_async_on_complete_awaited(self, runner: TaskRunner):
        called = []

        async def on_done(info: TaskInfo):
            called.append(info.task_id)

        tid = await runner.submit("test", lambda cb: None, on_complete=on_done)
        await wait_for_task(runner, tid)
        # Give the on_complete a moment to fire (it runs after status is set)
        await asyncio.sleep(0.1)
        assert tid in called

    @pytest.mark.asyncio
    async def test_dict_result_stored(self, runner: TaskRunner):
        def work(cb):
            return {"modules_added": 3}

        tid = await runner.submit("test", work)
        info = await wait_for_task(runner, tid)
        assert info.result == {"modules_added": 3}

    @pytest.mark.asyncio
    async def test_bool_result_wrapped(self, runner: TaskRunner):
        def work(cb):
            return True

        tid = await runner.submit("test", work)
        info = await wait_for_task(runner, tid)
        assert info.result == {"success": True}


class TestTaskRetrieval:
    """Tests for get_task() and list_tasks()."""

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unknown(self, runner: TaskRunner):
        assert runner.get_task("nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_returns_all(self, runner: TaskRunner):
        await runner.submit("a", lambda cb: None)
        await runner.submit("b", lambda cb: None)
        tasks = runner.list_tasks()
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_list_filter_by_type(self, runner: TaskRunner):
        await runner.submit("build", lambda cb: None)
        await runner.submit("fetch", lambda cb: None)
        assert len(runner.list_tasks(task_type="build")) == 1
        assert len(runner.list_tasks(task_type="fetch")) == 1
        assert len(runner.list_tasks(task_type="other")) == 0

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, runner: TaskRunner):
        t1 = await runner.submit("test", lambda cb: None)
        await wait_for_task(runner, t1)
        t2 = await runner.submit("test", lambda cb: None)
        await wait_for_task(runner, t2)

        completed = runner.list_tasks(status=TaskStatus.COMPLETED)
        assert len(completed) == 2


class TestLifecycle:
    """Tests for start/stop."""

    @pytest.mark.asyncio
    async def test_start_stop_without_errors(self):
        r = TaskRunner()
        await r.start()
        await r.stop()

    @pytest.mark.asyncio
    async def test_stop_with_pending_task(self):
        """Stopping while a task is queued doesn't crash."""
        r = TaskRunner()
        await r.start()

        # Submit a slow task — don't wait for it
        await r.submit("slow", lambda cb: __import__("time").sleep(0.3))
        await asyncio.sleep(0.05)
        await r.stop()  # Should not raise
