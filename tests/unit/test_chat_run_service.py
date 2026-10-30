"""Unit tests for the server-owned chat run registry."""

import asyncio

import pytest

from tensortruth.services.chat_run_service import (
    ChatRun,
    ChatRunService,
    RunConflictError,
)


async def _wait_task(run: ChatRun, timeout: float = 2.0) -> None:
    """Await a run's generation task."""
    assert run.task is not None
    await asyncio.wait_for(asyncio.shield(run.task), timeout=timeout)


async def _collect(service: ChatRunService, run: ChatRun, after: int = 0):
    return [event async for event in service.subscribe(run, after)]


class TestSequencing:
    """Event log sequencing, cursor replay and capping."""

    def test_seq_starts_at_one_and_is_monotonic(self):
        run = ChatRun(session_id="s", prompt="p")
        assert run.publish({"type": "token", "content": "a"}) == 1
        assert run.publish({"type": "token", "content": "b"}) == 2
        assert run.publish({"type": "done", "content": "ab"}) == 3
        assert [e["seq"] for e in run.events] == [1, 2, 3]
        assert run.publish({"type": "x", "seq": 99}) == 4

    def test_snapshot_from_cursor(self):
        run = ChatRun(session_id="s", prompt="p")
        for i in range(5):
            run.publish({"type": "token", "content": str(i)})

        assert [e["seq"] for e in run.snapshot(0)] == [1, 2, 3, 4, 5]
        assert [e["seq"] for e in run.snapshot(2)] == [3, 4, 5]
        assert run.snapshot(5) == []
        assert run.snapshot(100) == []

    def test_event_cap_drops_oldest_but_seq_stays_monotonic(self):
        run = ChatRun(session_id="s", prompt="p", max_events=3)
        for i in range(5):
            run.publish({"type": "token", "content": str(i)})

        assert [e["seq"] for e in run.events] == [3, 4, 5]
        assert [e["seq"] for e in run.snapshot(0)] == [3, 4, 5]
        assert [e["seq"] for e in run.snapshot(2)] == [3, 4, 5]
        assert run.seq == 5
        assert run.publish({"type": "token", "content": "x"}) == 6


class TestSubscription:
    """Replay + live tail behaviour."""

    @pytest.mark.asyncio
    async def test_replay_then_live(self):
        service = ChatRunService()
        release = asyncio.Event()

        async def runner(sink):
            await sink.send_json({"type": "token", "content": "a"})
            await release.wait()
            await sink.send_json({"type": "done", "content": "a"})

        run = await service.create("sess", "hi", runner=runner)
        events = []

        async def consume():
            async for event in service.subscribe(run, 0):
                events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert [e["seq"] for e in events] == [1]

        release.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert [e["seq"] for e in events] == [1, 2]
        assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_terminal_event_not_dropped_when_run_closes_during_stream(self):
        service = ChatRunService()

        async def runner(sink):
            for i in range(3):
                await sink.send_json({"type": "token", "content": str(i)})
            await sink.send_json({"type": "done", "content": "012"})

        run = await service.create("sess", "hi", runner=runner)
        events = []
        async for event in service.subscribe(run, 0):
            events.append(event)

        assert [e["seq"] for e in events] == [1, 2, 3, 4]
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_attach_from_cursor_skips_applied_events(self):
        service = ChatRunService()

        async def runner(sink):
            for i in range(4):
                await sink.send_json({"type": "token", "content": str(i)})
            await sink.send_json({"type": "done", "content": "0123"})

        run = await service.create("sess", "hi", runner=runner)
        await _wait_task(run)

        events = await _collect(service, run, after=2)
        assert [e["seq"] for e in events] == [3, 4, 5]

    @pytest.mark.asyncio
    async def test_subscriber_disconnect_does_not_stop_producer(self):
        service = ChatRunService()
        release = asyncio.Event()

        async def runner(sink):
            await sink.send_json({"type": "token", "content": "first"})
            await release.wait()
            await sink.send_json({"type": "token", "content": "second"})
            await sink.send_json({"type": "done", "content": "firstsecond"})

        run = await service.create("sess", "hi", runner=runner)

        generator = service.subscribe(run, 0)
        first = await generator.__anext__()
        assert first["content"] == "first"
        # Reader goes away mid-run.
        await generator.aclose()
        assert run._subscribers == set()

        release.set()
        await _wait_task(run)

        assert run.status == "completed"
        assert [e["seq"] for e in run.events] == [1, 2, 3]
        assert run.events[-1] == {"type": "done", "content": "firstsecond", "seq": 3}

    @pytest.mark.asyncio
    async def test_late_reattach_replays_full_log(self):
        service = ChatRunService()

        async def runner(sink):
            for i in range(3):
                await sink.send_json({"type": "token", "content": str(i)})
            await sink.send_json({"type": "done", "content": "012"})

        run = await service.create("sess", "hi", runner=runner)
        await _wait_task(run)
        assert run.status == "completed"

        events = await _collect(service, run, after=0)
        assert [e["seq"] for e in events] == [1, 2, 3, 4]
        assert events[-1]["type"] == "done"


class TestLifecycle:
    """Status, cancellation, conflict and retention rules."""

    @pytest.mark.asyncio
    async def test_terminal_completed_status(self):
        service = ChatRunService()

        async def runner(sink):
            await sink.send_json({"type": "done", "content": "ok"})

        run = await service.create("sess", "hi", runner=runner)
        await _wait_task(run)
        assert run.status == "completed"
        assert run.error is None
        assert run.finished_at is not None
        assert run.is_terminal

    @pytest.mark.asyncio
    async def test_failed_runner_publishes_error_and_marks_failed(self):
        service = ChatRunService()

        async def runner(sink):
            await sink.send_json({"type": "token", "content": "partial"})
            raise RuntimeError("boom")

        run = await service.create("sess", "hi", runner=runner)
        await _wait_task(run)

        assert run.status == "failed"
        assert run.error == "boom"
        assert run.events[-1]["type"] == "error"
        assert "boom" in run.events[-1]["detail"]

    @pytest.mark.asyncio
    async def test_cancel_sets_status_and_stops_task(self):
        service = ChatRunService()

        async def runner(sink):
            await sink.send_json({"type": "token", "content": "a"})
            await asyncio.sleep(30)

        run = await service.create("sess", "hi", runner=runner)
        await asyncio.sleep(0)
        assert run.status == "running"

        cancelled = await service.cancel(run.run_id)
        assert cancelled is run
        assert run.status == "cancelled"
        assert run.is_terminal
        assert run.task.done()
        assert run.events[-1]["type"] == "error"

    @pytest.mark.asyncio
    async def test_cancel_finished_run_is_noop(self):
        service = ChatRunService()

        async def runner(sink):
            await sink.send_json({"type": "done", "content": "ok"})

        run = await service.create("sess", "hi", runner=runner)
        await _wait_task(run)

        again = await service.cancel(run.run_id)
        assert again is run
        assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_one_active_run_per_session(self):
        service = ChatRunService()

        async def runner(sink):
            await asyncio.sleep(30)

        first = await service.create("sess", "one", runner=runner)
        with pytest.raises(RunConflictError) as excinfo:
            await service.create("sess", "two", runner=runner)
        assert excinfo.value.run_id == first.run_id

        # A different session is unaffected.
        other = await service.create("other", "one", runner=runner)
        assert other.run_id != first.run_id

        await service.cancel(first.run_id)
        await service.cancel(other.run_id)
        # Once terminal, the session can start a new run.
        again = await service.create("sess", "three", runner=runner)
        assert again.session_id == "sess"
        await service.cancel(again.run_id)

    @pytest.mark.asyncio
    async def test_active_for_session_returns_latest_running(self):
        service = ChatRunService()

        async def runner(sink):
            await asyncio.sleep(30)

        run = await service.create("sess", "hi", runner=runner)
        assert service.active_for_session("sess") is run
        assert service.active_for_session("missing") is None
        await service.cancel(run.run_id)
        assert service.active_for_session("sess") is None

    @pytest.mark.asyncio
    async def test_finished_runs_are_gc_after_retention(self):
        service = ChatRunService(retention_seconds=0.05)

        async def runner(sink):
            await sink.send_json({"type": "done", "content": "ok"})

        run = await service.create("sess", "hi", runner=runner)
        await _wait_task(run)
        run_id = run.run_id
        assert service.get(run_id) is run

        await asyncio.sleep(0.1)
        assert service.get(run_id) is None


class TestProducerDeath:
    """A run must never outlive its producer as a phantom "running" record."""

    async def _settle(self) -> None:
        """Let task done callbacks scheduled by the event loop run."""
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_cancelled_task_publishes_terminal_event_and_releases_lock(self):
        service = ChatRunService()
        started = asyncio.Event()

        async def runner(sink):
            await sink.send_json({"type": "token", "content": "partial"})
            started.set()
            await asyncio.sleep(30)

        run = await service.create("sess", "hi", runner=runner)
        await started.wait()
        assert run.status == "running"

        assert run.task is not None
        run.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run.task
        await self._settle()

        assert run.status == "cancelled"
        assert run.is_terminal
        assert run.events[-1]["type"] == "error"
        assert service.active_for_session("sess") is None

        async def quick(sink):
            await sink.send_json({"type": "done", "content": "ok"})

        again = await service.create("sess", "next", runner=quick)
        await _wait_task(again)
        assert again.status == "completed"

    @pytest.mark.asyncio
    async def test_task_cancelled_before_it_starts_still_finalises(self):
        # Cancelled before its first step, ``_execute`` never runs, so only the
        # done callback can release the lock.
        service = ChatRunService()

        async def runner(sink):
            await asyncio.sleep(30)

        run = await service.create("sess", "hi", runner=runner)
        assert run.task is not None
        run.task.cancel()
        await asyncio.gather(run.task, return_exceptions=True)
        await self._settle()

        assert run.status == "cancelled"
        assert run.events[-1]["type"] == "error"
        assert service.active_for_session("sess") is None

    @pytest.mark.asyncio
    async def test_producer_exception_publishes_error_and_releases_lock(self):
        service = ChatRunService()

        async def runner(sink):
            await sink.send_json({"type": "token", "content": "partial"})
            raise RuntimeError("producer died")

        run = await service.create("sess", "hi", runner=runner)
        await _wait_task(run)
        await self._settle()

        assert run.status == "failed"
        assert run.error == "producer died"
        assert run.events[-1]["type"] == "error"
        assert "producer died" in run.events[-1]["detail"]
        assert service.active_for_session("sess") is None

        async def quick(sink):
            await sink.send_json({"type": "done", "content": "ok"})

        again = await service.create("sess", "next", runner=quick)
        await _wait_task(again)
        assert again.status == "completed"


class TestStaleRunSweep:
    """Runs whose task object is gone must not wedge their session forever."""

    @staticmethod
    def _orphan(service: ChatRunService, session_id: str, idle: float) -> ChatRun:
        """Register a run with no task, as left behind by a lost producer."""
        run = ChatRun(session_id=session_id, prompt="orphan")
        run.publish({"type": "token", "content": "half"})
        run.last_event_at -= idle
        # Simulates the registry record surviving without its asyncio task.
        service._runs[run.run_id] = run
        return run

    @pytest.mark.asyncio
    async def test_stale_run_without_task_is_swept_and_new_run_allowed(self):
        service = ChatRunService(stale_after_seconds=5.0)
        orphan = self._orphan(service, "sess", idle=10.0)

        async def quick(sink):
            await sink.send_json({"type": "done", "content": "ok"})

        fresh = await service.create("sess", "next", runner=quick)

        assert fresh.session_id == "sess"
        assert orphan.status == "failed"
        assert orphan.is_terminal
        assert "abandoned" in (orphan.error or "").lower()
        assert orphan.events[-1]["type"] == "error"
        assert orphan.events[-1]["seq"] == 2
        await _wait_task(fresh)

    @pytest.mark.asyncio
    async def test_get_sweeps_stale_run(self):
        service = ChatRunService(stale_after_seconds=5.0)
        orphan = self._orphan(service, "sess", idle=10.0)

        fetched = service.get(orphan.run_id)

        assert fetched is orphan
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.events[-1]["type"] == "error"

    @pytest.mark.asyncio
    async def test_live_task_is_never_swept(self):
        service = ChatRunService(stale_after_seconds=5.0)
        started = asyncio.Event()

        async def runner(sink):
            await sink.send_json({"type": "token", "content": "working"})
            started.set()
            await asyncio.sleep(30)

        run = await service.create("sess", "hi", runner=runner)
        await started.wait()
        # Quiet log, but the producer is alive: a long model call looks like this.
        run.last_event_at -= 100.0

        assert service.get(run.run_id) is run
        assert run.status == "running"
        with pytest.raises(RunConflictError):
            await service.create("sess", "second", runner=runner)

        await service.cancel(run.run_id)
        assert run.status == "cancelled"

    @pytest.mark.asyncio
    async def test_recent_event_refreshes_staleness(self):
        service = ChatRunService(stale_after_seconds=5.0)
        run = await service.create("sess", "hi", runner=lambda sink: asyncio.sleep(30))
        run.last_event_at -= 100.0
        run.publish({"type": "token", "content": "still here"})

        assert service.sweep_stale() == []
        assert run.status == "running"
        await service.cancel(run.run_id)

    @pytest.mark.asyncio
    async def test_timeout_is_configurable(self):
        lenient = ChatRunService(stale_after_seconds=60.0)
        strict = ChatRunService(stale_after_seconds=1.0)
        run = ChatRun(session_id="sess", prompt="orphan")
        run.last_event_at -= 5.0
        lenient._runs[run.run_id] = run
        strict._runs[run.run_id] = run

        assert lenient.sweep_stale() == []
        assert run.status == "running"

        assert strict.sweep_stale() == [run]
        assert run.status == "failed"

    def test_timeout_comes_from_config_when_unset(self, monkeypatch):
        from tensortruth.services import chat_run_service as crs

        monkeypatch.setattr(crs, "configured_stale_after_seconds", lambda: 42.0)
        assert ChatRunService().stale_after() == 42.0
        assert ChatRunService(stale_after_seconds=7.0).stale_after() == 7.0


class TestStaleTimeoutConfig:
    """The chat.runs.stale_after_seconds setting round-trips through YAML."""

    def test_default_is_five_minutes(self):
        from tensortruth.app_utils.config_schema import TensorTruthConfig

        config = TensorTruthConfig.from_dict({})
        assert config.chat.runs.stale_after_seconds == 300.0

    def test_override_is_parsed(self):
        from tensortruth.app_utils.config_schema import TensorTruthConfig

        config = TensorTruthConfig.from_dict(
            {"chat": {"runs": {"stale_after_seconds": 45}}}
        )
        assert config.chat.runs.stale_after_seconds == 45
        assert config.to_dict()["chat"]["runs"]["stale_after_seconds"] == 45

    def test_non_positive_timeout_is_rejected(self):
        from tensortruth.app_utils.config_schema import ChatRunsConfig

        with pytest.raises(ValueError):
            ChatRunsConfig(stale_after_seconds=0)


class TestRunKindAndMetadata:
    """ChatRun kind/metadata survive creation and serialisation."""

    @pytest.mark.asyncio
    async def test_elaboration_kind_and_metadata_round_trip(self):
        service = ChatRunService()
        metadata = {
            "message_index": 3,
            "selected_text": "a passage",
            "occurrence_index": 1,
        }

        async def runner(sink):
            await sink.send_json({"type": "done", "content": "ok"})

        run = await service.create(
            "sess", "", kind="elaboration", metadata=metadata, runner=runner
        )
        await _wait_task(run)

        payload = run._to_dict()
        assert run.kind == "elaboration"
        assert payload["kind"] == "elaboration"
        assert payload["metadata"] == metadata

    def test_chat_run_defaults(self):
        run = ChatRun(session_id="s", prompt="p")
        payload = run._to_dict()

        assert run.kind == "chat"
        assert run.metadata == {}
        assert payload["kind"] == "chat"
        assert payload["metadata"] == {}

    @pytest.mark.asyncio
    async def test_metadata_is_copied(self):
        service = ChatRunService()
        metadata = {"selected_text": "x"}

        async def runner(sink):
            await sink.send_json({"type": "done", "content": "ok"})

        run = await service.create(
            "sess", "", kind="elaboration", metadata=metadata, runner=runner
        )
        metadata["selected_text"] = "mutated"
        await _wait_task(run)

        assert run.metadata["selected_text"] == "x"
