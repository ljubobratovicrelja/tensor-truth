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
