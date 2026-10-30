"""Server-owned chat runs: durable, sequence-numbered event logs per turn.

A chat turn is a *run*: a background asyncio task owned by this registry, with an
append-only, monotonically sequenced event log. Clients attach to a run with a
cursor (``after``) and replay from it; disconnecting never touches the producer.
See ``docs/RUNS.md`` for the design.
"""

import asyncio
import logging
import time
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
)

logger = logging.getLogger(__name__)

# Finished runs are retained this long (seconds) for late reconnects.
DEFAULT_RETENTION_SECONDS = 600.0
# Per-run event log cap; oldest events are dropped, ``seq`` stays monotonic.
DEFAULT_MAX_EVENTS = 20000
# A non-terminal run with no live task and no event for this long is abandoned.
DEFAULT_STALE_AFTER_SECONDS = 300.0

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_TERMINAL_TYPES = frozenset({"done", "error"})

_CLOSED = object()


class RunConflictError(RuntimeError):
    """Raised when a session already has an active run."""

    def __init__(self, session_id: str, run_id: str) -> None:
        super().__init__(f"Session {session_id} already has an active run ({run_id})")
        self.session_id = session_id
        self.run_id = run_id


class RunNotFoundError(KeyError):
    """Raised when a run id is unknown (or already GC'd)."""


class ChatRun:
    """State and event log for a single chat turn."""

    def __init__(
        self,
        session_id: str,
        prompt: str,
        images: Optional[List[Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
        max_events: int = DEFAULT_MAX_EVENTS,
        kind: str = "chat",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id: str = run_id or uuid.uuid4().hex
        self.session_id: str = session_id
        self.prompt: str = prompt
        self.images: List[Dict[str, Any]] = list(images or [])
        self.kind: str = kind
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.status: str = "running"
        self.error: Optional[str] = None
        self.events: List[Dict[str, Any]] = []
        self.created_at: float = time.monotonic()
        # Refreshed on every appended event; the staleness sweep reads it.
        self.last_event_at: float = self.created_at
        self.finished_at: Optional[float] = None
        self.task: Optional["asyncio.Task[Any]"] = None
        self.max_events = max_events
        # Type of the last terminal event published through the log.
        self.last_terminal_type: Optional[str] = None
        self._seq: int = 0
        self._subscribers: set[asyncio.Queue] = set()

    # -- state -------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Whether the run has reached a terminal status."""
        return self.status in TERMINAL_STATUSES

    @property
    def seq(self) -> int:
        """Last published sequence number (0 before the first event)."""
        return self._seq

    # -- event log ---------------------------------------------------------

    def publish(self, payload: Dict[str, Any]) -> int:
        """Append an event to the log and fan it out to live subscribers.

        Returns the assigned ``seq``. ``seq`` starts at 1 and is strictly
        increasing for the lifetime of the run, even after the log is trimmed
        or events are dropped for slow subscribers.
        """
        self._seq += 1
        event = dict(payload)
        event["seq"] = self._seq
        event_type = event.get("type")
        if event_type in _TERMINAL_TYPES:
            self.last_terminal_type = event_type

        # Any activity proves the producer is alive, so a long turn that keeps
        # streaming is never mistaken for an abandoned one.
        self.last_event_at = time.monotonic()

        self.events.append(event)
        overflow = len(self.events) - self.max_events
        if overflow > 0:
            del self.events[:overflow]

        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queues
                logger.warning(
                    "Dropping event seq=%s for slow subscriber of run %s",
                    event["seq"],
                    self.run_id,
                )
        return event["seq"]

    def snapshot(self, after: int = 0) -> List[Dict[str, Any]]:
        """Return retained events with ``seq > after``, in order."""
        if after <= 0:
            return list(self.events)
        return [event for event in self.events if event["seq"] > after]

    # -- lifecycle ---------------------------------------------------------

    def close(self, status: str = "completed", error: Optional[str] = None) -> None:
        """Mark the run terminal exactly once and wake all subscribers."""
        if self.is_terminal:
            return
        self.status = status
        self.error = error
        self.finished_at = time.monotonic()
        for queue in list(self._subscribers):
            queue.put_nowait(_CLOSED)

    # -- subscriber plumbing (internal) -----------------------------------

    def _subscribe(self) -> "asyncio.Queue[Any]":
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def _unsubscribe(self, queue: "asyncio.Queue[Any]") -> None:
        self._subscribers.discard(queue)

    def _to_dict(self, after: int = 0) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "kind": self.kind,
            "metadata": dict(self.metadata),
            "status": self.status,
            "error": self.error,
            "events": self.snapshot(after),
        }


RunnerFn = Callable[[Any], Awaitable[None]]


def configured_stale_after_seconds() -> float:
    """Return ``chat.runs.stale_after_seconds`` from config.

    Read lazily so this module stays import-light: the config service lives
    behind ``api.deps``, which imports route modules. Any config problem must
    not disable the sweep, so failures fall back to the built-in default.
    """
    try:
        from tensortruth.api.deps import get_config_service

        config_service = get_config_service()
        # Never let a config read create ~/.tensortruth on a bare machine: a
        # missing file simply means "use the built-in default".
        if not config_service.config_file.exists():
            return DEFAULT_STALE_AFTER_SECONDS
        return float(config_service.load().chat.runs.stale_after_seconds)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Falling back to the default stale-run timeout", exc_info=True)
        return DEFAULT_STALE_AFTER_SECONDS


def make_default_runner(
    session_id: str,
    prompt: str,
    images: Optional[List[Dict[str, Any]]] = None,
    kind: str = "chat",
    metadata: Optional[Dict[str, Any]] = None,
) -> RunnerFn:
    """Build the production runner that executes a real chat turn.

    Imported lazily so this module stays import-light and free of cycles.
    """
    from tensortruth.services.chat_runner import run_turn

    async def _runner(sink: Any) -> None:
        await run_turn(
            sink,
            session_id=session_id,
            prompt=prompt,
            images=images or [],
            kind=kind,
            metadata=metadata,
        )

    return _runner


class ChatRunService:
    """In-memory registry of chat runs (single-process app)."""

    def __init__(
        self,
        retention_seconds: float = DEFAULT_RETENTION_SECONDS,
        max_events: int = DEFAULT_MAX_EVENTS,
        stale_after_seconds: Optional[float] = None,
    ) -> None:
        self.retention_seconds = retention_seconds
        self.max_events = max_events
        # ``None`` means "read chat.runs.stale_after_seconds from config".
        self.stale_after_seconds = stale_after_seconds
        # Resolved once: the sweep runs on request paths and re-parsing YAML on
        # every poll would block the loop for no benefit. In-memory runs do not
        # survive a restart anyway, so a changed timeout applies next boot.
        self._resolved_stale_after: Optional[float] = None
        self._runs: Dict[str, ChatRun] = {}
        self._lock: Optional[asyncio.Lock] = None
        # Strong references; asyncio only keeps weak refs to tasks.
        self._tasks: set["asyncio.Task[Any]"] = set()

    # -- helpers -----------------------------------------------------------

    def _get_lock(self) -> asyncio.Lock:
        # Created lazily: an asyncio.Lock binds to the running loop.
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _gc(self) -> None:
        """Drop finished runs past their retention window."""
        now = time.monotonic()
        expired = [
            run_id
            for run_id, run in self._runs.items()
            if run.is_terminal
            and run.finished_at is not None
            and now - run.finished_at >= self.retention_seconds
        ]
        for run_id in expired:
            self._runs.pop(run_id, None)
            logger.debug("GC'd finished chat run %s", run_id)

    def stale_after(self) -> float:
        """Return the effective staleness timeout in seconds."""
        if self.stale_after_seconds is not None:
            return self.stale_after_seconds
        if self._resolved_stale_after is None:
            self._resolved_stale_after = configured_stale_after_seconds()
        return self._resolved_stale_after

    def sweep_stale(self) -> List[ChatRun]:
        """Finalise runs whose producer is gone, releasing the session lock.

        A run is dead when it is still non-terminal *and* either its asyncio
        task was never registered (this process never owned it) or its task is
        already done, *and* nothing has been appended to its log for longer
        than the timeout. A live, unfinished task is never swept, no matter how
        quiet the log is: a slow model call emits no events yet is healthy.

        Synchronous on purpose — it runs inside ``create``'s lock and inside
        request handlers, and finalising only touches in-memory state.
        """
        now = time.monotonic()
        timeout = self.stale_after()
        swept: List[ChatRun] = []
        for run in list(self._runs.values()):
            if run.is_terminal:
                continue
            task = run.task
            if task is not None and not task.done():
                continue
            idle = now - run.last_event_at
            if idle < timeout:
                continue

            detail = (
                f"Run abandoned: no activity for {idle:.0f}s and its generation "
                "task is no longer alive"
            )
            logger.warning(
                "Sweeping stale chat run %s for session %s (%s)",
                run.run_id,
                run.session_id,
                detail,
            )
            run.publish({"type": "error", "detail": detail})
            run.close("failed", error=detail)
            swept.append(run)
        return swept

    # -- public API --------------------------------------------------------

    async def create(
        self,
        session_id: str,
        prompt: str,
        images: Optional[List[Dict[str, Any]]] = None,
        *,
        kind: str = "chat",
        metadata: Optional[Dict[str, Any]] = None,
        runner: Optional[RunnerFn] = None,
    ) -> ChatRun:
        """Create a run and schedule its generation task on the server loop.

        Raises ``RunConflictError`` if the session already has an active run.
        The task is owned by the registry (strong ref + ``run.task``), never by
        an HTTP/WebSocket handler, so handler cancellation cannot kill it.
        """
        async with self._get_lock():
            self._gc()
            # A run left behind by a dead producer must not hold the session
            # lock forever, or the next turn can never start.
            self.sweep_stale()
            active = self.active_for_session(session_id)
            if active is not None:
                raise RunConflictError(session_id, active.run_id)

            run = ChatRun(
                session_id=session_id,
                prompt=prompt,
                images=images,
                max_events=self.max_events,
                kind=kind,
                metadata=metadata,
            )
            self._runs[run.run_id] = run

            if runner is None:
                runner = make_default_runner(
                    session_id, prompt, images, kind=kind, metadata=metadata
                )

            task = asyncio.create_task(
                self._execute(run, runner), name=f"chat-run-{run.run_id}"
            )
            run.task = task
            self._tasks.add(task)
            task.add_done_callback(lambda done: self._on_task_done(run, done))
            logger.info("Started chat run %s for session %s", run.run_id, session_id)
            return run

    def get(self, run_id: str) -> Optional[ChatRun]:
        """Return a run by id, or ``None`` once GC'd.

        Sweeps first so ``GET /api/runs/{id}`` reports an abandoned run as
        failed instead of a phantom that is forever "running".
        """
        self._gc()
        self.sweep_stale()
        return self._runs.get(run_id)

    def active_for_session(self, session_id: str) -> Optional[ChatRun]:
        """Return the latest non-terminal run for a session, if any."""
        self._gc()
        candidates = [
            run
            for run in self._runs.values()
            if run.session_id == session_id and not run.is_terminal
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.created_at)

    async def cancel(self, run_id: str) -> Optional[ChatRun]:
        """Cooperatively cancel a run. Returns the run, or ``None`` if unknown.

        Cancelling an already finished run is a no-op that returns the run.
        """
        run = self.get(run_id)
        if run is None:
            return None
        if run.is_terminal:
            return run

        run.publish({"type": "error", "detail": "Generation cancelled"})
        run.close("cancelled", error="cancelled")

        task = run.task
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait({task}, timeout=0.5)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error while awaiting cancelled run %s", run_id)
        return run

    async def subscribe(
        self, run: ChatRun, after: int = 0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Async generator: replay events with ``seq > after`` then live-tail.

        Exits cleanly when the run closes. Exceptions raised by a subscriber
        (or its disappearance) are confined to the subscriber's own generator;
        the producer is never affected.
        """
        queue = run._subscribe()
        last = after
        try:
            for event in run.snapshot(after):
                last = event["seq"]
                yield event

            # Drain anything already queued even if the run just closed: the
            # terminal event (done/error) must never be skipped.
            while True:
                if run.is_terminal and queue.empty():
                    break
                event = await queue.get()
                if event is _CLOSED:
                    break
                if event["seq"] <= last:
                    continue
                last = event["seq"]
                yield event
        finally:
            run._unsubscribe(queue)

    # -- internals ---------------------------------------------------------

    async def _execute(self, run: ChatRun, runner: RunnerFn) -> None:
        """Run the turn, owning terminal status and error reporting."""
        from tensortruth.services.chat_runner import RunEventSink

        sink = RunEventSink(run)
        try:
            await runner(sink)
        except asyncio.CancelledError:
            if not run.is_terminal:
                run.publish({"type": "error", "detail": "Generation cancelled"})
                run.close("cancelled", error="cancelled")
            raise
        except Exception as exc:
            logger.exception("Chat run %s failed", run.run_id)
            if sink.last_terminal_type != "error":
                run.publish({"type": "error", "detail": str(exc)})
            run.close("failed", error=str(exc))
        except BaseException as exc:
            # SystemExit/KeyboardInterrupt and friends must not leave the run
            # "running": publish the terminal event, then let it propagate.
            logger.exception("Chat run %s aborted", run.run_id)
            if not run.is_terminal:
                run.publish({"type": "error", "detail": str(exc) or type(exc).__name__})
                run.close("failed", error=str(exc) or type(exc).__name__)
            raise
        else:
            if not run.is_terminal:
                run.close("completed")

    def _on_task_done(self, run: ChatRun, task: "asyncio.Task[Any]") -> None:
        """Finalise a run whose task ended without reaching a terminal state.

        ``_execute`` normally closes the run itself. This callback covers the
        cases where its body never got to run at all — most importantly a task
        cancelled before its first step — or where a BaseException escaped it.
        Without it such a run stays "running" and blocks the session forever.
        """
        self._tasks.discard(task)
        if run.is_terminal:
            return

        if task.cancelled():
            logger.warning("Chat run %s task was cancelled; finalising", run.run_id)
            run.publish({"type": "error", "detail": "Generation cancelled"})
            run.close("cancelled", error="cancelled")
            return

        exc = task.exception()
        if exc is None:
            run.close("completed")
            return

        detail = str(exc) or type(exc).__name__
        logger.error("Chat run %s ended without finalising: %s", run.run_id, detail)
        run.publish({"type": "error", "detail": detail})
        run.close("failed", error=detail)


_service: Optional[ChatRunService] = None


def get_chat_run_service() -> ChatRunService:
    """Return the process-wide ChatRunService singleton."""
    global _service
    if _service is None:
        _service = ChatRunService()
    return _service
