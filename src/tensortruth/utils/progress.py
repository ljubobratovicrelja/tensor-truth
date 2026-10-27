"""Progress helpers for long-running, coarse-grained build stages.

Long index-build stages (Marker PDF conversion, bge-m3 embedding) are single
blocking calls that can take several minutes each and expose no page/chunk
callback. Reporting progress only at stage boundaries makes the UI look frozen
-- a build that has been converting its first PDF for six minutes still reads
"Converting X (1/2)" at 0%.

``stage_heartbeat`` keeps such a stage visibly alive: it reports the stage
immediately, then ticks a monotonically increasing percentage *inside the
stage's own band* while the blocking call runs, and reports the band's end once
the call returns. It never claims to have finished the stage early, so the
percentage stays honest -- it only stops looking like a hang.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)

# How often the heartbeat re-reports a busy stage.
DEFAULT_HEARTBEAT_INTERVAL = 2.0

ProgressCallback = Callable[[str, int, int], None]
MessageSpec = Union[str, Callable[[float], str]]


def format_elapsed(seconds: float) -> str:
    """Human-readable elapsed suffix, e.g. ``1m 04s``."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


@contextmanager
def stage_heartbeat(
    cb: ProgressCallback,
    message: MessageSpec,
    start_pct: int,
    end_pct: Optional[int] = None,
    interval: Optional[float] = None,
):
    """Keep a blocking stage reporting progress so it never looks frozen.

    Args:
        cb: Progress callback ``(stage, current, total)``; ``current`` and
            ``total`` are on a 0-100 scale.
        message: Base stage text, or a callable taking elapsed seconds and
            returning the text.
        start_pct: Percentage reported when the stage starts.
        end_pct: Percentage reported when the stage finishes. Ticks stay at
            most one point below it. Defaults to ``start_pct``.
        interval: Seconds between heartbeat ticks. Defaults to
            ``DEFAULT_HEARTBEAT_INTERVAL``; read at call time so tests can patch
            the module constant.

    Yields:
        None. Progress is delivered through ``cb``.

    The final percentage is emitted only if the block completes without raising,
    so a failing stage is not reported as finished.
    """
    start = max(0, min(int(start_pct), 100))
    end = start if end_pct is None else max(start, min(int(end_pct), 100))
    tick_interval = DEFAULT_HEARTBEAT_INTERVAL if interval is None else float(interval)
    if tick_interval <= 0:
        tick_interval = DEFAULT_HEARTBEAT_INTERVAL
    started_at = time.monotonic()
    stop = threading.Event()

    def _text(elapsed: float) -> str:
        base = message(elapsed) if callable(message) else str(message)
        return f"{base} \u2014 {format_elapsed(elapsed)} elapsed"

    def _report(pct: int, elapsed: float) -> None:
        cb(_text(elapsed), pct, 100)

    _report(start, 0.0)

    # Highest percentage we may creep to while still working.
    ceiling = end - 1 if end > start else start

    def _tick() -> None:
        pct = start
        try:
            while not stop.wait(tick_interval):
                if pct < ceiling:
                    pct += 1
                _report(pct, time.monotonic() - started_at)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Progress heartbeat stopped reporting: %s", exc)

    thread = threading.Thread(target=_tick, name="stage-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=max(tick_interval, 0.1))
    _report(end, time.monotonic() - started_at)


def monotonic_callback(cb: ProgressCallback) -> ProgressCallback:
    """Wrap ``cb`` so the reported percentage never goes backwards.

    Heartbeats and stage transitions are reported from different threads; a
    late tick from a finished stage must not move the bar backwards.
    """

    lock = threading.Lock()
    highest = -1

    def _wrapped(stage: str, current: int, total: int) -> None:
        nonlocal highest
        pct = min(int(current / total * 100), 100) if total else 0
        with lock:
            if pct < highest:
                return
            highest = pct
            # Emit while holding the lock so a late heartbeat tick from a
            # previous stage cannot interleave and reorder reports.
            cb(stage, current, total)

    return _wrapped
