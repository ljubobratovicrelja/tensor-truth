"""Event-loop lag watchdog.

TensorTruth runs a single uvicorn event loop that serves the REST API *and*
every chat token WebSocket. A synchronous call accidentally left inside an
``async def`` (PDF conversion, a large filesystem walk, a blocking HTTP
request) freezes every request and every in-flight stream until it returns.

This module starts a cheap daemon thread that measures event-loop lag. When
the loop is blocked longer than ``threshold`` seconds it logs a WARNING with
the stack of the main thread (``sys._current_frames()``), which points at the
offending synchronous call.

Usage::

    from tensortruth.services.loop_watchdog import start_loop_watchdog

    start_loop_watchdog(asyncio.get_running_loop(), threshold=0.5)

Set ``TENSORTRUTH_LOOP_WATCHDOG=0`` to disable it entirely.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Optional

logger = logging.getLogger(__name__)

# How often the watchdog pings the loop. Small enough to catch sub-second
# stalls without measurable overhead.
_TICK_SECONDS = 0.1

# Only report one warning per this many seconds while the loop stays blocked,
# so a 16s stall produces a handful of diagnostics instead of a flood.
_REPORT_COOLDOWN_SECONDS = 2.0

_watchdog_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_lock = threading.Lock()


def _disabled() -> bool:
    """Whether the watchdog is disabled by environment variable."""
    return os.environ.get("TENSORTRUTH_LOOP_WATCHDOG", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    )


def _format_main_thread_stack() -> str:
    """Return a formatted stack trace for the main thread, if available."""
    frame = sys._current_frames().get(threading.main_thread().ident or 0)
    if frame is None:
        return "  <main thread stack unavailable>"
    return "".join(traceback.format_stack(frame))


def _watchdog_loop(
    loop: asyncio.AbstractEventLoop,
    threshold: float,
    stop_event: threading.Event,
) -> None:
    """Body of the watchdog daemon thread."""
    last_report = 0.0
    while not stop_event.is_set():
        # A loop that is not running (e.g. between startup and serve) is not
        # "blocked"; skip rather than raising a false alarm.
        if not loop.is_running():
            stop_event.wait(_TICK_SECONDS)
            continue

        started = time.monotonic()
        blocked = False
        try:
            # Schedule a no-op callback on the loop. If the loop is blocked,
            # this future will not complete until the blocking call returns.
            future = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop)
            future.result(timeout=threshold)
        except FutureTimeoutError:
            # The round trip took longer than the threshold: the loop was
            # blocked for at least ``threshold`` seconds by definition.
            blocked = True
        except RuntimeError:
            # Loop shut down between ticks.
            if loop.is_closed():
                return
        except Exception:  # pragma: no cover - defensive
            if loop.is_closed():
                return

        if blocked:
            now = time.monotonic()
            if (now - last_report) >= _REPORT_COOLDOWN_SECONDS:
                last_report = now
                logger.warning(
                    "Event loop blocked for at least %.2fs (threshold %.2fs). "
                    "Main thread stack:\n%s",
                    now - started,
                    threshold,
                    _format_main_thread_stack(),
                )
        stop_event.wait(_TICK_SECONDS)


def start_loop_watchdog(
    loop: Optional[asyncio.AbstractEventLoop] = None,
    threshold: float = 0.5,
) -> bool:
    """Start the event-loop lag watchdog.

    Idempotent: a second call while the watchdog is running is a no-op.

    Args:
        loop: Event loop to watch. Defaults to the running loop. The watchdog
            is a no-op when there is no running loop or when called from a
            non-main thread (uvicorn's loop lives on the main thread, and the
            stack dump targets that thread).
        threshold: Seconds of lag before a WARNING is logged.

    Returns:
        True when the watchdog is running after the call, False when it was
        skipped (disabled, already running is reported as True, wrong thread).
    """
    global _watchdog_thread, _stop_event

    if _disabled():
        logger.debug("Loop watchdog disabled via TENSORTRUTH_LOOP_WATCHDOG")
        return False

    if threading.current_thread() is not threading.main_thread():
        logger.debug("Loop watchdog not started: not on the main thread")
        return False

    with _lock:
        if _watchdog_thread is not None and _watchdog_thread.is_alive():
            return True

        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.debug("Loop watchdog not started: no running event loop")
                return False

        stop_event = threading.Event()
        thread = threading.Thread(
            target=_watchdog_loop,
            args=(loop, threshold, stop_event),
            name="tensortruth-loop-watchdog",
            daemon=True,
        )
        _stop_event = stop_event
        _watchdog_thread = thread
        thread.start()
        logger.debug(
            "Loop watchdog started (threshold=%.2fs, tick=%.2fs)",
            threshold,
            _TICK_SECONDS,
        )
        return True


def stop_loop_watchdog() -> None:
    """Stop the watchdog if running. Idempotent and safe to call twice."""
    global _watchdog_thread, _stop_event

    with _lock:
        thread = _watchdog_thread
        stop_event = _stop_event
        _watchdog_thread = None
        _stop_event = None

    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)


def is_running() -> bool:
    """Whether the watchdog thread is currently running."""
    thread = _watchdog_thread
    return thread is not None and thread.is_alive()
