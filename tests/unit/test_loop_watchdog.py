"""Tests for the event-loop lag watchdog.

The watchdog is the safety net that makes future blocking calls visible: a
daemon thread measures loop lag and logs a WARNING plus the main-thread stack
when the loop stalls past the threshold.
"""

import asyncio
import logging
import threading
import time

import pytest

from tensortruth.services.loop_watchdog import (
    is_running,
    start_loop_watchdog,
    stop_loop_watchdog,
)

LOGGER_NAME = "tensortruth.services.loop_watchdog"


def _watchdog_warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.WARNING
    ]


def test_blocking_loop_logs_exactly_one_warning_with_stack(caplog):
    """A 0.3s stall with a 0.1s threshold produces one WARNING + stack."""
    stop_loop_watchdog()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        assert start_loop_watchdog(loop, threshold=0.1) is True
        # Simulate a synchronous call accidentally left in an async handler.
        time.sleep(0.3)
        # Let the watchog observe the loop recovering before we tear down.
        await asyncio.sleep(0.3)

    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            loop.run_until_complete(_run())
    finally:
        stop_loop_watchdog()
        loop.close()
        asyncio.set_event_loop(None)

    warnings = _watchdog_warnings(caplog)
    assert len(warnings) == 1, f"expected one WARNING, got {len(warnings)}"
    message = warnings[0].getMessage()
    assert "Event loop blocked" in message
    assert "Main thread stack" in message
    # The stack must point at the frame that blocked the loop.
    assert "time.sleep" in message


def test_start_stop_are_idempotent():
    stop_loop_watchdog()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        assert start_loop_watchdog(loop, threshold=0.5) is True
        assert is_running() is True
        # Second start is a no-op, not an error or a second thread.
        assert start_loop_watchdog(loop, threshold=0.5) is True
        assert is_running() is True

        stop_loop_watchdog()
        assert is_running() is False
        # Second stop is also a no-op.
        stop_loop_watchdog()
        assert is_running() is False
    finally:
        stop_loop_watchdog()
        loop.close()
        asyncio.set_event_loop(None)


def test_env_kill_switch_disables_watchdog(monkeypatch):
    stop_loop_watchdog()
    monkeypatch.setenv("TENSORTRUTH_LOOP_WATCHDOG", "0")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        assert start_loop_watchdog(loop, threshold=0.1) is False
        assert is_running() is False
    finally:
        stop_loop_watchdog()
        loop.close()
        asyncio.set_event_loop(None)


def test_start_is_noop_off_the_main_thread():
    """uvicorn's loop lives on the main thread; dump that thread's stack."""
    stop_loop_watchdog()
    result = {}

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["started"] = start_loop_watchdog(loop, threshold=0.1)
        finally:
            loop.close()

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()

    assert result["started"] is False
    assert is_running() is False


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " 0 "])
def test_kill_switch_accepts_common_falsey_spellings(monkeypatch, value):
    stop_loop_watchdog()
    monkeypatch.setenv("TENSORTRUTH_LOOP_WATCHDOG", value)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        assert start_loop_watchdog(loop, threshold=0.1) is False
    finally:
        stop_loop_watchdog()
        loop.close()
        asyncio.set_event_loop(None)
