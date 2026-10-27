"""Unit tests for the long-stage progress heartbeat.

Regression context: a Marker PDF conversion or a bge-m3 embedding pass is a
single blocking call with no inner callback. Reporting progress only at stage
boundaries left the UI stuck on "Converting X (1/2)" at 0% for the whole
conversion, which is indistinguishable from a hang.
"""

import threading
import time

import pytest

from tensortruth.utils import progress as progress_mod
from tensortruth.utils.progress import (
    format_elapsed,
    monotonic_callback,
    stage_heartbeat,
)


@pytest.fixture(autouse=True)
def fast_heartbeat(monkeypatch):
    """Shrink the heartbeat interval so tests stay fast."""
    monkeypatch.setattr(progress_mod, "DEFAULT_HEARTBEAT_INTERVAL", 0.02)


def _recorder():
    calls = []
    lock = threading.Lock()

    def _cb(stage, current, total):
        with lock:
            calls.append((stage, current, total))

    return calls, _cb


def test_heartbeat_advances_while_blocked():
    calls, cb = _recorder()
    with stage_heartbeat(cb, "Converting a.pdf (1/2)", 0, 30):
        time.sleep(0.25)

    pcts = [c[1] for c in calls]
    assert len(calls) > 3, f"expected repeated reports, got {calls}"
    assert pcts[0] == 0
    assert max(pcts) > 0, "progress never left 0% while the stage ran"
    assert pcts[-1] == 30, "final report should be the stage band end"
    assert pcts == sorted(pcts), f"progress went backwards: {pcts}"
    assert all(c[2] == 100 for c in calls)


def test_heartbeat_stays_below_stage_end_while_working():
    calls, cb = _recorder()
    with stage_heartbeat(cb, "Converting", 10, 20):
        # Interval is 20ms; long enough that the creep would overshoot without a cap.
        time.sleep(0.4)

    during = [c[1] for c in calls[:-1]]
    assert during, "no progress reported"
    assert max(during) <= 19, f"claimed to finish the stage early: {during}"
    assert calls[-1][1] == 20


def test_heartbeat_includes_elapsed_time():
    calls, cb = _recorder()
    with stage_heartbeat(cb, "Converting a.pdf (1/2)", 0, 30):
        time.sleep(0.1)

    assert "elapsed" in calls[0][0]
    assert calls[0][0].startswith("Converting a.pdf (1/2)")


def test_heartbeat_accepts_callable_message():
    calls, cb = _recorder()
    with stage_heartbeat(cb, lambda elapsed: f"Embedding ({int(elapsed)}s)", 70, 80):
        time.sleep(0.05)

    assert calls[0][0].startswith("Embedding")


def test_heartbeat_does_not_report_completion_on_error():
    calls, cb = _recorder()
    with pytest.raises(RuntimeError):
        with stage_heartbeat(cb, "Converting", 0, 30):
            time.sleep(0.05)
            raise RuntimeError("marker exploded")

    assert all(c[1] < 30 for c in calls), f"reported the stage as done: {calls}"


def test_heartbeat_stops_after_exit():
    calls, cb = _recorder()
    with stage_heartbeat(cb, "Converting", 0, 30):
        time.sleep(0.05)
    count_after_exit = len(calls)
    time.sleep(0.15)
    assert len(calls) == count_after_exit, "heartbeat kept running after the stage"


def test_heartbeat_single_point_band_reports_once_plus_end():
    calls, cb = _recorder()
    with stage_heartbeat(cb, "Loading model", 0, 0):
        time.sleep(0.1)

    assert [c[1] for c in calls][0] == 0
    assert calls[-1][1] == 0


def test_monotonic_callback_drops_regressions():
    calls, cb = _recorder()
    wrapped = monotonic_callback(cb)

    wrapped("Converting", 10, 100)
    wrapped("Converting", 40, 100)
    wrapped("Embedding", 5, 100)
    wrapped("Complete", 100, 100)

    assert [c[1] for c in calls] == [10, 40, 100]
    assert calls[-1][0] == "Complete"


def test_monotonic_callback_handles_zero_total():
    calls, cb = _recorder()
    wrapped = monotonic_callback(cb)
    wrapped("Converting", 0, 0)
    assert calls[0][1] == 0


def test_format_elapsed():
    assert format_elapsed(0) == "0s"
    assert format_elapsed(59.9) == "59s"
    assert format_elapsed(64) == "1m 04s"
    assert format_elapsed(3725) == "1h 02m"
