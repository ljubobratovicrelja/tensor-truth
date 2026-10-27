"""Per-category accounting of the context window sent to the LLM.

The status bar and the /compact command both need to answer "what is filling
the context?".  Exact numbers only exist at the moment the orchestrator
composes a prompt, so the orchestrator records a snapshot there
(:func:`record_snapshot`); readers then call :func:`build_report` on it.

Categories:
    tools      — JSON schemas + descriptions of the tool set
    system     — orchestrator system prompt, excluding modules/documents
    modules    — indexed-module listing embedded in the system prompt
    documents  — direct-to-context document text
    history    — conversation messages after budgeting
    prompt     — the user message for the turn

All sizes are characters; tokens are estimated at ``CHARS_PER_TOKEN`` and are
labelled as estimates everywhere they surface.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4

#: Ordered category ids, cheapest-to-reason-about first for the UI.
CATEGORIES: List[str] = ["system", "tools", "modules", "documents", "history", "prompt"]

#: Human labels for the UI.
CATEGORY_LABELS: Dict[str, str] = {
    "system": "System prompt",
    "tools": "Tool schemas",
    "modules": "Indexed modules",
    "documents": "Attached documents",
    "history": "Conversation history",
    "prompt": "Current message",
}

#: Short explanations shown as tooltips.
CATEGORY_HINTS: Dict[str, str] = {
    "system": "Role, tool-routing rules and instructions sent with every turn",
    "tools": "JSON schemas of the available tools (always sent)",
    "modules": "Descriptions of the indexed modules listed in the prompt",
    "documents": "Documents pinned to the conversation context",
    "history": "Earlier messages included after budgeting",
    "prompt": "The message you just typed",
}


@dataclass
class ContextSnapshot:
    """Per-category character counts measured on one composed prompt."""

    session_id: str
    chars: Dict[str, int] = field(default_factory=dict)
    context_window: int = 0
    model_name: Optional[str] = None
    messages_included: int = 0
    messages_total: int = 0
    measured_at: float = field(default_factory=time.time)

    @property
    def total_chars(self) -> int:
        return sum(self.chars.values())


_snapshots: Dict[str, ContextSnapshot] = {}

#: Last measured system/tools/modules overhead. These are constant for a given
#: model + module set, so an unmeasured session can still show them instead of
#: pretending they are zero.
_static_overhead: Dict[str, int] = {}
_lock = threading.Lock()

#: Bound the cache; snapshots are cheap to recompute and only useful for the
#: sessions the user is actually looking at.
_MAX_SNAPSHOTS = 256


def record_snapshot(
    session_id: str,
    chars: Dict[str, int],
    *,
    context_window: int = 0,
    model_name: Optional[str] = None,
    messages_included: int = 0,
    messages_total: int = 0,
) -> ContextSnapshot:
    """Record the measured context composition for a session."""
    cleaned = {k: max(0, int(chars.get(k, 0))) for k in CATEGORIES}
    snapshot = ContextSnapshot(
        session_id=session_id,
        chars=cleaned,
        context_window=int(context_window or 0),
        model_name=model_name,
        messages_included=messages_included,
        messages_total=messages_total,
    )
    with _lock:
        _snapshots[session_id] = snapshot
        if len(_snapshots) > _MAX_SNAPSHOTS:
            oldest = sorted(_snapshots.items(), key=lambda kv: kv[1].measured_at)
            for stale_id, _ in oldest[: len(_snapshots) - _MAX_SNAPSHOTS]:
                _snapshots.pop(stale_id, None)
    return snapshot


def get_snapshot(session_id: str) -> Optional[ContextSnapshot]:
    """Return the last measured snapshot for a session, if any."""
    with _lock:
        return _snapshots.get(session_id)


def clear_snapshot(session_id: str) -> None:
    """Drop a session's snapshot (after compaction or message deletion)."""
    with _lock:
        _snapshots.pop(session_id, None)


def record_static_overhead(chars: Dict[str, int]) -> None:
    """Record the fixed per-turn overhead (system prompt + tool schemas).

    Measured once when the orchestrator builds its tool set, so unmeasured
    sessions can still show a realistic breakdown instead of zeros.
    """
    with _lock:
        _static_overhead.update(
            {
                "system": max(0, int(chars.get("system", 0))),
                "tools": max(0, int(chars.get("tools", 0))),
                "modules": max(0, int(chars.get("modules", 0))),
            }
        )


def get_static_overhead() -> Dict[str, int]:
    """Last measured system/tools/modules char counts (empty before a turn)."""
    with _lock:
        return dict(_static_overhead)


def clear_all() -> None:
    """Drop every snapshot (tests)."""
    with _lock:
        _snapshots.clear()


def build_report(
    chars: Dict[str, int],
    *,
    context_window: int = 0,
    measured: bool = False,
    model_name: Optional[str] = None,
    messages_included: int = 0,
    messages_total: int = 0,
    measured_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Turn raw character counts into the API payload.

    ``measured`` is False when the numbers come from a live estimate rather
    than an actually-composed prompt — the UI labels those differently.
    """
    total_chars = sum(max(0, int(chars.get(k, 0))) for k in CATEGORIES)
    total_tokens = total_chars // CHARS_PER_TOKEN
    categories = []
    for key in CATEGORIES:
        n_chars = max(0, int(chars.get(key, 0)))
        tokens = n_chars // CHARS_PER_TOKEN
        categories.append(
            {
                "id": key,
                "label": CATEGORY_LABELS[key],
                "hint": CATEGORY_HINTS[key],
                "chars": n_chars,
                "tokens": tokens,
                "share": (n_chars / total_chars) if total_chars else 0.0,
            }
        )
    return {
        "measured": measured,
        "categories": categories,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "context_window": int(context_window or 0),
        "used_percent": (
            (total_tokens / context_window * 100.0) if context_window else 0.0
        ),
        "remaining_tokens": (
            max(0, context_window - total_tokens) if context_window else None
        ),
        "model_name": model_name,
        "messages_included": messages_included,
        "messages_total": messages_total,
        "measured_at": measured_at,
    }


def report_for_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Report from the stored snapshot, or None when nothing was measured."""
    snapshot = get_snapshot(session_id)
    if snapshot is None:
        return None
    return build_report(
        snapshot.chars,
        context_window=snapshot.context_window,
        measured=True,
        model_name=snapshot.model_name,
        messages_included=snapshot.messages_included,
        messages_total=snapshot.messages_total,
        measured_at=snapshot.measured_at,
    )


def tool_schema_chars(tools: List[Any]) -> int:
    """Character size of the JSON tool schemas llama-index sends to the LLM."""
    import json

    total = 0
    for tool in tools:
        metadata = getattr(tool, "metadata", None)
        if metadata is None:
            continue
        total += len(metadata.name or "")
        total += len(metadata.description or "")
        fn_schema = getattr(metadata, "fn_schema", None)
        if fn_schema is not None:
            try:
                total += len(json.dumps(fn_schema.model_json_schema()))
            except Exception:  # pragma: no cover - defensive
                logger.debug("Could not serialize tool schema for %s", metadata.name)
    return total
