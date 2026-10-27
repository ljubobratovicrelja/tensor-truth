"""Conversation compaction ("/compact").

Older messages are summarised into a single assistant message, the most recent
``keep_last`` messages are kept verbatim, and the replaced messages are written
to a backup file so nothing is lost silently.

The summarisation itself is delegated to a callable so the pure message
surgery (which is what needs testing) is separate from the LLM provider.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Messages left untouched at the end of the conversation.
DEFAULT_KEEP_LAST = 6

#: Per-message cap when building the transcript fed to the summariser.
PER_MESSAGE_CAP = 6_000

#: Transcript chunk size for summarising very long histories.
CHUNK_CHARS = 60_000

SUMMARY_HEADER = "[Conversation summary]"

SUMMARY_PROMPT = """You are compacting a long technical conversation so it can \
continue with less context. Summarise the transcript below.

Keep, in this order of priority:
1. Decisions and conclusions actually reached, with their reasoning.
2. Concrete facts, numbers, names, file paths, config values, API names.
3. Questions the user asked that are still open.
4. Sources cited (paper titles, arXiv ids, URLs) and what they support.

Drop: greetings, restatements, dead ends that were abandoned, and any tool
plumbing that produced no result. Write in dense bullet points grouped by
topic, in the third person ("The user asked...", "We established..."). Do not
invent anything that is not in the transcript.

Transcript:
---
{transcript}
---

Summary:"""


def _render_transcript(messages: List[Dict[str, Any]]) -> str:
    """Flatten messages into a role-labelled transcript, capped per message."""
    lines: List[str] = []
    for message in messages:
        role = (message.get("role") or "user").upper()
        content = str(message.get("content") or "")
        if len(content) > PER_MESSAGE_CAP:
            content = content[:PER_MESSAGE_CAP] + " ...[truncated]"
        if not content.strip():
            continue
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def split_into_chunks(text: str, size: int = CHUNK_CHARS) -> List[str]:
    """Split a transcript on paragraph boundaries into <= ``size`` chunks."""
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for paragraph in text.split("\n\n"):
        if current_len + len(paragraph) > size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def summarize_transcript(summarize: Callable[[str], str], transcript: str) -> str:
    """Summarise a transcript, chaining chunks for very long conversations."""
    summary = ""
    for index, chunk in enumerate(split_into_chunks(transcript)):
        payload = (
            chunk
            if not summary
            else f"Summary so far:\n{summary}\n\nNext part:\n{chunk}"
        )
        summary = (summarize(payload) or "").strip()
        if not summary:
            raise ValueError(f"Summariser returned nothing for chunk {index}")
    return summary


def plan_compaction(
    messages: List[Dict[str, Any]], keep_last: int = DEFAULT_KEEP_LAST
) -> Optional[Dict[str, Any]]:
    """Split messages into the part to compact and the part to keep.

    Returns None when there is nothing worth compacting. Messages tagged
    ``is_compaction`` are not summarised again — they are folded into the new
    summary's source transcript but never duplicated into it.
    """
    keep_last = max(1, int(keep_last))
    if len(messages) <= keep_last:
        return None

    older = messages[:-keep_last]
    recent = messages[-keep_last:]
    if not any(str(m.get("content") or "").strip() for m in older):
        return None

    return {
        "older": older,
        "recent": recent,
        "older_chars": sum(len(str(m.get("content") or "")) for m in older),
        "recent_chars": sum(len(str(m.get("content") or "")) for m in recent),
        "transcript": _render_transcript(older),
    }


def apply_compaction(
    messages: List[Dict[str, Any]],
    summary: str,
    keep_last: int = DEFAULT_KEEP_LAST,
) -> List[Dict[str, Any]]:
    """Build the new message list: summary message followed by the tail."""
    plan = plan_compaction(messages, keep_last)
    if plan is None:
        return list(messages)
    compacted_messages = plan["older"]
    summary_message = {
        "role": "assistant",
        "content": (
            f"{SUMMARY_HEADER} {len(compacted_messages)} earlier messages were "
            f"compacted to save context.\n\n{summary.strip()}"
        ),
        "is_compaction": True,
        "compacted_messages": len(compacted_messages),
    }
    return [summary_message] + plan["recent"]


def backup_replaced_messages(
    session_dir: Path, session_id: str, plan: Dict[str, Any], summary: str
) -> Optional[Path]:
    """Write the replaced messages to disk so compaction is recoverable."""
    try:
        backup_dir = Path(session_dir) / session_id / "compactions"
        backup_dir.mkdir(parents=True, exist_ok=True)
        path = backup_dir / f"{int(time.time())}.json"
        path.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "compacted_at": time.time(),
                    "replaced_messages": plan["older"],
                    "summary": summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path
    except Exception:  # pragma: no cover - best effort
        logger.warning("Could not back up compacted messages", exc_info=True)
        return None
