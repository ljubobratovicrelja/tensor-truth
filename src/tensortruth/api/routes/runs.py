"""REST endpoints for server-owned chat runs.

A run is created here, executes as a background asyncio task owned by the run
registry, and is pollable with a ``seq`` cursor. See ``docs/RUNS.md``.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from tensortruth.api.deps import SessionServiceDep
from tensortruth.services.chat_run_service import (
    RunConflictError,
    get_chat_run_service,
)
from tensortruth.services.elaboration import MAX_SELECTED_TEXT_LENGTH

logger = logging.getLogger(__name__)

router = APIRouter()


class StartRunRequest(BaseModel):
    """Body for creating a run.

    ``kind="elaboration"`` starts a passage-explanation run: ``prompt`` is
    unused and the target is ``message_index`` plus ``selected_text``.
    """

    prompt: str = ""
    images: List[Dict[str, Any]] = []
    kind: str = "chat"
    message_index: Optional[int] = None
    selected_text: Optional[str] = None
    occurrence_index: int = 0


def _elaboration_metadata(
    body: StartRunRequest, session: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate an elaboration request and return its run metadata.

    Raises HTTPException(400) when the target message or selection is invalid.
    ``occurrence_index`` is opaque display metadata but must be non-negative.
    """
    if body.kind not in ("chat", "elaboration"):
        raise HTTPException(status_code=400, detail=f"Unknown run kind: {body.kind}")

    if body.kind == "chat":
        return {}

    if body.images:
        raise HTTPException(
            status_code=400, detail="Images are not supported in elaboration mode"
        )

    selected_text = (body.selected_text or "").strip()
    if not selected_text:
        raise HTTPException(status_code=400, detail="Empty selected_text")
    if len(selected_text) > MAX_SELECTED_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"selected_text exceeds {MAX_SELECTED_TEXT_LENGTH} characters",
        )

    if body.message_index is None:
        raise HTTPException(status_code=400, detail="message_index is required")

    messages = session.get("messages") or []
    if not 0 <= body.message_index < len(messages):
        raise HTTPException(status_code=400, detail="message_index out of range")
    if messages[body.message_index].get("role") != "assistant":
        raise HTTPException(
            status_code=400, detail="message_index must reference an assistant message"
        )

    if body.occurrence_index < 0:
        raise HTTPException(
            status_code=400, detail="occurrence_index must be non-negative"
        )

    return {
        "message_index": body.message_index,
        "selected_text": selected_text,
        "occurrence_index": body.occurrence_index,
    }


def _run_payload(run: Any, after: int = 0) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "session_id": run.session_id,
        "prompt": run.prompt,
        "kind": run.kind,
        "metadata": run.metadata,
        "status": run.status,
        "error": run.error,
        "events": run.snapshot(after),
    }


@router.post("/sessions/{session_id}/runs", status_code=201)
async def create_run(
    session_id: str,
    body: StartRunRequest,
    session_service: SessionServiceDep,
) -> Dict[str, Any]:
    """Create a run for a session and schedule its generation task.

    Returns 404 if the session is unknown and 409 if the session already has an
    active run.
    """
    data = session_service.load()
    session = session_service.get_session(session_id, data)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = _elaboration_metadata(body, session)

    if body.kind == "chat" and not body.prompt and not body.images:
        raise HTTPException(status_code=400, detail="Empty prompt")

    service = get_chat_run_service()
    try:
        run = await service.create(
            session_id,
            body.prompt,
            body.images,
            kind=body.kind,
            metadata=metadata,
        )
    except RunConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "detail": "Session already has an active run",
                "run_id": exc.run_id,
            },
        ) from exc

    return _run_payload(run, after=0)


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    after: int = Query(0, ge=0, description="Return events with seq > after"),
) -> Dict[str, Any]:
    """Return run status and events with ``seq > after``."""
    run = get_chat_run_service().get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_payload(run, after=after)


@router.get("/sessions/{session_id}/runs/active")
async def get_active_run(session_id: str) -> Optional[Dict[str, Any]]:
    """Return the latest non-terminal run with all events, or JSON ``null``."""
    run = get_chat_run_service().active_for_session(session_id)
    if run is None:
        return None
    return _run_payload(run, after=0)


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> Dict[str, Any]:
    """Cooperatively cancel a run. Cancelling a finished run is a no-op."""
    run = await get_chat_run_service().cancel(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_payload(run, after=0)
