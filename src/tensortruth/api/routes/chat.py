"""Chat endpoints including the WebSocket run-attach stream.

Generation no longer happens inside the socket handler: a turn is a server-owned
*run* (see ``docs/RUNS.md``). The WebSocket is a thin attach loop that replays a
run's event log from a client-supplied cursor and live-tails it. Closing the
socket only ends the subscription — the producer keeps running.
"""

import asyncio
import json
import logging
from typing import Any, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from tensortruth.api.deps import (
    ChatServiceDep,
    ProjectServiceDep,
    SessionServiceDep,
    get_chat_service,
    get_project_service,
    get_session_service,
)
from tensortruth.api.schemas import (
    ChatRequest,
    ChatResponse,
    SourceNode,
)
from tensortruth.services.chat_run_service import (
    ChatRun,
    RunConflictError,
    get_chat_run_service,
)

# ChatContext, the orchestrator helpers and the turn body used to live here;
# they now live in the runner service and are re-exported for callers/tests.
from tensortruth.services.chat_runner import (  # noqa: F401
    ChatContext,
    _is_orchestrator_enabled,
    _localize_images,
    _run_orchestrator_path,
    _save_assistant_message,
    run_turn,
)

logger = logging.getLogger(__name__)

# REST endpoints (mounted under /api)
rest_router = APIRouter()
# WebSocket endpoint (mounted at root, not under /api)
ws_router = APIRouter()
# Legacy alias for backwards compatibility
router = rest_router


@rest_router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    session_service: SessionServiceDep,
    chat_service: ChatServiceDep,
    project_service: ProjectServiceDep,
) -> ChatResponse:
    """Non-streaming chat endpoint."""
    # 1. Validate session
    data = session_service.load()
    session = session_service.get_session(session_id, data)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # 2. Build context
    from tensortruth.api.deps import get_document_service

    with get_document_service(session_id, "session") as pdf_service:
        context = ChatContext.from_session(
            session_id,
            body.prompt,
            session,
            pdf_service,
            project_service=project_service,
        )

    # 3. Save user message
    data = session_service.add_message(
        session_id, {"role": "user", "content": body.prompt}, data
    )
    session_service.save(data)

    # 4. Execute query via ChatService
    result = chat_service.execute(
        prompt=context.prompt,
        modules=context.modules,
        params=context.params,
        session_messages=context.session_messages,
        additional_index_paths=context.additional_index_paths,
    )

    # 5. Save assistant message
    _save_assistant_message(
        session_service, session_id, result.response, result.sources, result.metrics
    )

    # Convert source dicts to SourceNode models for response
    sources = [SourceNode(**s) for s in result.sources]

    return ChatResponse(
        content=result.response,
        sources=sources,
        confidence_level="normal",
        metrics=result.metrics,
    )


async def _send_protocol_error(websocket: WebSocket, detail: str) -> None:
    """Send a socket-level error (not part of any run's log).

    ``seq`` is 0: it belongs to no run, and clients dedupe against a run cursor.
    """
    await websocket.send_json({"type": "error", "detail": detail, "seq": 0})


async def _stream_run(
    websocket: WebSocket,
    run: ChatRun,
    after: int,
) -> None:
    """Replay a run from ``after`` then live-tail until it closes.

    Exiting this coroutine only unsubscribes; it cannot stop the run's task.
    """
    try:
        async for event in get_chat_run_service().subscribe(run, after):
            await websocket.send_json(event)
    except asyncio.CancelledError:
        raise
    except Exception:
        # Socket is gone or errored: drop the subscription quietly.
        logger.debug("Run stream for %s ended on send error", run.run_id)


async def _resolve_attachment(
    raw: Any,
    session_id: str,
) -> Optional[Tuple[ChatRun, int]]:
    """Turn an inbound WS message into ``(run, after)``, or ``None``.

    Supports ``{"type": "start", ...}``, ``{"type": "attach", ...}`` and the
    legacy bare ``{"prompt": ...}`` payload (which creates a run). Protocol
    errors return ``None``; the caller reports them.
    """
    if isinstance(raw, str):
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            message = {"prompt": raw}
    else:  # pragma: no cover - receive_text always yields str
        message = {"prompt": str(raw)}

    if not isinstance(message, dict):
        message = {"prompt": str(message)}

    kind = message.get("type")
    service = get_chat_run_service()

    if kind == "attach":
        run_id = message.get("run_id")
        after = message.get("after", 0)
        if not isinstance(run_id, str) or not run_id:
            return None
        try:
            after = int(after)
        except (TypeError, ValueError):
            after = 0
        run = service.get(run_id)
        if run is None or run.session_id != session_id:
            return None
        return run, max(after, 0)

    # "start" and legacy bare prompts both create a run.
    prompt = message.get("prompt", "")
    images = message.get("images") or []
    if not isinstance(prompt, str):
        prompt = str(prompt)
    if not isinstance(images, list):
        images = []
    if not prompt and not images:
        return None
    if not prompt:
        prompt = "Describe this image."

    run = await service.create(session_id, prompt, images)
    # No synthetic "run_started" frame: a legacy client reads the next event as
    # the answer to its prompt (its first frame may be `error`), and the
    # run_id is never needed on this path — clients that want a handle create
    # the run through POST /api/sessions/{id}/runs and attach with it.
    return run, 0


@ws_router.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    session_service=Depends(get_session_service),
    chat_service=Depends(get_chat_service),
    project_service=Depends(get_project_service),
) -> None:
    """WebSocket attach loop for server-owned chat runs.

    Protocol:
    - Client sends ``{"type": "start", "prompt": ..., "images": [...]}`` to
      create a run, or ``{"type": "attach", "run_id": ..., "after": N}`` to
      resume one. A bare ``{"prompt": ...}`` payload is treated as ``start``.
    - Server sends the run's events, each carrying its integer ``seq``.
    """
    await websocket.accept()

    data = session_service.load()
    if session_service.get_session(session_id, data) is None:
        await _send_protocol_error(websocket, "Session not found")
        await websocket.close(code=1008)
        return

    pump: Optional["asyncio.Task[None]"] = None
    reader: Optional["asyncio.Task[str]"] = None

    async def _cancel(task: Optional["asyncio.Task[Any]"]) -> None:
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    try:
        reader = asyncio.create_task(websocket.receive_text())
        while True:
            waiters: set["asyncio.Task[Any]"] = {reader}
            if pump is not None:
                waiters.add(pump)
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)

            if pump is not None and pump in done:
                error = pump.exception()
                pump = None
                if error is not None:
                    # The socket died mid-stream; the run keeps going.
                    break

            if reader not in done:
                continue

            try:
                raw = reader.result()
            except WebSocketDisconnect:
                break
            except Exception:
                break
            reader = None

            try:
                attachment = await _resolve_attachment(raw, session_id)
            except RunConflictError as exc:
                await _send_protocol_error(
                    websocket,
                    f"Session already has an active run: {exc.run_id}",
                )
                attachment = None
            except Exception as exc:
                logger.exception("Failed to start/attach run")
                await _send_protocol_error(websocket, str(exc))
                attachment = None

            if attachment is None:
                await _send_protocol_error(websocket, "Invalid or unknown run")
                reader = asyncio.create_task(websocket.receive_text())
                continue

            run, after = attachment
            await _cancel(pump)
            pump = asyncio.create_task(_stream_run(websocket, run, after))
            reader = asyncio.create_task(websocket.receive_text())

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("WebSocket handler error")
        try:
            await _send_protocol_error(websocket, str(exc))
        except Exception:
            pass
    finally:
        # Cancel only *our* subscription/reader tasks. The run's generation
        # task is owned by the registry and is deliberately left alone.
        await _cancel(pump)
        await _cancel(reader)
