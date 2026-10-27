# Server-owned chat runs

## Problem

Today the *client* owns the generation. `websocket_chat` in
`src/tensortruth/api/routes/chat.py` receives a prompt and then drives the whole turn inline in
the socket handler: it iterates `orchestrator.execute(...)` and pushes every event through
`await websocket.send_json(...)`. When the tab reloads, the socket dies, the handler's loop
breaks / the coroutine is cancelled, and the half-finished turn is lost — tokens already paid
for, RAG work already done, nothing in `session.json`.

The browser must be a *renderer*, not a participant.

## Design

A chat turn becomes a **run**: a server-owned background task that survives client disconnect,
with an append-only, sequence-numbered event log per run. Clients *attach* to a run and replay
from a cursor. This is the same shape as OpenAI Assistants threads/runs, LangGraph's
`join_run_stream` and TanStack AI resumable streams: durable run + ordered log + client-supplied
cursor (`Last-Event-ID` semantics), so a reconnect replays without duplicating and without
re-running the model.

```
POST /api/sessions/{id}/runs          -> create run, schedule asyncio task, return run_id
GET  /api/runs/{run_id}?after=N       -> status + events with seq > N   (poll / hydrate)
GET  /api/sessions/{id}/runs/active   -> latest non-terminal run + all events (tab reload)
POST /api/runs/{run_id}/cancel        -> cooperative cancel of the task
WS   /ws/chat/{id}                    -> attach + live tail (same event log)
```

### Event envelope

Every event pushed to the client is the payload used today, plus a monotonic `seq`:

```json
{"seq": 7, "type": "token", "content": "Hel"}
```

`seq` starts at 1 per run and is strictly increasing. A client that has applied `seq = N` sends
`after = N` and never sees an event twice; duplicate delivery is harmless because the client
drops `seq <= lastApplied`.

Terminal events: the existing `done` (success) and `error` (failure). No new terminal type — the
frontend already handles both.

### Ownership / lifetime rules

- The generation task is created with `asyncio.create_task` on the server loop and held by the
  run registry, **not** by the WebSocket handler. Handler cancellation must not touch it.
- Run state is in memory (single-process app). A browser reload is invisible to it. A *server*
  restart loses in-flight runs — acceptable and explicitly out of scope; the run registry marks
  such runs gone on startup and the client falls back to normal behaviour.
- One active run per session: a second `POST` while a run is running returns `409`.
- Finished runs are retained (with their events) for 10 minutes for late reconnects, then GC'd.
  Event log capped at 20,000 events per run (then oldest dropped, cursor still monotonic).
- Assistant messages are persisted to `session.json` exactly as today, by the runner.

## Implementation map

### Backend

1. `src/tensortruth/services/chat_run_service.py` (new)
   - `ChatRun`: `run_id`, `session_id`, `prompt`, `status` (`running|completed|failed|cancelled`),
     `events: list[dict]`, `error`, `created_at`, `finished_at`, `_subscribers`.
     `publish(payload) -> seq`, `snapshot(after: int) -> list[dict]`, `close(status, error=None)`.
   - `ChatRunService`: `create(...)`, `get(run_id)`, `active_for_session(session_id)`,
     `cancel(run_id)`, `subscribe(run, after)` (async generator: replay then live, exits cleanly
     when the run closes), retention GC. Module-level singleton `get_chat_run_service()`.
   - `subscribe` must never raise into the caller when the reader disappears; delivery to a dead
     subscriber is dropped, generation continues.
2. `src/tensortruth/services/chat_runner.py` (new)
   - Move the per-prompt body of `websocket_chat` (image persistence, `/command` dispatch,
     orchestrator path, direct `ChatService` fallback, assistant-message save, title generation)
     into `run_turn(sink, ...)` unchanged in behaviour.
   - `sink` is a duck-typed socket with `async send_json(dict)` — reused verbatim with the
     existing `_run_orchestrator_path(websocket=...)` and `ToolCommand.execute(..., websocket)`.
   - `RunEventSink(run)` publishes into the run instead of a socket.
3. `src/tensortruth/api/routes/runs.py` (new, mounted under `/api`) — the four REST endpoints.
4. `chat.py` — keep `ChatContext` and `_run_orchestrator_path`; `websocket_chat` becomes a thin
   attach loop: `start` / `attach` / legacy bare `{"prompt": ...}` (creates a run for backward
   compatibility). The POST `/api/sessions/{id}/chat` non-streaming route is unchanged.

### Frontend

5. `frontend/src/api/runs.ts` (new): `startRun`, `getActiveRun`, `getRunEvents`, `cancelRun` via
   `apiGet`/`apiPost`, types in `api/types.ts`.
6. `frontend/src/hooks/useWebSocket.ts`:
   - Extract the `switch (data.type)` handling into one shared, pure
     `applyStreamMessage(data, ctx)` so live and replayed events go through identical code.
   - `sendMessage`: `POST` the run first (so the turn exists server-side even if the socket never
     connects), then open the socket and send `{"type":"attach","run_id":...,"after":0}`.
     Track `lastAppliedSeq`; on socket close with the run still running, reconnect and attach from
     the cursor (a few retries), so a flaky socket no longer loses tokens either.
   - On mount / session change: `getActiveRun(session_id)`; if one is running, hydrate
     (`startStreaming(run.prompt)` + replay its events through `applyStreamMessage`) and attach.
   - `cancelStreaming` calls `POST /api/runs/{run_id}/cancel` in addition to closing the socket.

### Tests

- `tests/unit/test_chat_run_service.py`: sequencing, snapshot/replay from cursor, subscriber
  disconnect does not stop the producer, late re-attach gets the full log, cancel sets status,
  one-active-run rule, terminal status.
- `tests/integration/test_api_runs.py`: create → poll events → active lookup → 409 on second
  create → cancel, with a stubbed runner coroutine (no LLM).
- `frontend/src/hooks/useWebSocket.resume.test.tsx`: reload hydration replays a run's events and
  does not duplicate on re-attach; `seq` dedupe.

## Non-goals

- No persistence of runs across server restarts.
- No change to retrieval, tool semantics, or `session.json` format.
- No SSE endpoint: WebSocket attach + REST polling cover it, and the app already speaks WS.

## Gates (must be green before this branch is considered done)

```bash
./scripts/test.sh          # unit + integration
./scripts/lint.sh          # black/isort/flake8/mypy + tsc -b/eslint/prettier
```
