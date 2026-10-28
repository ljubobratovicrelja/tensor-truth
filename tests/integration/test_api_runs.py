"""Integration tests for the server-owned chat run REST API.

The generation coroutine is always a stub — no LLM is ever called.
"""

import asyncio
from typing import Any, Dict, List

import pytest
from httpx import ASGITransport, AsyncClient

from tensortruth.api.main import create_app
from tensortruth.services import chat_run_service as crs


@pytest.fixture
def service(monkeypatch):
    """Fresh run registry for each test."""
    svc = crs.ChatRunService()
    monkeypatch.setattr(crs, "_service", svc)
    return svc


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def mock_session_paths(tmp_path, monkeypatch):
    """Patch session storage to a temp directory."""
    sessions_file = tmp_path / "chat_sessions.json"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    monkeypatch.setattr("tensortruth.api.deps.get_sessions_file", lambda: sessions_file)
    monkeypatch.setattr(
        "tensortruth.api.deps.get_sessions_data_dir", lambda: sessions_dir
    )
    monkeypatch.setattr(
        "tensortruth.api.deps.get_session_dir", lambda sid: sessions_dir / sid
    )

    from tensortruth.api.deps import get_session_service

    get_session_service.cache_clear()

    yield sessions_file, sessions_dir
    get_session_service.cache_clear()


@pytest.fixture
async def session_id(client, mock_session_paths, service) -> str:
    """Create a real session through the API."""
    response = await client.post("/api/sessions", json={"modules": [], "params": {}})
    assert response.status_code in (200, 201), response.text
    sid = response.json()["session_id"]

    sessions_dir = mock_session_paths[1]
    (sessions_dir / sid).mkdir(parents=True, exist_ok=True)
    return sid


def stub_runner(events: List[Dict[str, Any]], delay: float = 0.0):
    """Build a runner that publishes fixed events, optionally slowly."""

    async def _runner(sink) -> None:
        for event in events:
            if delay:
                await asyncio.sleep(delay)
            await sink.send_json(event)

    return _runner


def install_stub(monkeypatch, runner):
    """Make the registry's default runner the stub, whatever the prompt."""
    monkeypatch.setattr(
        crs,
        "make_default_runner",
        lambda session_id, prompt, images=None, **kwargs: runner,
    )


async def _poll_until_terminal(client, run_id, tries=100):
    for _ in range(tries):
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} did not finish: {body}")


class TestRunsAPI:
    """create -> poll -> active -> 409 -> cancel."""

    @pytest.mark.asyncio
    async def test_create_then_poll_events(self, client, session_id, monkeypatch):
        install_stub(
            monkeypatch,
            stub_runner(
                [
                    {"type": "token", "content": "Hel"},
                    {"type": "token", "content": "lo"},
                    {"type": "done", "content": "Hello"},
                ]
            ),
        )

        response = await client.post(
            f"/api/sessions/{session_id}/runs", json={"prompt": "hi"}
        )
        assert response.status_code == 201, response.text
        body = response.json()
        run_id = body["run_id"]
        assert body["session_id"] == session_id
        assert body["prompt"] == "hi"

        final = await _poll_until_terminal(client, run_id)
        assert final["status"] == "completed"
        assert [e["seq"] for e in final["events"]] == [1, 2, 3]
        assert [e["type"] for e in final["events"]] == ["token", "token", "done"]
        assert final["events"][-1]["content"] == "Hello"

        # Cursor replay.
        replay = await client.get(f"/api/runs/{run_id}", params={"after": 2})
        assert replay.status_code == 200
        assert [e["seq"] for e in replay.json()["events"]] == [3]

        late = await client.get(f"/api/runs/{run_id}", params={"after": 3})
        assert late.json()["events"] == []

    @pytest.mark.asyncio
    async def test_active_lookup_and_conflict(self, client, session_id, monkeypatch):
        install_stub(
            monkeypatch, stub_runner([{"type": "done", "content": "ok"}], delay=5)
        )

        response = await client.post(
            f"/api/sessions/{session_id}/runs", json={"prompt": "slow"}
        )
        assert response.status_code == 201, response.text
        run_id = response.json()["run_id"]

        active = await client.get(f"/api/sessions/{session_id}/runs/active")
        assert active.status_code == 200
        active_body = active.json()
        assert active_body is not None
        assert active_body["run_id"] == run_id
        assert active_body["status"] == "running"

        conflict = await client.post(
            f"/api/sessions/{session_id}/runs", json={"prompt": "again"}
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["run_id"] == run_id

    @pytest.mark.asyncio
    async def test_active_lookup_is_null_when_idle(
        self, client, session_id, monkeypatch
    ):
        install_stub(monkeypatch, stub_runner([{"type": "done", "content": "ok"}]))
        response = await client.get(f"/api/sessions/{session_id}/runs/active")
        assert response.status_code == 200
        assert response.json() is None

    @pytest.mark.asyncio
    async def test_cancel(self, client, session_id, monkeypatch):
        install_stub(
            monkeypatch, stub_runner([{"type": "done", "content": "ok"}], delay=5)
        )

        response = await client.post(
            f"/api/sessions/{session_id}/runs", json={"prompt": "slow"}
        )
        run_id = response.json()["run_id"]

        cancelled = await client.post(f"/api/runs/{run_id}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        body = cancelled.json()
        assert body["run_id"] == run_id
        assert body["status"] == "cancelled"

        # Active lookup no longer reports it, and the session can start anew.
        assert (
            await client.get(f"/api/sessions/{session_id}/runs/active")
        ).json() is None

        install_stub(monkeypatch, stub_runner([{"type": "done", "content": "ok"}]))
        again = await client.post(
            f"/api/sessions/{session_id}/runs", json={"prompt": "fresh"}
        )
        assert again.status_code == 201, again.text
        await _poll_until_terminal(client, again.json()["run_id"])

    @pytest.mark.asyncio
    async def test_cancel_unknown_run_404(self, client, session_id):
        response = await client.post("/api/runs/nope/cancel")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_run_and_session_404(self, client, session_id):
        assert (await client.get("/api/runs/nope")).status_code == 404
        response = await client.post(
            "/api/sessions/nonexistent/runs", json={"prompt": "hi"}
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_prompt_rejected(self, client, session_id):
        response = await client.post(
            f"/api/sessions/{session_id}/runs", json={"prompt": "", "images": []}
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_run_exposes_kind_and_empty_metadata(
        self, client, session_id, monkeypatch
    ):
        install_stub(monkeypatch, stub_runner([{"type": "done", "content": "ok"}]))

        response = await client.post(
            f"/api/sessions/{session_id}/runs", json={"prompt": "hi"}
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["kind"] == "chat"
        assert body["metadata"] == {}
        await _poll_until_terminal(client, body["run_id"])

    @pytest.mark.asyncio
    async def test_failed_run_reports_error(self, client, session_id, monkeypatch):
        async def failing_runner(sink):
            await sink.send_json({"type": "token", "content": "partial"})
            raise RuntimeError("model exploded")

        install_stub(monkeypatch, failing_runner)

        response = await client.post(
            f"/api/sessions/{session_id}/runs", json={"prompt": "boom"}
        )
        run_id = response.json()["run_id"]

        final = await _poll_until_terminal(client, run_id)
        assert final["status"] == "failed"
        assert final["error"] == "model exploded"
        assert final["events"][-1]["type"] == "error"


def _seed_messages(session_id: str, messages: List[Dict[str, Any]]) -> None:
    """Write messages into the real session through the API's service."""
    from tensortruth.api.deps import get_session_service

    service = get_session_service()
    data = service.load()
    for message in messages:
        data = service.add_message(session_id, message, data)
    service.save(data)


class TestElaborationRuns:
    """Elaboration run creation and request validation."""

    MESSAGES = [
        {"role": "user", "content": "What is momentum?"},
        {"role": "assistant", "content": "Momentum is mass times velocity."},
    ]

    @pytest.mark.asyncio
    async def test_create_exposes_kind_and_metadata(
        self, client, session_id, monkeypatch
    ):
        _seed_messages(session_id, self.MESSAGES)
        install_stub(monkeypatch, stub_runner([{"type": "done", "content": "ok"}]))

        response = await client.post(
            f"/api/sessions/{session_id}/runs",
            json={
                "kind": "elaboration",
                "message_index": 1,
                "selected_text": "mass times velocity",
                "occurrence_index": 0,
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["kind"] == "elaboration"
        assert body["metadata"] == {
            "message_index": 1,
            "selected_text": "mass times velocity",
            "occurrence_index": 0,
        }
        await _poll_until_terminal(client, body["run_id"])

    @pytest.mark.parametrize(
        "payload",
        [
            {"kind": "elaboration", "message_index": 1, "selected_text": "   "},
            {"kind": "elaboration", "message_index": 1},
            {"kind": "elaboration", "message_index": 9, "selected_text": "x"},
            {"kind": "elaboration", "message_index": -1, "selected_text": "x"},
            {"kind": "elaboration", "message_index": 0, "selected_text": "x"},
            {"kind": "elaboration", "message_index": 1, "selected_text": "x" * 4001},
            {
                "kind": "elaboration",
                "message_index": 1,
                "selected_text": "x",
                "images": [{"data": "abc", "mimetype": "image/png"}],
            },
            {
                "kind": "elaboration",
                "message_index": 1,
                "selected_text": "x",
                "occurrence_index": -1,
            },
            {"kind": "bogus", "prompt": "hi"},
        ],
    )
    @pytest.mark.asyncio
    async def test_invalid_elaboration_requests_are_400(
        self, client, session_id, monkeypatch, payload
    ):
        _seed_messages(session_id, self.MESSAGES)
        install_stub(monkeypatch, stub_runner([{"type": "done", "content": "ok"}]))

        response = await client.post(f"/api/sessions/{session_id}/runs", json=payload)
        assert response.status_code == 400, response.text
