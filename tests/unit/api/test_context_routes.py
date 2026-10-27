"""Route tests for the context breakdown and /compact endpoints."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensortruth.api.deps import (
    get_chat_history_service,
    get_config_service,
    get_session_service,
)
from tensortruth.api.routes import sessions as session_routes
from tensortruth.services import context_accounting as acc
from tensortruth.services.models import SessionData


class FakeSessionService:
    def __init__(self, session, tmp_path):
        self.session = session
        self.sessions_dir = tmp_path
        self.saved = None

    def load(self):
        return SessionData(current_id="s1", sessions={"s1": self.session})

    def get_session(self, session_id, data):
        return data.sessions.get(session_id)

    def save(self, data):
        self.saved = data
        self.session = data.sessions["s1"]


def _session(messages, params=None):
    return {
        "title": "Test",
        "created_at": "2026-01-01",
        "messages": messages,
        "params": params or {"model": "test-model", "context_window": 65536},
    }


@pytest.fixture
def client_factory(tmp_path, monkeypatch):
    def _make(session):
        service = FakeSessionService(session, tmp_path)
        config = SimpleNamespace(history_cleaning=SimpleNamespace(enabled=False))
        chat_history = SimpleNamespace(
            build_history=lambda messages, max_turns=None, apply_cleaning=True: (
                SimpleNamespace(messages=list(messages))
            )
        )

        app = FastAPI()
        app.include_router(session_routes.router, prefix="/api/sessions")
        app.dependency_overrides[get_session_service] = lambda: service
        app.dependency_overrides[get_config_service] = lambda: SimpleNamespace(
            load=lambda: config
        )
        app.dependency_overrides[get_chat_history_service] = lambda: chat_history
        return TestClient(app), service

    yield _make
    acc.clear_all()
    acc._static_overhead.clear()


@pytest.mark.unit
class TestContextEndpoint:
    def test_measured_snapshot_is_returned(self, client_factory):
        client, _ = client_factory(_session([{"role": "user", "content": "hi"}]))
        acc.record_snapshot(
            "s1",
            {"system": 800, "tools": 4000, "history": 400, "prompt": 40},
            context_window=65536,
            messages_included=1,
            messages_total=1,
        )
        body = client.get("/api/sessions/s1/context").json()
        assert body["measured"] is True
        assert body["total_chars"] == 5240
        assert body["context_window"] == 65536
        assert {c["id"] for c in body["categories"]} == set(acc.CATEGORIES)

    def test_falls_back_to_live_estimate(self, client_factory):
        client, _ = client_factory(
            _session(
                [
                    {"role": "user", "content": "x" * 400},
                    {"role": "assistant", "content": "y" * 400},
                ]
            )
        )
        acc.record_static_overhead({"system": 500, "tools": 1000, "modules": 0})
        body = client.get("/api/sessions/s1/context").json()
        assert body["measured"] is False
        cats = {c["id"]: c for c in body["categories"]}
        assert cats["history"]["chars"] == 800
        # static overhead is reused from the last measurement elsewhere
        assert cats["tools"]["chars"] == 1000
        assert cats["system"]["chars"] == 500
        assert cats["prompt"]["chars"] == 0

    def test_unknown_session_404(self, client_factory):
        client, _ = client_factory(_session([]))
        assert client.get("/api/sessions/nope/context").status_code == 404


@pytest.mark.unit
class TestCompactEndpoint:
    def _fake_llm(self, monkeypatch, text="SUMMARY TEXT"):
        monkeypatch.setattr(
            "tensortruth.core.providers.resolve_model_from_params",
            lambda params, name: SimpleNamespace(model_name=name),
        )

        class _Completion:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return self.value

        monkeypatch.setattr(
            "tensortruth.core.providers.create_llm",
            lambda ref, **kw: SimpleNamespace(
                complete=lambda prompt: _Completion(text)
            ),
        )

    def test_compacts_and_backs_up(self, client_factory, monkeypatch):
        self._fake_llm(monkeypatch)
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}" * 20}
            for i in range(10)
        ]
        client, service = client_factory(_session(messages))
        acc.record_snapshot("s1", {"system": 100})

        body = client.post("/api/sessions/s1/compact", json={"keep_last": 4}).json()

        assert body["compacted"] is True
        assert body["messages_before"] == 10
        assert body["messages_after"] == 5
        assert body["messages_compacted"] == 6
        assert body["tokens_saved"] > 0
        assert body["backup_path"]

        new_messages = service.saved.sessions["s1"]["messages"]
        assert new_messages[0]["is_compaction"] is True
        assert "SUMMARY TEXT" in new_messages[0]["content"]
        assert len(new_messages) == 5
        # snapshot invalidated so the meter is re-measured on the next turn
        assert acc.report_for_session("s1") is None

    def test_nothing_to_compact(self, client_factory, monkeypatch):
        self._fake_llm(monkeypatch)
        client, service = client_factory(
            _session([{"role": "user", "content": "only one"}])
        )
        body = client.post("/api/sessions/s1/compact", json={}).json()
        assert body["compacted"] is False
        assert "Nothing to compact" in body["message"]
        assert service.saved is None

    def test_no_model_returns_503(self, client_factory, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("no provider")

        monkeypatch.setattr(
            "tensortruth.core.providers.resolve_model_from_params", _boom
        )
        client, _ = client_factory(
            _session([{"role": "user", "content": f"m{i}"} for i in range(10)])
        )
        response = client.post("/api/sessions/s1/compact", json={"keep_last": 2})
        assert response.status_code == 503
        assert "No language model available" in response.json()["detail"]
