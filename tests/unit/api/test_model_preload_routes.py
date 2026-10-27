"""Route tests for the model preload status / trigger endpoints.

The preload manager is faked: no model work happens.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensortruth.api.routes import system as system_routes


def _status_payload(state="idle"):
    return {
        "embedder": {
            "kind": "embedder",
            "state": state,
            "model_name": "BAAI/bge-m3",
            "device": "cuda",
            "step": "warm-up" if state == "loading" else "ready",
            "started_at": 100.0,
            "finished_at": None if state == "loading" else 102.5,
            "elapsed_seconds": 2.5,
            "error": None,
        },
        "reranker": {
            "kind": "reranker",
            "state": state,
            "model_name": "BAAI/bge-reranker-v2-m3",
            "device": "cuda",
            "step": "loading weights" if state == "loading" else "ready",
            "started_at": 102.5,
            "finished_at": None if state == "loading" else 104.0,
            "elapsed_seconds": 1.5,
            "error": None,
        },
        "any_loading": state == "loading",
        "all_ready": state == "ready",
    }


@pytest.fixture
def client(monkeypatch):
    fake_manager = MagicMock()
    fake_manager.status.return_value = _status_payload("idle")
    fake_manager.start.return_value = _status_payload("loading")
    monkeypatch.setattr(
        "tensortruth.services.model_preload.get_preload_manager",
        lambda: fake_manager,
    )

    app = FastAPI()
    app.include_router(system_routes.router, prefix="/api/system")
    return TestClient(app), fake_manager


@pytest.mark.unit
class TestModelPreloadRoutes:
    def test_status_endpoint_returns_per_model_state(self, client):
        test_client, fake_manager = client

        response = test_client.get("/api/system/model-preload/status")

        assert response.status_code == 200
        body = response.json()
        assert body["embedder"]["state"] == "idle"
        assert body["embedder"]["model_name"] == "BAAI/bge-m3"
        assert body["reranker"]["kind"] == "reranker"
        assert body["any_loading"] is False
        assert body["all_ready"] is False
        fake_manager.status.assert_called_once_with()
        fake_manager.start.assert_not_called()

    def test_trigger_endpoint_starts_preload_and_returns_state(self, client):
        test_client, fake_manager = client

        response = test_client.post("/api/system/model-preload")

        assert response.status_code == 200
        body = response.json()
        assert body["any_loading"] is True
        assert body["embedder"]["step"] == "warm-up"
        fake_manager.start.assert_called_once_with()

    def test_error_state_is_serialized(self, client):
        test_client, fake_manager = client
        payload = _status_payload("error")
        payload["embedder"]["error"] = "CUDA out of memory"
        fake_manager.status.return_value = payload

        response = test_client.get("/api/system/model-preload/status")

        assert response.status_code == 200
        assert response.json()["embedder"]["error"] == "CUDA out of memory"
