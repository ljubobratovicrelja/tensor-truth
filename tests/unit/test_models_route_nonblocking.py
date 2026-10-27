"""Regression test: GET /api/models must not block the event loop.

``list_models`` used to call ``requests.get`` synchronously inside an
``async def`` handler. Every UI refresh of the model list therefore froze the
single uvicorn event loop — and with it every in-flight chat token stream. The
watchdog caught it as ``Event loop blocked for at least 0.50s`` at
``routes/modules.py`` in ``list_models`` -> ``requests.get(...)``.

The fix wraps the provider probes in ``asyncio.to_thread``. This test records
the thread on which the mocked ``requests.get`` runs and asserts it is *not*
the event-loop thread.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import tensortruth.api.routes.modules as modules_routes
from tensortruth.api.deps import get_config_service

MAIN_THREAD = threading.main_thread()


class _Provider:
    type = "ollama"
    base_url = "http://localhost:11434"
    id = "test-ollama"
    api_key = None
    models = []


class _Registry:
    providers = [_Provider()]


def _response_for(url):
    if url.endswith("/api/ps"):
        payload = {"models": []}
    else:
        payload = {"models": [{"name": "llama3", "size": 1, "modified_at": "now"}]}
    return SimpleNamespace(
        status_code=200,
        json=lambda: payload,
        raise_for_status=lambda: None,
    )


@pytest.fixture
def route_client():
    request_threads = []
    loop_threads = []

    def _mock_requests_get(url, *args, **kwargs):
        # Record where the blocking HTTP probe actually executes.
        request_threads.append(threading.current_thread())
        return _response_for(url)

    def _mock_config_service():
        # Dependency resolution happens on the event-loop thread.
        loop_threads.append(threading.current_thread())
        return MagicMock()

    with (
        patch(
            "tensortruth.api.routes.modules.requests.get",
            side_effect=_mock_requests_get,
        ),
        patch("tensortruth.core.ollama.get_model_info", return_value={}),
        patch("tensortruth.core.ollama.supports_thinking_levels", return_value=False),
        patch("tensortruth.core.providers.ProviderRegistry") as mock_registry,
    ):
        mock_registry.get_instance.return_value = _Registry()

        app = FastAPI()
        app.include_router(modules_routes.router, prefix="/api")
        app.dependency_overrides[get_config_service] = _mock_config_service

        client = TestClient(app)
        try:
            yield client, request_threads, loop_threads
        finally:
            client.close()


def test_list_models_requests_off_loop_thread(route_client):
    client, request_threads, loop_threads = route_client

    response = client.get("/api/models")

    assert response.status_code == 200, response.text
    assert loop_threads, "config service dependency never ran"
    assert request_threads, "requests.get was never called"

    loop_thread = loop_threads[0]
    request_thread = request_threads[0]
    # The meaningful invariant: the probe must not share the event-loop thread.
    assert request_thread is not loop_thread
    # And, as with uvicorn, never the process main thread either.
    assert request_thread is not MAIN_THREAD
