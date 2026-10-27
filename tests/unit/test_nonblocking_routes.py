"""Regression tests: document routes/tools must not block the event loop.

Both the ``PATCH .../context`` route and the ``add_arxiv_paper`` agent tool
used to call ``DocumentService`` synchronously inside ``async def`` code. A
50-page PDF therefore stalled the single uvicorn loop for ~16s. These tests
record the thread each blocking service call runs on and fail if it shares the
event-loop thread.

Note on ``TestClient``: Starlette runs the ASGI app on a *portal* thread, not
the process main thread. Under uvicorn both are the same thread, which is why
the watchdog dumps ``threading.main_thread()``. Here we record the loop thread
via an inline dependency call and assert the service work left it.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import tensortruth.api.routes.documents as documents_routes
from tensortruth.api.deps import get_session_service

MAIN_THREAD = threading.main_thread()


class _SessionData:
    def __init__(self, ids):
        self.sessions = {session_id: {} for session_id in ids}


class _SessionService:
    """Records the thread that runs the route's inline dependency call."""

    def __init__(self, loop_threads):
        self._loop_threads = loop_threads

    def load(self):
        self._loop_threads.append(threading.current_thread())
        return _SessionData(["test-session"])


def _mock_document_service(threads):
    """DocumentService stand-in that records the calling thread of blocking calls."""
    service = MagicMock()

    def _set_in_context(doc_id, in_context=True):
        threads.append(threading.current_thread())
        return 4000

    def _upload(content, filename):
        threads.append(threading.current_thread())
        return SimpleNamespace(
            pdf_id="pdf_test",
            filename=filename,
            file_size=len(content),
            page_count=1,
        )

    service.set_in_context.side_effect = _set_in_context
    service.upload.side_effect = _upload
    return service


def _context_manager(service):
    ctx = MagicMock()
    ctx.__enter__.return_value = service
    ctx.__exit__.return_value = False
    return ctx


@pytest.fixture
def route_client(monkeypatch):
    """Minimal app exposing the documents router, with recorded threads."""
    service_threads = []
    loop_threads = []

    service = _mock_document_service(service_threads)
    monkeypatch.setattr(
        documents_routes,
        "get_document_service",
        lambda *args, **kwargs: _context_manager(service),
    )

    app = FastAPI()
    app.include_router(documents_routes.router, prefix="/api")
    app.dependency_overrides[get_session_service] = lambda: _SessionService(
        loop_threads
    )

    client = TestClient(app)
    try:
        yield client, service_threads, loop_threads
    finally:
        client.close()


def test_attach_to_context_runs_off_the_loop_thread(route_client):
    client, service_threads, loop_threads = route_client

    response = client.patch(
        "/api/sessions/test-session/documents/doc_123/context",
        json={"in_context": True},
    )

    assert response.status_code == 200, response.text
    assert loop_threads, "session_service.load() never ran"
    assert len(service_threads) == 1, service_threads

    loop_thread = loop_threads[0]
    worker_thread = service_threads[0]
    # The meaningful invariant: never the event-loop thread.
    assert worker_thread is not loop_thread
    # And, as with uvicorn, never the process main thread either.
    assert worker_thread is not MAIN_THREAD


def test_upload_pdf_runs_off_the_loop_thread(route_client):
    client, service_threads, loop_threads = route_client

    response = client.post(
        "/api/sessions/test-session/documents/upload",
        files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 201, response.text
    assert loop_threads, "session_service.load() never ran"
    assert len(service_threads) == 1, service_threads
    assert service_threads[0] is not loop_threads[0]
    assert service_threads[0] is not MAIN_THREAD


# ---------------------------------------------------------------------------
# add_arxiv_paper agent tool
# ---------------------------------------------------------------------------


def _install_fake_arxiv(monkeypatch):
    """Avoid network access: return one fake paper with a tiny PDF."""
    import arxiv

    class _Paper:
        title = "Test Paper"
        authors = [SimpleNamespace(name="Author One")]

        def download_pdf(self, dirpath=None, filename="paper.pdf"):
            from pathlib import Path

            Path(dirpath, filename).write_bytes(b"%PDF-1.4 fake")

    class _Search:
        def __init__(self, id_list=None):
            self.id_list = id_list

        def results(self):
            return iter([_Paper()])

    monkeypatch.setattr(arxiv, "Search", _Search)


@pytest.mark.asyncio
async def test_add_arxiv_paper_offloads_document_service_calls(monkeypatch):
    """upload() and set_in_context() must run on worker threads, not the loop."""
    import tensortruth.api.deps as deps
    import tensortruth.services.orchestrator_tool_wrappers as wrappers
    from tensortruth.services.orchestrator_tool_wrappers import (
        create_add_arxiv_paper_tool,
    )

    _install_fake_arxiv(monkeypatch)

    service_threads = []
    service = _mock_document_service(service_threads)
    monkeypatch.setattr(
        deps, "get_document_service", lambda *args, **kwargs: _context_manager(service)
    )

    async def _approve(**kwargs):
        return "approved"

    monkeypatch.setattr(wrappers, "request_confirmation", _approve)

    async def _noop_queue(*args, **kwargs):
        return None

    monkeypatch.setattr(wrappers, "_queue_index_build", _noop_queue)

    tool = create_add_arxiv_paper_tool(
        progress_emitter=MagicMock(),
        confirmation_service=MagicMock(),
        session_id="test-session",
    )

    result = str(await tool.acall(arxiv_id="2301.00001", destination="context"))

    assert "Successfully added arXiv paper" in result
    # upload (in _store_paper) + set_in_context (in _attach_to_context)
    assert len(service_threads) == 2, service_threads
    for thread in service_threads:
        assert thread is not MAIN_THREAD
