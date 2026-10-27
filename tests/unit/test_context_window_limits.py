"""Context-window size round-trip and budget-math tests.

Covers the 512k (524288) and 1M (1048576) token options added to the UI:
they must survive the config PATCH and session PATCH endpoints unchanged
(no clamping) and must produce proportional context budgets.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensortruth.api import deps
from tensortruth.api.routes import config as config_routes
from tensortruth.api.routes import sessions as session_routes
from tensortruth.core.synthesis import CHARS_PER_TOKEN, fit_sources_to_context
from tensortruth.services.config_service import ConfigService
from tensortruth.services.orchestrator_service import (
    CHAT_HISTORY_PCT,
    RESPONSE_BUFFER_PCT,
    OrchestratorService,
)
from tensortruth.services.session_service import SessionService

NEW_CONTEXT_WINDOWS = [524288, 1048576]
EXISTING_CONTEXT_WINDOWS = [
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]


@pytest.fixture
def config_service(tmp_path: Path) -> ConfigService:
    return ConfigService(tmp_path / "config.yaml")


@pytest.fixture
def session_service(tmp_path: Path) -> SessionService:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return SessionService(
        sessions_file=tmp_path / "sessions.json",
        sessions_dir=sessions_dir,
    )


@pytest.fixture
def client(
    config_service: ConfigService, session_service: SessionService
) -> TestClient:
    app = FastAPI()
    app.include_router(config_routes.router, prefix="/api/config")
    app.include_router(session_routes.router, prefix="/api/sessions")
    app.dependency_overrides[deps.get_config_service] = lambda: config_service
    app.dependency_overrides[deps.get_session_service] = lambda: session_service
    return TestClient(app)


@pytest.mark.unit
class TestConfigContextWindowRoundTrip:
    @pytest.mark.parametrize("size", NEW_CONTEXT_WINDOWS)
    def test_patch_config_persists_without_clamping(
        self, client: TestClient, config_service: ConfigService, size: int
    ) -> None:
        response = client.patch(
            "/api/config", json={"updates": {"llm_default_context_window": size}}
        )
        assert response.status_code == 200
        assert response.json()["llm"]["default_context_window"] == size

        # Re-read through a fresh service to bypass the response and cache.
        reloaded = ConfigService(config_service.config_file).load()
        assert reloaded.llm.default_context_window == size

        assert client.get("/api/config").json()["llm"]["default_context_window"] == size

    @pytest.mark.parametrize("size", EXISTING_CONTEXT_WINDOWS)
    def test_existing_sizes_still_round_trip(
        self, client: TestClient, size: int
    ) -> None:
        response = client.patch(
            "/api/config", json={"updates": {"llm_default_context_window": size}}
        )
        assert response.status_code == 200
        assert response.json()["llm"]["default_context_window"] == size


@pytest.mark.unit
class TestSessionContextWindowRoundTrip:
    @pytest.mark.parametrize("size", NEW_CONTEXT_WINDOWS)
    def test_create_and_patch_session_params(
        self,
        client: TestClient,
        session_service: SessionService,
        size: int,
    ) -> None:
        created = client.post(
            "/api/sessions",
            json={
                "modules": [],
                "params": {"model": "test-model", "context_window": size},
            },
        )
        assert created.status_code == 201
        body = created.json()
        assert body["params"]["context_window"] == size
        session_id = body["session_id"]

        # Patch with the other new size to exercise the update path too.
        other = 524288 if size == 1048576 else 1048576
        patched = client.patch(
            f"/api/sessions/{session_id}",
            json={"params": {"model": "test-model", "context_window": other}},
        )
        assert patched.status_code == 200
        assert patched.json()["params"]["context_window"] == other

        # Verify persistence on disk, not just the in-memory cache.
        reloaded = SessionService(
            session_service.legacy_sessions_file, session_service.sessions_dir
        ).load()
        assert reloaded.sessions[session_id]["params"]["context_window"] == other

    def test_patch_session_rejects_nothing_for_512k_and_1m(
        self, client: TestClient
    ) -> None:
        created = client.post("/api/sessions", json={"modules": [], "params": {}})
        assert created.status_code == 201
        session_id = created.json()["session_id"]

        for size in NEW_CONTEXT_WINDOWS:
            response = client.patch(
                f"/api/sessions/{session_id}",
                json={"params": {"context_window": size}},
            )
            assert response.status_code == 200
            assert response.json()["params"]["context_window"] == size


@pytest.mark.unit
class TestContextBudgetMathForLargeWindows:
    @staticmethod
    def _orchestrator(context_window: int) -> OrchestratorService:
        # Only _context_window is used by _budget_history; skip heavy __init__.
        service = object.__new__(OrchestratorService)
        service._context_window = context_window
        return service

    @pytest.mark.parametrize("size", NEW_CONTEXT_WINDOWS)
    def test_history_budget_scales_proportionally(self, size: int) -> None:
        service = self._orchestrator(size)
        total_chars = size * CHARS_PER_TOKEN
        expected_budget = int(total_chars * CHAT_HISTORY_PCT)

        history = [
            {"role": "user", "content": "u" * 1000},
            {"role": "assistant", "content": "a" * 1000},
        ] * 5000  # 10M chars, far beyond the budget

        kept = service._budget_history(history, "system", "prompt")
        kept_chars = sum(len(m["content"]) for m in kept)

        assert 0 < kept_chars <= expected_budget
        assert len(kept) < len(history)

    def test_history_budget_leaves_room_for_response_buffer(self) -> None:
        size = 1048576
        service = self._orchestrator(size)
        total_chars = size * CHARS_PER_TOKEN
        response_buffer_chars = int(total_chars * RESPONSE_BUFFER_PCT)

        # System + prompt consume nearly the whole non-buffer space.
        system = "s" * (total_chars // 2)
        prompt = "p" * (total_chars // 4)
        history = [{"role": "user", "content": "h" * 1000}] * 1000

        remaining = total_chars - len(system) - len(prompt) - response_buffer_chars
        assert remaining < 0  # response buffer alone overruns the remaining space

        kept = service._budget_history(history, system, prompt)
        assert kept == []

    @pytest.mark.parametrize("size", NEW_CONTEXT_WINDOWS)
    def test_source_fitting_budgets_are_proportional(self, size: int) -> None:
        pages = [
            {
                "url": f"https://example.com/{i}",
                "title": f"p{i}",
                "content": "x" * 900000,
            }
            for i in range(3)
        ]
        fitted, allocations = fit_sources_to_context(pages, None, size)

        total_budget = int(size * 0.6 * CHARS_PER_TOKEN)
        per_source_cap = int(size * 0.15 * CHARS_PER_TOKEN)

        assert sum(allocations.values()) <= total_budget
        for page in fitted:
            assert len(page["content"]) <= per_source_cap
