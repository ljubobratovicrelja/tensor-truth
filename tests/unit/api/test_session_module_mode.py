"""`module_mode` defaults and legacy-session fallbacks for the session API."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensortruth.api import deps
from tensortruth.api.routes import sessions as session_routes
from tensortruth.services.config_service import ConfigService
from tensortruth.services.session_service import SessionData, SessionService


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
    app.include_router(session_routes.router, prefix="/api/sessions")
    app.dependency_overrides[deps.get_config_service] = lambda: config_service
    app.dependency_overrides[deps.get_session_service] = lambda: session_service
    return TestClient(app)


@pytest.mark.unit
def test_create_without_module_mode_defaults_to_auto(client: TestClient) -> None:
    response = client.post("/api/sessions", json={"modules": [], "params": {}})

    assert response.status_code == 201, response.text
    assert response.json()["module_mode"] == "auto"


@pytest.mark.unit
def test_create_with_explicit_manual_mode_is_respected(client: TestClient) -> None:
    response = client.post(
        "/api/sessions",
        json={"modules": ["papers_dl"], "module_mode": "manual", "params": {}},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["module_mode"] == "manual"
    assert body["modules"] == ["papers_dl"]


@pytest.mark.unit
def test_legacy_session_without_module_mode_reads_as_auto(
    client: TestClient, session_service: SessionService
) -> None:
    session_service.save(
        SessionData(
            current_id="legacy",
            sessions={
                "legacy": {
                    "title": "Legacy",
                    "created_at": "2025-01-01",
                    "messages": [],
                    "modules": ["papers_dl"],
                    "params": {},
                }
            },
        )
    )

    response = client.get("/api/sessions/legacy")

    assert response.status_code == 200, response.text
    assert response.json()["module_mode"] == "auto"
