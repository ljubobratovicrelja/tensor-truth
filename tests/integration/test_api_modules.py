"""Integration tests for modules API endpoints."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from tensortruth.api.main import create_app


@pytest.fixture
def app():
    """Create test application."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


class TestModulesAPI:
    """Test modules/models/presets endpoints."""

    @pytest.mark.asyncio
    async def test_list_modules_empty(self, client, tmp_path, monkeypatch):
        """Test listing modules when none exist."""
        indexes_dir = tmp_path / "indexes"
        indexes_dir.mkdir()
        monkeypatch.setattr(
            "tensortruth.api.routes.modules.get_indexes_dir", lambda: indexes_dir
        )

        response = await client.get("/api/modules")
        assert response.status_code == 200
        assert response.json()["modules"] == []

    @pytest.mark.asyncio
    async def test_list_modules_with_index(self, client, tmp_path, monkeypatch):
        """Test listing modules when indexes exist."""
        from tensortruth.services import ConfigService

        indexes_dir = tmp_path / "indexes"
        indexes_dir.mkdir()

        # Create a fake index directory with versioned path: indexes/{model_id}/module/
        model_dir = indexes_dir / "bge-m3"
        model_dir.mkdir()
        module_dir = model_dir / "pytorch"
        module_dir.mkdir()
        (module_dir / "chroma.sqlite3").touch()

        monkeypatch.setattr(
            "tensortruth.api.routes.modules.get_indexes_dir", lambda: indexes_dir
        )

        # Mock config service to return default embedding model
        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)
        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", lambda: test_service
        )

        response = await client.get("/api/modules")
        assert response.status_code == 200
        modules = response.json()["modules"]
        assert len(modules) == 1
        assert modules[0]["name"] == "pytorch"

    @pytest.mark.asyncio
    async def test_list_models_ollama_unavailable(self, client, tmp_path, monkeypatch):
        """Test listing models when Ollama is not available."""
        import requests

        def mock_get(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Connection refused")

        monkeypatch.setattr("requests.get", mock_get)

        # Need to also mock the config service
        from tensortruth.services import ConfigService

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)
        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", lambda: test_service
        )

        response = await client.get("/api/models")
        assert response.status_code == 503
        assert "Ollama" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_list_models_success(self, client, tmp_path, monkeypatch):
        """Test listing models when Ollama is available."""
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:7b", "size": 3800000000, "modified_at": "2023-12-01"},
                {
                    "name": "deepseek:8b",
                    "size": 5000000000,
                    "modified_at": "2023-12-02",
                },
            ]
        }

        monkeypatch.setattr("requests.get", lambda *args, **kwargs: mock_response)

        from tensortruth.services import ConfigService

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)
        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", lambda: test_service
        )

        response = await client.get("/api/models")
        assert response.status_code == 200
        models = response.json()["models"]
        assert len(models) == 2
        assert models[0]["name"] == "llama2:7b"

    @pytest.mark.asyncio
    async def test_list_presets_empty(self, client, tmp_path, monkeypatch):
        """Test listing presets when none exist."""
        presets_file = tmp_path / "presets.json"
        monkeypatch.setattr(
            "tensortruth.api.routes.modules.get_presets_file", lambda: presets_file
        )

        response = await client.get("/api/presets")
        assert response.status_code == 200
        assert response.json()["presets"] == []

    @pytest.mark.asyncio
    async def test_list_presets_with_data(self, client, tmp_path, monkeypatch):
        """Test listing presets when data exists."""
        presets_file = tmp_path / "presets.json"
        presets_data = {
            "coding": {"model": "deepseek:8b", "temperature": 0.1},
            "creative": {"model": "llama2:7b", "temperature": 0.9},
        }
        with open(presets_file, "w") as f:
            json.dump(presets_data, f)

        monkeypatch.setattr(
            "tensortruth.api.routes.modules.get_presets_file", lambda: presets_file
        )

        response = await client.get("/api/presets")
        assert response.status_code == 200
        presets = response.json()["presets"]
        assert len(presets) == 2
        preset_names = [p["name"] for p in presets]
        assert "coding" in preset_names
        assert "creative" in preset_names
