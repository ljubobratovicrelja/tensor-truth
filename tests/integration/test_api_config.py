"""Integration tests for config API endpoints."""

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


class TestConfigAPI:
    """Test config endpoints."""

    @pytest.mark.asyncio
    async def test_get_config(self, client, tmp_path, monkeypatch):
        """Test getting current configuration."""
        monkeypatch.setattr(
            "tensortruth.api.deps.ConfigService.__init__",
            lambda self, config_file=None: setattr(
                self, "config_file", tmp_path / "config.yaml"
            )
            or setattr(self, "config_dir", tmp_path),
        )
        from tensortruth.api.deps import get_config_service

        get_config_service.cache_clear()

        response = await client.get("/api/config")
        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert "ollama" in data
        assert "ui" in data
        assert "rag" in data
        assert "models" in data
        assert "agent" in data

        # Check some default values
        assert "base_url" in data["ollama"]
        assert "default_temperature" in data["ui"]

    @pytest.mark.asyncio
    async def test_get_default_config(self, client):
        """Test getting default configuration."""
        response = await client.get("/api/config/defaults")
        assert response.status_code == 200
        data = response.json()

        # Check default values
        assert data["ollama"]["base_url"] == "http://localhost:11434"
        assert data["ui"]["default_temperature"] == 0.1
        assert data["agent"]["max_iterations"] == 10

    @pytest.mark.asyncio
    async def test_update_config(self, client, tmp_path, monkeypatch):
        """Test updating configuration."""
        # Use temp config file
        config_file = tmp_path / "config.yaml"

        # Create a fresh ConfigService for the test
        from tensortruth.services import ConfigService

        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        # Update a value
        response = await client.patch(
            "/api/config",
            json={"updates": {"ollama_timeout": 600}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ollama"]["timeout"] == 600

        # Verify it persisted
        response = await client.get("/api/config")
        assert response.json()["ollama"]["timeout"] == 600
