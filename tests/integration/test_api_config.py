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
        assert data["ui"]["default_temperature"] == 0.7
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

    @pytest.mark.asyncio
    async def test_get_config_includes_rag_embedding_model(
        self, client, tmp_path, monkeypatch
    ):
        """Test that GET /config returns default_embedding_model in rag section."""
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

        # Verify embedding model is in rag section
        assert "default_embedding_model" in data["rag"]
        assert data["rag"]["default_embedding_model"] == "BAAI/bge-m3"

    @pytest.mark.asyncio
    async def test_get_config_includes_rag_reranker(
        self, client, tmp_path, monkeypatch
    ):
        """Test that GET /config returns default_reranker in rag section."""
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

        # Verify reranker is in rag section (not ui)
        assert "default_reranker" in data["rag"]
        assert data["rag"]["default_reranker"] == "BAAI/bge-reranker-v2-m3"
        assert "default_reranker" not in data["ui"]

    @pytest.mark.asyncio
    async def test_update_config_rag_reranker(self, client, tmp_path, monkeypatch):
        """Test updating default_reranker via rag_default_reranker key."""
        config_file = tmp_path / "config.yaml"

        from tensortruth.services import ConfigService

        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        # Update reranker using rag_ prefix
        response = await client.patch(
            "/api/config",
            json={"updates": {"rag_default_reranker": "BAAI/bge-reranker-base"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rag"]["default_reranker"] == "BAAI/bge-reranker-base"

        # Verify it persisted
        response = await client.get("/api/config")
        assert response.json()["rag"]["default_reranker"] == "BAAI/bge-reranker-base"

    @pytest.mark.asyncio
    async def test_update_config_agent_function_agent_model(
        self, client, tmp_path, monkeypatch
    ):
        """Test updating function_agent_model via API (regression for prefix bug)."""
        config_file = tmp_path / "config.yaml"

        from tensortruth.services import ConfigService

        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        response = await client.patch(
            "/api/config",
            json={"updates": {"agent_function_agent_model": "custom-function:14b"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["agent"]["function_agent_model"] == "custom-function:14b"

        # Verify it persisted
        response = await client.get("/api/config")
        assert response.json()["agent"]["function_agent_model"] == "custom-function:14b"

    @pytest.mark.asyncio
    async def test_api_response_no_agent_reasoning_model(
        self, client, tmp_path, monkeypatch
    ):
        """API should not return agent.reasoning_model (dead code removed)."""
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
        assert "reasoning_model" not in response.json()["agent"]
