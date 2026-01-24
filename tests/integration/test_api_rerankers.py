"""Integration tests for rerankers API endpoints."""

from unittest.mock import MagicMock, patch

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


class TestRerankersAPI:
    """Test rerankers endpoints."""

    @pytest.mark.asyncio
    async def test_list_rerankers(self, client, tmp_path, monkeypatch):
        """Test getting list of configured rerankers."""
        # Create a fresh ConfigService for the test
        from tensortruth.api import deps
        from tensortruth.services import ConfigService

        # Clear the LRU cache to ensure fresh config service
        deps.get_config_service.cache_clear()

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        response = await client.get("/api/rerankers")
        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert "models" in data
        assert "current" in data
        assert isinstance(data["models"], list)

        # Check default models are present
        model_names = [m["model"] for m in data["models"]]
        assert "BAAI/bge-reranker-v2-m3" in model_names

        # Current should match default
        assert data["current"] == "BAAI/bge-reranker-v2-m3"

    @pytest.mark.asyncio
    async def test_add_reranker_success(self, client, tmp_path, monkeypatch):
        """Test adding a new reranker model (mocked validation)."""
        from tensortruth.api import deps
        from tensortruth.services import ConfigService

        # Clear the LRU cache to ensure fresh config service
        deps.get_config_service.cache_clear()

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        # Mock HuggingFace validation to succeed
        mock_model_info = MagicMock()
        with patch("huggingface_hub.model_info", return_value=mock_model_info):
            response = await client.post(
                "/api/rerankers", json={"model": "test/my-reranker"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "added"
        assert data["model"] == "test/my-reranker"

        # Verify it's in the list now
        response = await client.get("/api/rerankers")
        model_names = [m["model"] for m in response.json()["models"]]
        assert "test/my-reranker" in model_names

    @pytest.mark.asyncio
    async def test_add_reranker_already_exists(self, client, tmp_path, monkeypatch):
        """Test adding a reranker that already exists."""
        from tensortruth.services import ConfigService

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        # Add existing default model
        response = await client.post(
            "/api/rerankers", json={"model": "BAAI/bge-reranker-v2-m3"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "added"  # No error, just confirms it exists

    @pytest.mark.asyncio
    async def test_add_reranker_validation_failure(self, client, tmp_path, monkeypatch):
        """Test adding a reranker that doesn't exist on HuggingFace."""
        from tensortruth.services import ConfigService

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        # Mock HuggingFace validation to fail with 404
        with patch(
            "huggingface_hub.model_info",
            side_effect=Exception("404 Client Error"),
        ):
            response = await client.post(
                "/api/rerankers", json={"model": "nonexistent/model"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_add_reranker_empty_name(self, client, tmp_path, monkeypatch):
        """Test adding a reranker with empty name."""
        from tensortruth.services import ConfigService

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        response = await client.post("/api/rerankers", json={"model": ""})
        assert response.status_code == 400

        response = await client.post("/api/rerankers", json={"model": "   "})
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_remove_reranker_success(self, client, tmp_path, monkeypatch):
        """Test removing a reranker."""
        from tensortruth.services import ConfigService

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        # First add a custom reranker
        mock_model_info = MagicMock()
        with patch("huggingface_hub.model_info", return_value=mock_model_info):
            await client.post("/api/rerankers", json={"model": "test/to-remove"})

        # Now remove it
        response = await client.delete("/api/rerankers/test%2Fto-remove")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "removed"

        # Verify it's gone
        response = await client.get("/api/rerankers")
        model_names = [m["model"] for m in response.json()["models"]]
        assert "test/to-remove" not in model_names

    @pytest.mark.asyncio
    async def test_remove_reranker_not_found(self, client, tmp_path, monkeypatch):
        """Test removing a reranker that doesn't exist."""
        from tensortruth.services import ConfigService

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        response = await client.delete("/api/rerankers/nonexistent%2Fmodel")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_remove_default_reranker_fails(self, client, tmp_path, monkeypatch):
        """Test that removing the currently selected default reranker fails."""
        from tensortruth.api import deps
        from tensortruth.services import ConfigService

        # Clear the LRU cache to ensure fresh config service
        deps.get_config_service.cache_clear()

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        def get_test_config_service():
            return test_service

        monkeypatch.setattr(
            "tensortruth.api.deps.get_config_service", get_test_config_service
        )

        # Try to remove the default reranker - should fail
        response = await client.delete("/api/rerankers/BAAI%2Fbge-reranker-v2-m3")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "default" in data["error"].lower()
