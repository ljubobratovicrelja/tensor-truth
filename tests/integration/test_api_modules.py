"""Integration tests for modules API endpoints."""

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
    """Test modules/models endpoints."""

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
    async def test_list_modules_with_index(self, app, tmp_path, monkeypatch):
        """Test listing modules when indexes exist."""
        from tensortruth.api.deps import get_config_service
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

        # Use FastAPI's dependency override mechanism
        app.dependency_overrides[get_config_service] = lambda: test_service

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/modules")
                assert response.status_code == 200
                modules = response.json()["modules"]
                assert len(modules) == 1
                assert modules[0]["name"] == "pytorch"
        finally:
            app.dependency_overrides.clear()

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


class TestEmbeddingModelModulesIntegration:
    """Integration tests for embedding model -> modules relationship."""

    @pytest.mark.asyncio
    async def test_modules_change_when_embedding_model_changes(
        self, app, tmp_path, monkeypatch
    ):
        """Test that changing embedding model changes which modules are returned."""
        from tensortruth.api.deps import get_config_service
        from tensortruth.services import ConfigService

        indexes_dir = tmp_path / "indexes"
        indexes_dir.mkdir()

        # Create indexes for two different embedding models
        # BGE-M3 model has pytorch module
        bge_dir = indexes_dir / "bge-m3"
        bge_dir.mkdir()
        bge_module = bge_dir / "pytorch"
        bge_module.mkdir()
        (bge_module / "chroma.sqlite3").touch()

        # MiniLM model has numpy module
        minilm_dir = indexes_dir / "all-minilm-l6-v2"
        minilm_dir.mkdir()
        minilm_module = minilm_dir / "numpy"
        minilm_module.mkdir()
        (minilm_module / "chroma.sqlite3").touch()

        monkeypatch.setattr(
            "tensortruth.api.routes.modules.get_indexes_dir", lambda: indexes_dir
        )

        # Set up config with BGE-M3 (this is the default)
        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        # Use FastAPI's dependency override mechanism
        app.dependency_overrides[get_config_service] = lambda: test_service

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Initial request should return pytorch (BGE-M3 default)
                response1 = await client.get("/api/modules")
                assert response1.status_code == 200
                modules1 = [m["name"] for m in response1.json()["modules"]]
                assert "pytorch" in modules1
                assert "numpy" not in modules1

                # Change config to MiniLM model
                test_service.update(
                    rag_default_embedding_model="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Request should now return numpy (MiniLM model)
                response2 = await client.get("/api/modules")
                assert response2.status_code == 200
                modules2 = [m["name"] for m in response2.json()["modules"]]
                assert "numpy" in modules2
                assert "pytorch" not in modules2
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_modules_empty_for_nonexistent_model(
        self, app, tmp_path, monkeypatch
    ):
        """Test that modules list is empty when embedding model has no indexes."""
        from tensortruth.api.deps import get_config_service
        from tensortruth.services import ConfigService

        indexes_dir = tmp_path / "indexes"
        indexes_dir.mkdir()

        # Only create indexes for BGE-M3
        bge_dir = indexes_dir / "bge-m3"
        bge_dir.mkdir()
        bge_module = bge_dir / "pytorch"
        bge_module.mkdir()
        (bge_module / "chroma.sqlite3").touch()

        monkeypatch.setattr(
            "tensortruth.api.routes.modules.get_indexes_dir", lambda: indexes_dir
        )

        # Set up config with a model that has no indexes
        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)
        # Update config to use a model with no indexes
        test_service.update(rag_default_embedding_model="some-model/no-indexes")

        # Use FastAPI's dependency override mechanism
        app.dependency_overrides[get_config_service] = lambda: test_service

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/modules")
                assert response.status_code == 200
                modules = response.json()["modules"]
                assert len(modules) == 0
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_list_embedding_models(self, app, tmp_path, monkeypatch):
        """Test listing available embedding models."""
        from tensortruth.api.deps import get_config_service
        from tensortruth.services import ConfigService

        indexes_dir = tmp_path / "indexes"
        indexes_dir.mkdir()

        # Create indexes for two models
        bge_dir = indexes_dir / "bge-m3"
        bge_dir.mkdir()
        (bge_dir / "pytorch").mkdir()
        (bge_dir / "pytorch" / "chroma.sqlite3").touch()
        (bge_dir / "numpy").mkdir()
        (bge_dir / "numpy" / "chroma.sqlite3").touch()

        minilm_dir = indexes_dir / "all-minilm-l6-v2"
        minilm_dir.mkdir()
        (minilm_dir / "tensorflow").mkdir()
        (minilm_dir / "tensorflow" / "chroma.sqlite3").touch()

        monkeypatch.setattr(
            "tensortruth.api.routes.modules.get_indexes_dir", lambda: indexes_dir
        )

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        # Use FastAPI's dependency override mechanism
        app.dependency_overrides[get_config_service] = lambda: test_service

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/api/embedding-models")
                assert response.status_code == 200
                data = response.json()

                assert "models" in data
                assert "current" in data
                assert data["current"] == "bge-m3"  # Default model

                model_ids = [m["model_id"] for m in data["models"]]
                assert "bge-m3" in model_ids
                assert "all-minilm-l6-v2" in model_ids

                # Check module counts
                bge_model = next(m for m in data["models"] if m["model_id"] == "bge-m3")
                assert bge_model["index_count"] == 2
                assert set(bge_model["modules"]) == {"pytorch", "numpy"}
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_embedding_models_reflects_config_change(
        self, app, tmp_path, monkeypatch
    ):
        """Test that current embedding model reflects config changes."""
        from tensortruth.api.deps import get_config_service
        from tensortruth.services import ConfigService

        indexes_dir = tmp_path / "indexes"
        indexes_dir.mkdir()

        # Create minimal index structure
        bge_dir = indexes_dir / "bge-m3"
        bge_dir.mkdir()
        (bge_dir / "pytorch").mkdir()
        (bge_dir / "pytorch" / "chroma.sqlite3").touch()

        monkeypatch.setattr(
            "tensortruth.api.routes.modules.get_indexes_dir", lambda: indexes_dir
        )

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)

        # Use FastAPI's dependency override mechanism
        app.dependency_overrides[get_config_service] = lambda: test_service

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Initial check - should be bge-m3
                response1 = await client.get("/api/embedding-models")
                assert response1.json()["current"] == "bge-m3"

                # Change config using the update method with prefixed key
                test_service.update(
                    rag_default_embedding_model="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Should reflect new model
                response2 = await client.get("/api/embedding-models")
                assert response2.json()["current"] == "all-minilm-l6-v2"
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_embedding_models_contains_module_list_per_model(
        self, app, tmp_path, monkeypatch
    ):
        """Test that /embedding-models returns modules array for each model.

        This is critical for the frontend: when session settings select an embedding
        model that differs from config's default, the frontend must use the modules
        array from /embedding-models (NOT from /modules) to determine which modules
        are available.
        """
        from tensortruth.api.deps import get_config_service
        from tensortruth.services import ConfigService

        indexes_dir = tmp_path / "indexes"
        indexes_dir.mkdir()

        # Create distinct modules for each embedding model
        bge_dir = indexes_dir / "bge-m3"
        bge_dir.mkdir()
        for name in ["calculus_strang", "linear_algebra"]:
            (bge_dir / name).mkdir()
            (bge_dir / name / "chroma.sqlite3").touch()

        minilm_dir = indexes_dir / "all-minilm-l6-v2"
        minilm_dir.mkdir()
        for name in ["deep_learning"]:
            (minilm_dir / name).mkdir()
            (minilm_dir / name / "chroma.sqlite3").touch()

        monkeypatch.setattr(
            "tensortruth.api.routes.modules.get_indexes_dir", lambda: indexes_dir
        )

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)
        # Default config uses bge-m3
        app.dependency_overrides[get_config_service] = lambda: test_service

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Get embedding models - this provides the complete picture
                response = await client.get("/api/embedding-models")
                assert response.status_code == 200
                data = response.json()

                # Find minilm model
                minilm_model = next(
                    m for m in data["models"] if m["model_id"] == "all-minilm-l6-v2"
                )
                # Frontend can use minilm_model["modules"] to know available modules for this model
                assert minilm_model["modules"] == ["deep_learning"]

                bge_model = next(m for m in data["models"] if m["model_id"] == "bge-m3")
                assert set(bge_model["modules"]) == {
                    "calculus_strang",
                    "linear_algebra",
                }

                # Verify /modules only returns config default (bge-m3) modules
                modules_response = await client.get("/api/modules")
                module_names = [m["name"] for m in modules_response.json()["modules"]]
                assert set(module_names) == {"calculus_strang", "linear_algebra"}
                # NOT deep_learning because config default is bge-m3
                assert "deep_learning" not in module_names
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_frontend_session_settings_use_case(self, app, tmp_path, monkeypatch):
        """Test the frontend use case: session selects different embedding model than config.

        Scenario:
        - Config default: bge-m3
        - Session settings: user selects all-minilm-l6-v2
        - Frontend module selector should show minilm modules, not bge modules

        The frontend must:
        1. Use session's embedding_model (not config's default)
        2. Get available modules from /embedding-models response
        3. NOT rely on /modules endpoint (which only returns config default)
        """
        from tensortruth.api.deps import get_config_service
        from tensortruth.services import ConfigService

        indexes_dir = tmp_path / "indexes"
        indexes_dir.mkdir()

        # BGE has many modules (config default)
        bge_dir = indexes_dir / "bge-m3"
        bge_dir.mkdir()
        for name in ["book_calculus", "book_linear_algebra", "paper_attention"]:
            (bge_dir / name).mkdir()
            (bge_dir / name / "chroma.sqlite3").touch()

        # MiniLM has few modules (session selection)
        minilm_dir = indexes_dir / "all-minilm-l6-v2"
        minilm_dir.mkdir()
        for name in ["book_deep_learning"]:
            (minilm_dir / name).mkdir()
            (minilm_dir / name / "chroma.sqlite3").touch()

        monkeypatch.setattr(
            "tensortruth.api.routes.modules.get_indexes_dir", lambda: indexes_dir
        )

        config_file = tmp_path / "config.yaml"
        test_service = ConfigService(config_file=config_file)
        # Config default is bge-m3
        app.dependency_overrides[get_config_service] = lambda: test_service

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Frontend gets embedding models
                emb_response = await client.get("/api/embedding-models")
                emb_data = emb_response.json()
                assert emb_data["current"] == "bge-m3"  # Config default

                # Frontend gets modules (this returns bge-m3 modules)
                modules_response = await client.get("/api/modules")
                api_modules = [m["name"] for m in modules_response.json()["modules"]]
                # This has 3 modules (bge-m3's modules)
                assert len(api_modules) == 3

                # USER CHANGES SESSION EMBEDDING MODEL TO MINILM
                # Frontend now needs minilm modules, but /modules still returns bge's
                session_embedding_model = "all-minilm-l6-v2"

                # CORRECT APPROACH: Get modules from /embedding-models
                minilm_model = next(
                    m
                    for m in emb_data["models"]
                    if m["model_id"] == session_embedding_model
                )
                session_modules = minilm_model["modules"]
                # Only 1 module for minilm
                assert session_modules == ["book_deep_learning"]

                # WRONG APPROACH: Filtering /modules by minilm
                # This would return empty because /modules has bge modules only
                filtered = [m for m in api_modules if m in session_modules]
                assert (
                    filtered == []
                )  # Empty! This is the bug if frontend uses /modules
        finally:
            app.dependency_overrides.clear()
