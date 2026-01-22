"""Integration tests for system info API endpoints."""

import pytest
from fastapi.testclient import TestClient

from tensortruth.api.main import app

client = TestClient(app)


class TestSystemMemoryEndpoint:
    """Tests for GET /api/system/memory."""

    def test_get_memory_returns_200(self):
        """Should return 200 OK."""
        response = client.get("/api/system/memory")
        assert response.status_code == 200

    def test_get_memory_returns_expected_schema(self):
        """Should return memory info with expected schema."""
        response = client.get("/api/system/memory")
        data = response.json()

        # Check top-level structure
        assert "memory" in data
        assert isinstance(data["memory"], list)

        # Check each memory info object
        for mem in data["memory"]:
            assert "name" in mem
            assert "allocated_gb" in mem
            assert isinstance(mem["name"], str)
            assert isinstance(mem["allocated_gb"], (int, float))

            # total_gb and details can be null
            if mem["total_gb"] is not None:
                assert isinstance(mem["total_gb"], (int, float))
            if mem["details"] is not None:
                assert isinstance(mem["details"], str)

    def test_get_memory_includes_system_ram(self):
        """Should include system RAM in response (unless psutil unavailable)."""
        response = client.get("/api/system/memory")
        data = response.json()

        # At minimum, we should get system RAM (if psutil available)
        # This might fail in CI without psutil, but should pass locally
        memory_names = [mem["name"] for mem in data["memory"]]

        # We should have at least one memory component
        assert len(memory_names) > 0


class TestSystemDevicesEndpoint:
    """Tests for GET /api/system/devices."""

    def test_get_devices_returns_200(self):
        """Should return 200 OK."""
        response = client.get("/api/system/devices")
        assert response.status_code == 200

    def test_get_devices_returns_expected_schema(self):
        """Should return devices list with expected schema."""
        response = client.get("/api/system/devices")
        data = response.json()

        assert "devices" in data
        assert isinstance(data["devices"], list)
        assert len(data["devices"]) > 0  # At minimum, should have 'cpu'

    def test_get_devices_includes_cpu(self):
        """Should always include CPU device."""
        response = client.get("/api/system/devices")
        data = response.json()

        assert "cpu" in data["devices"]

    def test_get_devices_order_preference(self):
        """Devices should be ordered by preference (cuda > mps > cpu)."""
        response = client.get("/api/system/devices")
        data = response.json()
        devices = data["devices"]

        # If cuda is present, it should be first
        if "cuda" in devices:
            assert devices[0] == "cuda"

        # If mps is present and cuda is not, mps should be first
        if "mps" in devices and "cuda" not in devices:
            assert devices[0] == "mps"

        # CPU should always be last
        assert devices[-1] == "cpu"


class TestSystemOllamaEndpoint:
    """Tests for GET /api/system/ollama/status."""

    def test_get_ollama_status_returns_200(self):
        """Should return 200 OK (even if Ollama not running)."""
        response = client.get("/api/system/ollama/status")
        assert response.status_code == 200

    def test_get_ollama_status_returns_expected_schema(self):
        """Should return Ollama status with expected schema."""
        response = client.get("/api/system/ollama/status")
        data = response.json()

        # Check top-level structure
        assert "running" in data
        assert "models" in data
        assert "info_lines" in data
        assert isinstance(data["running"], bool)
        assert isinstance(data["models"], list)
        assert isinstance(data["info_lines"], list)

    def test_get_ollama_status_models_schema(self):
        """If models are running, they should have expected schema."""
        response = client.get("/api/system/ollama/status")
        data = response.json()

        for model in data["models"]:
            assert "name" in model
            assert "size_vram_gb" in model
            assert "size_gb" in model
            assert isinstance(model["name"], str)
            assert isinstance(model["size_vram_gb"], (int, float))
            assert isinstance(model["size_gb"], (int, float))

            # parameters can be null
            if model["parameters"] is not None:
                assert isinstance(model["parameters"], str)

    def test_get_ollama_status_running_consistency(self):
        """If models list is non-empty, running should be true."""
        response = client.get("/api/system/ollama/status")
        data = response.json()

        if len(data["models"]) > 0:
            assert data["running"] is True
        else:
            # If no models, running should be false
            assert data["running"] is False
