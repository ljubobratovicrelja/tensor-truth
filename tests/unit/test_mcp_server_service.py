"""Tests for MCPServerService."""

import json

import pytest

from tensortruth.services.mcp_server_service import MCPServerService


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary config path."""
    return tmp_path / "mcp_servers.json"


@pytest.fixture
def service(tmp_config):
    """Create a service with a temp config path."""
    return MCPServerService(config_path=tmp_config)


class TestListAll:
    def test_lists_builtin_servers(self, service):
        servers = service.list_all()
        assert len(servers) >= 1
        builtin = [s for s in servers if s["builtin"]]
        assert len(builtin) >= 1
        assert builtin[0]["name"] == "tensor-truth-web"

    def test_lists_user_servers(self, service, tmp_config):
        tmp_config.write_text(
            json.dumps(
                {
                    "servers": [
                        {
                            "name": "test-server",
                            "type": "stdio",
                            "command": "echo",
                            "args": ["hello"],
                            "enabled": True,
                        }
                    ]
                }
            )
        )
        servers = service.list_all()
        user = [s for s in servers if not s["builtin"]]
        assert len(user) == 1
        assert user[0]["name"] == "test-server"

    def test_builtin_override_enabled_state(self, service, tmp_config):
        tmp_config.write_text(
            json.dumps({"servers": [{"name": "tensor-truth-web", "enabled": False}]})
        )
        servers = service.list_all()
        builtin = next(s for s in servers if s["name"] == "tensor-truth-web")
        assert builtin["enabled"] is False
        assert builtin["builtin"] is True


class TestAdd:
    def test_add_server(self, service, tmp_config):
        result = service.add(
            {
                "name": "my-server",
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "some-package"],
            }
        )
        assert result["name"] == "my-server"
        assert result["builtin"] is False

        # Verify persisted
        data = json.loads(tmp_config.read_text())
        assert len(data["servers"]) == 1

    def test_add_duplicate_raises(self, service, tmp_config):
        service.add(
            {
                "name": "test",
                "type": "stdio",
                "command": "echo",
            }
        )
        with pytest.raises(ValueError, match="already exists"):
            service.add(
                {
                    "name": "test",
                    "type": "stdio",
                    "command": "echo",
                }
            )

    def test_add_builtin_name_raises(self, service):
        with pytest.raises(ValueError, match="already exists"):
            service.add(
                {
                    "name": "tensor-truth-web",
                    "type": "stdio",
                    "command": "echo",
                }
            )

    def test_add_invalid_config_raises(self, service):
        with pytest.raises(ValueError):
            service.add(
                {
                    "name": "bad",
                    "type": "stdio",
                    # missing command
                }
            )

    def test_add_sse_server(self, service):
        result = service.add(
            {
                "name": "sse-server",
                "type": "sse",
                "url": "http://localhost:3000/sse",
            }
        )
        assert result["name"] == "sse-server"


class TestUpdate:
    def test_update_server(self, service):
        service.add(
            {
                "name": "test",
                "type": "stdio",
                "command": "echo",
                "description": "old",
            }
        )
        result = service.update("test", {"description": "new"})
        assert result["description"] == "new"

    def test_update_builtin_raises(self, service):
        with pytest.raises(ValueError, match="built-in"):
            service.update("tensor-truth-web", {"description": "new"})

    def test_update_nonexistent_raises(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.update("nonexistent", {"description": "new"})


class TestRemove:
    def test_remove_server(self, service, tmp_config):
        service.add({"name": "test", "type": "stdio", "command": "echo"})
        service.remove("test")
        data = json.loads(tmp_config.read_text())
        assert len(data["servers"]) == 0

    def test_remove_builtin_raises(self, service):
        with pytest.raises(ValueError, match="built-in"):
            service.remove("tensor-truth-web")

    def test_remove_nonexistent_raises(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.remove("nonexistent")


class TestToggle:
    def test_toggle_user_server(self, service):
        service.add({"name": "test", "type": "stdio", "command": "echo"})
        result = service.toggle("test", False)
        assert result["enabled"] is False

        result = service.toggle("test", True)
        assert result["enabled"] is True

    def test_toggle_builtin_server(self, service, tmp_config):
        result = service.toggle("tensor-truth-web", False)
        assert result["enabled"] is False
        assert result["builtin"] is True

        # Verify persisted
        data = json.loads(tmp_config.read_text())
        override = next(s for s in data["servers"] if s["name"] == "tensor-truth-web")
        assert override["enabled"] is False

    def test_toggle_nonexistent_raises(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.toggle("nonexistent", True)


class TestEnvStatus:
    def test_env_status_resolved(self, service, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret123")
        service.add(
            {
                "name": "test",
                "type": "stdio",
                "command": "echo",
                "env": {"TOKEN": "$MY_TOKEN"},
            }
        )
        servers = service.list_all()
        user = next(s for s in servers if s["name"] == "test")
        assert user["env_status"]["MY_TOKEN"] is True

    def test_env_status_unresolved(self, service, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        service.add(
            {
                "name": "test",
                "type": "stdio",
                "command": "echo",
                "env": {"TOKEN": "$MISSING_VAR"},
            }
        )
        servers = service.list_all()
        user = next(s for s in servers if s["name"] == "test")
        assert user["env_status"]["MISSING_VAR"] is False


class TestPresets:
    def test_get_presets(self):
        presets = MCPServerService.get_presets()
        assert "context7" in presets
        assert "github" in presets
        assert "huggingface" in presets
        assert "playwright" in presets
        assert "memory" in presets
        assert "scholarly-research" in presets
        assert "wikipedia" in presets
        assert len(presets) == 7
        assert presets["context7"]["command"] == "npx"
        assert presets["playwright"]["command"] == "npx"
        assert presets["wikipedia"]["command"] == "npx"


class TestGetConfiguredServerNames:
    def test_includes_builtin(self, service):
        names = service.get_configured_server_names()
        assert "tensor-truth-web" in names

    def test_includes_user_servers(self, service):
        service.add({"name": "test", "type": "stdio", "command": "echo"})
        names = service.get_configured_server_names()
        assert "test" in names

    def test_excludes_disabled(self, service):
        service.add(
            {
                "name": "test",
                "type": "stdio",
                "command": "echo",
                "enabled": False,
            }
        )
        names = service.get_configured_server_names()
        assert "test" not in names
