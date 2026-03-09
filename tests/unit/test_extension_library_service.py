"""Tests for ExtensionLibraryService."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tensortruth.services.extension_catalog import CACHE_TTL_SECONDS
from tensortruth.services.extension_library_service import ExtensionLibraryService
from tensortruth.services.mcp_server_service import MCPServerService

SAMPLE_CATALOG = {
    "version": 1,
    "generated_at": "2026-03-08T12:00:00Z",
    "extensions": [
        {
            "name": "test_cmd",
            "type": "command",
            "filename": "test_cmd.yaml",
            "description": "A test command",
        },
        {
            "name": "helper",
            "type": "agent",
            "filename": "helper.yaml",
            "description": "A helper agent",
        },
        {
            "name": "context7",
            "type": "command",
            "filename": "context7.yaml",
            "description": "Context7 lookup",
            "requires_mcp": "context7",
        },
    ],
}


@pytest.fixture
def user_dir(tmp_path):
    """Create temp user directory."""
    d = tmp_path / "user"
    (d / "commands").mkdir(parents=True)
    (d / "agents").mkdir(parents=True)
    return d


@pytest.fixture
def mcp_service(tmp_path):
    """Create an MCP service with temp config."""
    return MCPServerService(config_path=tmp_path / "mcp_servers.json")


@pytest.fixture
def service(user_dir, mcp_service):
    """Create service with temp user dir and mock catalog URL."""
    return ExtensionLibraryService(
        user_dir=user_dir,
        catalog_url="https://example.com/ext",
        mcp_service=mcp_service,
    )


def _write_yaml(directory: Path, filename: str, content: str):
    (directory / filename).write_text(content, encoding="utf-8")


def _write_cache(user_dir: Path, catalog: dict, expired: bool = False):
    """Write a catalog cache file."""
    fetched_at = time.time()
    if expired:
        fetched_at -= CACHE_TTL_SECONDS + 100
    payload = {"fetched_at": fetched_at, "catalog": catalog}
    (user_dir / "catalog_cache.json").write_text(json.dumps(payload), encoding="utf-8")


class TestListInstalled:
    def test_empty(self, service):
        assert service.list_installed() == []

    def test_lists_commands(self, service, user_dir):
        _write_yaml(
            user_dir / "commands",
            "test.yaml",
            """
name: test
description: A test command
usage: "/test"
steps:
  - tool: search_web
    params:
      queries: "{{args}}"
""",
        )
        installed = service.list_installed()
        assert len(installed) == 1
        assert installed[0]["name"] == "test"
        assert installed[0]["type"] == "command"
        assert installed[0]["filename"] == "test.yaml"

    def test_lists_agents(self, service, user_dir):
        _write_yaml(
            user_dir / "agents",
            "helper.yaml",
            """
name: helper
description: A helper agent
tools: [search_web]
""",
        )
        installed = service.list_installed()
        assert len(installed) == 1
        assert installed[0]["name"] == "helper"
        assert installed[0]["type"] == "agent"

    def test_mcp_available_true_when_configured(self, service, user_dir, mcp_service):
        # Add a configured MCP server
        mcp_service.add(
            {
                "name": "context7",
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp@latest"],
            }
        )
        _write_yaml(
            user_dir / "commands",
            "c7.yaml",
            """
name: context7
description: Context7 lookup
usage: "/context7"
requires_mcp: context7
steps:
  - tool: resolve-library-id
    params:
      libraryName: "test"
""",
        )
        installed = service.list_installed()
        assert installed[0]["mcp_available"] is True

    def test_mcp_available_false_when_not_configured(self, service, user_dir):
        _write_yaml(
            user_dir / "commands",
            "c7.yaml",
            """
name: context7
description: Context7 lookup
usage: "/context7"
requires_mcp: context7
steps:
  - tool: resolve-library-id
    params:
      libraryName: "test"
""",
        )
        installed = service.list_installed()
        assert installed[0]["mcp_available"] is False


class TestListLibrary:
    @patch("tensortruth.services.extension_library_service.fetch_catalog")
    def test_cache_miss_fetches_remote(self, mock_fetch, service):
        mock_fetch.return_value = SAMPLE_CATALOG
        library = service.list_library()
        assert len(library) == 3
        assert library[0]["name"] == "test_cmd"
        assert library[0]["installed"] is False
        mock_fetch.assert_called_once()

    def test_cache_hit_skips_fetch(self, service, user_dir):
        _write_cache(user_dir, SAMPLE_CATALOG)
        with patch(
            "tensortruth.services.extension_library_service.fetch_catalog"
        ) as mock_fetch:
            library = service.list_library()
            assert len(library) == 3
            mock_fetch.assert_not_called()

    @patch("tensortruth.services.extension_library_service.fetch_catalog")
    def test_fetch_failure_uses_expired_cache(self, mock_fetch, service, user_dir):
        _write_cache(user_dir, SAMPLE_CATALOG, expired=True)
        mock_fetch.side_effect = ConnectionError("no network")
        library = service.list_library()
        assert len(library) == 3

    @patch("tensortruth.services.extension_library_service.fetch_catalog")
    def test_fully_offline_returns_empty(self, mock_fetch, service):
        mock_fetch.side_effect = ConnectionError("no network")
        library = service.list_library()
        assert library == []

    @patch("tensortruth.services.extension_library_service.fetch_catalog")
    def test_marks_installed(self, mock_fetch, service, user_dir):
        mock_fetch.return_value = SAMPLE_CATALOG
        _write_yaml(
            user_dir / "commands",
            "test_cmd.yaml",
            "name: test_cmd\ndescription: A test command",
        )
        library = service.list_library()
        test_cmd = next(e for e in library if e["name"] == "test_cmd")
        assert test_cmd["installed"] is True

    @patch("tensortruth.services.extension_library_service.fetch_catalog")
    def test_mcp_available_flag(self, mock_fetch, service):
        mock_fetch.return_value = SAMPLE_CATALOG
        library = service.list_library()
        c7 = next(e for e in library if e["name"] == "context7")
        assert c7["requires_mcp"] == "context7"
        assert c7["mcp_available"] is False


class TestInstall:
    @patch("tensortruth.services.extension_library_service.fetch_extension_yaml")
    def test_install_command(self, mock_fetch, service, user_dir):
        mock_fetch.return_value = "name: test\ndescription: A command\n"
        result = service.install("command", "test.yaml")
        assert result == "test.yaml"
        assert (user_dir / "commands" / "test.yaml").exists()
        content = (user_dir / "commands" / "test.yaml").read_text()
        assert "name: test" in content

    @patch("tensortruth.services.extension_library_service.fetch_extension_yaml")
    def test_install_agent(self, mock_fetch, service, user_dir):
        mock_fetch.return_value = "name: helper\ndescription: A helper\n"
        result = service.install("agent", "helper.yaml")
        assert result == "helper.yaml"
        assert (user_dir / "agents" / "helper.yaml").exists()

    def test_install_invalid_type_raises(self, service):
        with pytest.raises(ValueError, match="Invalid extension type"):
            service.install("invalid", "test.yaml")

    def test_install_path_traversal_raises(self, service):
        with pytest.raises(ValueError, match="Invalid filename"):
            service.install("command", "../etc/passwd")

    def test_install_non_yaml_raises(self, service):
        with pytest.raises(ValueError, match="Invalid filename"):
            service.install("command", "test.py")

    @patch("tensortruth.services.extension_library_service.fetch_extension_yaml")
    def test_install_network_failure_raises(self, mock_fetch, service):
        mock_fetch.side_effect = ConnectionError("no network")
        with pytest.raises(ConnectionError):
            service.install("command", "test.yaml")


class TestUninstall:
    def test_uninstall(self, service, user_dir):
        _write_yaml(user_dir / "commands", "test.yaml", "name: test")
        service.uninstall("command", "test.yaml")
        assert not (user_dir / "commands" / "test.yaml").exists()

    def test_uninstall_not_found_raises(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.uninstall("command", "nonexistent.yaml")

    def test_uninstall_path_traversal_raises(self, service):
        with pytest.raises(ValueError, match="Invalid filename"):
            service.uninstall("command", "../../etc/passwd")
