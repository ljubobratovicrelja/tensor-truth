"""Tests for ExtensionLibraryService."""

from pathlib import Path

import pytest

from tensortruth.services.extension_library_service import ExtensionLibraryService
from tensortruth.services.mcp_server_service import MCPServerService


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temp user and library directories."""
    user_dir = tmp_path / "user"
    library_dir = tmp_path / "library"
    # User directories
    (user_dir / "commands").mkdir(parents=True)
    (user_dir / "agents").mkdir(parents=True)
    # Library directories
    (library_dir / "commands").mkdir(parents=True)
    (library_dir / "agents").mkdir(parents=True)
    return user_dir, library_dir


@pytest.fixture
def mcp_service(tmp_path):
    """Create an MCP service with temp config."""
    return MCPServerService(config_path=tmp_path / "mcp_servers.json")


@pytest.fixture
def service(tmp_dirs, mcp_service):
    """Create service with temp directories."""
    user_dir, library_dir = tmp_dirs
    return ExtensionLibraryService(
        user_dir=user_dir,
        library_dir=library_dir,
        mcp_service=mcp_service,
    )


def _write_yaml(directory: Path, filename: str, content: str):
    (directory / filename).write_text(content, encoding="utf-8")


class TestListInstalled:
    def test_empty(self, service):
        assert service.list_installed() == []

    def test_lists_commands(self, service, tmp_dirs):
        user_dir, _ = tmp_dirs
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

    def test_lists_agents(self, service, tmp_dirs):
        user_dir, _ = tmp_dirs
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

    def test_mcp_available_true_when_configured(self, service, tmp_dirs, mcp_service):
        user_dir, _ = tmp_dirs
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

    def test_mcp_available_false_when_not_configured(self, service, tmp_dirs):
        user_dir, _ = tmp_dirs
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
    def test_empty(self, service):
        assert service.list_library() == []

    def test_lists_library_extensions(self, service, tmp_dirs):
        _, library_dir = tmp_dirs
        _write_yaml(
            library_dir / "commands",
            "test.yaml",
            """
name: test
description: A library command
usage: "/test"
steps:
  - tool: search_web
    params:
      queries: "{{args}}"
""",
        )
        library = service.list_library()
        assert len(library) == 1
        assert library[0]["name"] == "test"
        assert library[0]["installed"] is False

    def test_marks_installed(self, service, tmp_dirs):
        user_dir, library_dir = tmp_dirs
        yaml_content = """
name: test
description: A command
usage: "/test"
steps:
  - tool: search_web
    params:
      queries: "{{args}}"
"""
        _write_yaml(library_dir / "commands", "test.yaml", yaml_content)
        _write_yaml(user_dir / "commands", "test.yaml", yaml_content)
        library = service.list_library()
        assert library[0]["installed"] is True


class TestInstall:
    def test_install_command(self, service, tmp_dirs):
        user_dir, library_dir = tmp_dirs
        _write_yaml(
            library_dir / "commands",
            "test.yaml",
            """
name: test
description: A command
usage: "/test"
steps:
  - tool: search_web
    params:
      queries: "{{args}}"
""",
        )
        result = service.install("command", "test.yaml")
        assert result == "test.yaml"
        assert (user_dir / "commands" / "test.yaml").exists()

    def test_install_agent(self, service, tmp_dirs):
        user_dir, library_dir = tmp_dirs
        _write_yaml(
            library_dir / "agents",
            "helper.yaml",
            """
name: helper
description: A helper
tools: [search_web]
""",
        )
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

    def test_install_not_found_raises(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.install("command", "nonexistent.yaml")


class TestUninstall:
    def test_uninstall(self, service, tmp_dirs):
        user_dir, _ = tmp_dirs
        _write_yaml(user_dir / "commands", "test.yaml", "name: test")
        service.uninstall("command", "test.yaml")
        assert not (user_dir / "commands" / "test.yaml").exists()

    def test_uninstall_not_found_raises(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.uninstall("command", "nonexistent.yaml")

    def test_uninstall_path_traversal_raises(self, service):
        with pytest.raises(ValueError, match="Invalid filename"):
            service.uninstall("command", "../../etc/passwd")
