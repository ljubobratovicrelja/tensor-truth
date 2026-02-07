"""Tests for UserExtensionLoader."""

from unittest.mock import MagicMock

import pytest

from tensortruth.api.routes.commands import CommandRegistry
from tensortruth.extensions.loader import UserExtensionLoader


@pytest.fixture
def tmp_ext_dir(tmp_path):
    """Create a temporary extension directory structure."""
    (tmp_path / "commands").mkdir()
    (tmp_path / "agents").mkdir()
    return tmp_path


@pytest.fixture
def cmd_registry():
    return CommandRegistry()


@pytest.fixture
def mock_agent_service():
    svc = MagicMock()
    svc.register_agent = MagicMock()
    return svc


@pytest.fixture
def mock_tool_service():
    svc = MagicMock()
    return svc


class TestLoaderYamlCommands:
    @pytest.mark.asyncio
    async def test_loads_steps_command(
        self, tmp_ext_dir, cmd_registry, mock_agent_service, mock_tool_service
    ):
        yaml_content = """\
name: test
description: A test command
usage: "/test <query>"
steps:
  - tool: my-tool
    params:
      q: "{{args}}"
"""
        (tmp_ext_dir / "commands" / "test.yaml").write_text(yaml_content)

        loader = UserExtensionLoader(user_dir=tmp_ext_dir)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert result.commands_loaded == 1
        assert len(result.errors) == 0
        assert cmd_registry.get("test") is not None

    @pytest.mark.asyncio
    async def test_loads_agent_command(
        self, tmp_ext_dir, cmd_registry, mock_agent_service, mock_tool_service
    ):
        yaml_content = """\
name: research_docs
description: Research docs
agent: doc_researcher
"""
        (tmp_ext_dir / "commands" / "research.yaml").write_text(yaml_content)

        loader = UserExtensionLoader(user_dir=tmp_ext_dir)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert result.commands_loaded == 1
        cmd = cmd_registry.get("research_docs")
        assert cmd is not None

    @pytest.mark.asyncio
    async def test_invalid_yaml_skipped(
        self, tmp_ext_dir, cmd_registry, mock_agent_service, mock_tool_service
    ):
        (tmp_ext_dir / "commands" / "bad.yaml").write_text("{{invalid: yaml: [")

        loader = UserExtensionLoader(user_dir=tmp_ext_dir)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert result.commands_loaded == 0
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_schema_error_skipped(
        self, tmp_ext_dir, cmd_registry, mock_agent_service, mock_tool_service
    ):
        # Missing both steps and agent
        yaml_content = """\
name: bad
description: Missing steps and agent
"""
        (tmp_ext_dir / "commands" / "bad.yaml").write_text(yaml_content)

        loader = UserExtensionLoader(user_dir=tmp_ext_dir)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert result.commands_loaded == 0
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_aliases_registered(
        self, tmp_ext_dir, cmd_registry, mock_agent_service, mock_tool_service
    ):
        yaml_content = """\
name: context7
description: Context7 lookup
aliases: [c7]
steps:
  - tool: resolve-library-id
    params:
      libraryName: "{{args}}"
"""
        (tmp_ext_dir / "commands" / "context7.yaml").write_text(yaml_content)

        loader = UserExtensionLoader(user_dir=tmp_ext_dir)
        await loader.load_all(cmd_registry, mock_agent_service, mock_tool_service)

        assert cmd_registry.get("context7") is not None
        assert cmd_registry.get("c7") is not None
        assert cmd_registry.get("context7") is cmd_registry.get("c7")


class TestLoaderYamlAgents:
    @pytest.mark.asyncio
    async def test_loads_agent(
        self, tmp_ext_dir, cmd_registry, mock_agent_service, mock_tool_service
    ):
        yaml_content = """\
name: doc_researcher
description: Research using Context7
tools:
  - get-library-docs
  - search_web
agent_type: function
system_prompt: "You are a doc researcher."
max_iterations: 8
"""
        (tmp_ext_dir / "agents" / "doc_researcher.yaml").write_text(yaml_content)

        loader = UserExtensionLoader(user_dir=tmp_ext_dir)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert result.agents_loaded == 1
        assert len(result.errors) == 0
        mock_agent_service.register_agent.assert_called_once()

        config = mock_agent_service.register_agent.call_args.args[0]
        assert config.name == "doc_researcher"
        assert config.agent_type == "function"
        assert config.max_iterations == 8
        assert "get-library-docs" in config.tools


class TestLoaderPython:
    @pytest.mark.asyncio
    async def test_loads_python_extension(
        self, tmp_ext_dir, cmd_registry, mock_agent_service, mock_tool_service
    ):
        py_content = """\
from tensortruth.api.routes.commands import ToolCommand


class TestCmd(ToolCommand):
    name = "pycmd"
    aliases = []
    description = "Python command"
    usage = "/pycmd"

    async def execute(self, args, session, websocket):
        await websocket.send_json({"type": "done", "content": "ok"})


def register(command_registry, agent_service, tool_service):
    command_registry.register(TestCmd())
"""
        (tmp_ext_dir / "commands" / "pycmd.py").write_text(py_content)

        loader = UserExtensionLoader(user_dir=tmp_ext_dir)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert result.commands_loaded == 1
        assert cmd_registry.get("pycmd") is not None

    @pytest.mark.asyncio
    async def test_missing_register_skipped(
        self, tmp_ext_dir, cmd_registry, mock_agent_service, mock_tool_service
    ):
        py_content = """\
# No register() function
x = 1
"""
        (tmp_ext_dir / "commands" / "noreg.py").write_text(py_content)

        loader = UserExtensionLoader(user_dir=tmp_ext_dir)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert result.commands_loaded == 0
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_import_error_skipped(
        self, tmp_ext_dir, cmd_registry, mock_agent_service, mock_tool_service
    ):
        py_content = """\
import nonexistent_module_xyz123

def register(command_registry, agent_service, tool_service):
    pass
"""
        (tmp_ext_dir / "commands" / "bad_import.py").write_text(py_content)

        loader = UserExtensionLoader(user_dir=tmp_ext_dir)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert result.commands_loaded == 0
        assert len(result.errors) == 1


class TestLoaderMissingDirs:
    @pytest.mark.asyncio
    async def test_missing_dirs_ok(
        self, tmp_path, cmd_registry, mock_agent_service, mock_tool_service
    ):
        """No commands/ or agents/ dirs -> no errors, no extensions."""
        loader = UserExtensionLoader(user_dir=tmp_path)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert result.commands_loaded == 0
        assert result.agents_loaded == 0
        assert len(result.errors) == 0


class TestExtensionLoadResult:
    @pytest.mark.asyncio
    async def test_repr_with_data(
        self, tmp_ext_dir, cmd_registry, mock_agent_service, mock_tool_service
    ):
        yaml_content = """\
name: test
description: Test
steps:
  - tool: t
"""
        (tmp_ext_dir / "commands" / "test.yaml").write_text(yaml_content)

        loader = UserExtensionLoader(user_dir=tmp_ext_dir)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert "1 commands" in repr(result)

    @pytest.mark.asyncio
    async def test_repr_empty(
        self, tmp_path, cmd_registry, mock_agent_service, mock_tool_service
    ):
        loader = UserExtensionLoader(user_dir=tmp_path)
        result = await loader.load_all(
            cmd_registry, mock_agent_service, mock_tool_service
        )

        assert repr(result) == "no extensions"
