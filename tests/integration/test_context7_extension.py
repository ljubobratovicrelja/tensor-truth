"""Integration tests for Context7 extension loading and execution.

These tests require:
1. The Context7 MCP server to be available (npx @upstash/context7-mcp)
2. Network connectivity

Run with: pytest tests/integration/test_context7_extension.py --run-mcp --run-network -v
"""

import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tensortruth.api.routes.commands import CommandRegistry
from tensortruth.extensions import load_user_extensions
from tensortruth.extensions.yaml_command import YamlCommand
from tensortruth.services.tool_service import ToolService

# Location of the YAML extension in the repo
EXTENSION_LIBRARY = Path(__file__).parent.parent.parent / "extension_library"


@pytest.mark.requires_mcp
@pytest.mark.requires_network
class TestContext7ExtensionLoading:
    """Test that the Context7 YAML extension loads correctly."""

    @pytest.mark.asyncio
    async def test_context7_yaml_loads(self, tmp_path):
        """Verify context7.yaml loads and registers the command."""
        ext_dir = tmp_path / "ext"
        (ext_dir / "commands").mkdir(parents=True)
        shutil.copy(
            EXTENSION_LIBRARY / "commands" / "context7.yaml",
            ext_dir / "commands" / "context7.yaml",
        )

        registry = CommandRegistry()
        from unittest.mock import MagicMock

        mock_agent_svc = MagicMock()
        mock_tool_svc = MagicMock()

        result = await load_user_extensions(
            command_registry=registry,
            agent_service=mock_agent_svc,
            tool_service=mock_tool_svc,
            user_dir=ext_dir,
        )

        assert result.commands_loaded == 1
        assert len(result.errors) == 0

        # Primary name
        cmd = registry.get("context7")
        assert cmd is not None
        assert isinstance(cmd, YamlCommand)

        # Alias
        alias_cmd = registry.get("c7")
        assert alias_cmd is cmd

    @pytest.mark.asyncio
    async def test_context7_python_loads(self, tmp_path):
        """Verify context7.py loads and registers the command."""
        ext_dir = tmp_path / "ext"
        (ext_dir / "commands").mkdir(parents=True)
        shutil.copy(
            EXTENSION_LIBRARY / "commands" / "context7.py",
            ext_dir / "commands" / "context7.py",
        )

        registry = CommandRegistry()
        from unittest.mock import MagicMock

        mock_agent_svc = MagicMock()
        mock_tool_svc = MagicMock()

        result = await load_user_extensions(
            command_registry=registry,
            agent_service=mock_agent_svc,
            tool_service=mock_tool_svc,
            user_dir=ext_dir,
        )

        assert result.commands_loaded == 1
        cmd = registry.get("context7")
        assert cmd is not None

    @pytest.mark.asyncio
    async def test_doc_researcher_agent_loads(self, tmp_path):
        """Verify doc_researcher.yaml loads and registers the agent."""
        ext_dir = tmp_path / "ext"
        (ext_dir / "agents").mkdir(parents=True)
        shutil.copy(
            EXTENSION_LIBRARY / "agents" / "doc_researcher.yaml",
            ext_dir / "agents" / "doc_researcher.yaml",
        )

        registry = CommandRegistry()
        from unittest.mock import MagicMock

        mock_agent_svc = MagicMock()
        mock_tool_svc = MagicMock()

        result = await load_user_extensions(
            command_registry=registry,
            agent_service=mock_agent_svc,
            tool_service=mock_tool_svc,
            user_dir=ext_dir,
        )

        assert result.agents_loaded == 1
        assert len(result.errors) == 0
        mock_agent_svc.register_agent.assert_called_once()

        config = mock_agent_svc.register_agent.call_args.args[0]
        assert config.name == "doc_researcher"
        assert config.agent_type == "function"
        assert "search_web" in config.tools


@pytest.mark.requires_mcp
@pytest.mark.requires_network
class TestContext7ExtensionExecution:
    """Test actual Context7 MCP tool execution via the YAML command.

    These tests call the real Context7 MCP server and require:
    - npx available on PATH
    - Network connectivity to Context7 API
    - MCP server config in ~/.tensortruth/mcp_servers.json with context7 entry
    """

    @pytest.fixture
    async def tool_service_with_mcp(self):
        """Create a ToolService and load tools (including MCP)."""
        svc = ToolService()
        await svc.load_tools()
        return svc

    @pytest.fixture
    def context7_command(self, tmp_path):
        """Load the Context7 YAML command."""
        ext_dir = tmp_path / "ext"
        (ext_dir / "commands").mkdir(parents=True)
        shutil.copy(
            EXTENSION_LIBRARY / "commands" / "context7.yaml",
            ext_dir / "commands" / "context7.yaml",
        )

        import yaml

        from tensortruth.extensions.schema import CommandSpec

        raw = yaml.safe_load((ext_dir / "commands" / "context7.yaml").read_text())
        spec = CommandSpec(**raw)
        return YamlCommand(spec)

    @pytest.mark.asyncio
    async def test_resolve_library_tool_exists(self, tool_service_with_mcp):
        """Verify the Context7 MCP tools are available."""
        tool_names = [t.metadata.name for t in tool_service_with_mcp.tools]
        assert (
            "resolve-library-id" in tool_names
        ), f"resolve-library-id not found. Available tools: {tool_names}"
        assert (
            "get-library-docs" in tool_names
        ), f"get-library-docs not found. Available tools: {tool_names}"

    @pytest.mark.asyncio
    async def test_resolve_library_id(self, tool_service_with_mcp):
        """Test resolving a library name to a Context7 ID."""
        result = await tool_service_with_mcp.execute_tool(
            "resolve-library-id",
            {"libraryName": "pytorch"},
        )
        assert result["success"], f"Tool failed: {result.get('error')}"
        data = str(result["data"])
        # Should contain a library ID path
        assert "/" in data or "pytorch" in data.lower()

    @pytest.mark.asyncio
    async def test_full_context7_pipeline(
        self, tool_service_with_mcp, context7_command
    ):
        """End-to-end: run /context7 pytorch tensors via YAML command."""
        websocket = AsyncMock()
        websocket.send_json = AsyncMock()

        from unittest.mock import patch

        with patch(
            "tensortruth.api.deps.get_tool_service",
            return_value=tool_service_with_mcp,
        ):
            await context7_command.execute("pytorch tensors", {}, websocket)

        # Collect messages sent to websocket
        calls = websocket.send_json.call_args_list
        messages = [c.args[0] for c in calls]

        # Should have progress messages for each step
        progress_msgs = [m for m in messages if m.get("type") == "agent_progress"]
        assert (
            len(progress_msgs) >= 2
        ), f"Expected at least 2 progress messages, got {len(progress_msgs)}"

        # Should end with done
        done_msgs = [m for m in messages if m.get("type") == "done"]
        assert (
            len(done_msgs) == 1
        ), f"Expected 1 done message, got {len(done_msgs)}: {messages}"

        # The done content should contain documentation
        content = done_msgs[0]["content"]
        assert (
            len(content) > 50
        ), f"Expected substantial documentation content, got: {content[:100]}"

        # Should not have errors
        error_msgs = [m for m in messages if m.get("type") == "error"]
        assert len(error_msgs) == 0, f"Unexpected errors: {error_msgs}"
