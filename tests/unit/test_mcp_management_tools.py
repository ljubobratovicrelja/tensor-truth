"""Tests for MCP management orchestrator tools.

Tests the three MCP management tool wrappers:
- list_mcp_servers: returns current server configurations
- get_mcp_presets: returns preset templates
- manage_mcp_server: requests confirmation, blocks, then applies
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensortruth.services.orchestrator_tool_wrappers import (
    create_get_mcp_presets_tool,
    create_list_mcp_servers_tool,
    create_manage_mcp_server_tool,
)
from tensortruth.services.tool_confirmation_service import ToolConfirmationService

VERIFY_PATCH = "tensortruth.services.orchestrator_tool_wrappers._verify_mcp_server"


@pytest.fixture
def mcp_server_service():
    """Mock MCPServerService."""
    svc = MagicMock()
    svc.list_all.return_value = [
        {
            "name": "search_web",
            "type": "stdio",
            "command": "node",
            "args": ["server.js"],
            "enabled": True,
            "builtin": True,
        },
        {
            "name": "my-server",
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "my-pkg"],
            "enabled": True,
            "builtin": False,
        },
    ]
    svc.get_presets.return_value = {
        "context7": {
            "name": "context7",
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp@latest"],
            "description": "Context7",
            "enabled": True,
        },
    }
    svc.add.return_value = {"name": "test-server", "builtin": False}
    svc.update.return_value = {"name": "test-server", "builtin": False}
    svc.remove.return_value = None
    return svc


@pytest.fixture
def confirmation_service():
    """Real ToolConfirmationService instance."""
    return ToolConfirmationService()


@pytest.fixture
def tool_service():
    """Mock ToolService."""
    svc = MagicMock()
    svc.reload = AsyncMock()
    return svc


@pytest.fixture
def progress_emitter():
    """Mock progress emitter."""
    return AsyncMock()


def _auto_approve(confirmation_service, delay=0.05):
    """Create a task that auto-approves the next pending confirmation."""

    async def _approver():
        await asyncio.sleep(delay)
        for c in confirmation_service._confirmations.values():
            if c.status == "pending":
                confirmation_service.resolve(c.confirmation_id, "approved")
                return

    return asyncio.create_task(_approver())


def _auto_reject(confirmation_service, delay=0.05):
    """Create a task that auto-rejects the next pending confirmation."""

    async def _rejector():
        await asyncio.sleep(delay)
        for c in confirmation_service._confirmations.values():
            if c.status == "pending":
                confirmation_service.resolve(c.confirmation_id, "rejected")
                return

    return asyncio.create_task(_rejector())


class TestListMCPServersTool:
    @pytest.mark.asyncio
    async def test_returns_json_list_of_servers(
        self, mcp_server_service, progress_emitter
    ):
        tool = create_list_mcp_servers_tool(mcp_server_service, progress_emitter)
        result = await tool.acall()
        parsed = json.loads(str(result))
        assert len(parsed) == 2
        assert parsed[0]["name"] == "search_web"
        assert parsed[1]["name"] == "my-server"

    @pytest.mark.asyncio
    async def test_emits_listing_progress(self, mcp_server_service, progress_emitter):
        tool = create_list_mcp_servers_tool(mcp_server_service, progress_emitter)
        await tool.acall()
        progress_emitter.assert_awaited_once()
        tp = progress_emitter.call_args[0][0]
        assert tp.tool_id == "list_mcp_servers"
        assert tp.phase == "listing"

    def test_tool_metadata(self, mcp_server_service, progress_emitter):
        tool = create_list_mcp_servers_tool(mcp_server_service, progress_emitter)
        assert tool.metadata.name == "list_mcp_servers"
        assert "MCP" in tool.metadata.description


class TestGetMCPPresetsTool:
    @pytest.mark.asyncio
    async def test_returns_presets_json(self, mcp_server_service, progress_emitter):
        tool = create_get_mcp_presets_tool(mcp_server_service, progress_emitter)
        result = await tool.acall()
        parsed = json.loads(str(result))
        assert "context7" in parsed
        assert parsed["context7"]["command"] == "npx"

    @pytest.mark.asyncio
    async def test_emits_listing_progress(self, mcp_server_service, progress_emitter):
        tool = create_get_mcp_presets_tool(mcp_server_service, progress_emitter)
        await tool.acall()
        progress_emitter.assert_awaited_once()
        tp = progress_emitter.call_args[0][0]
        assert tp.tool_id == "get_mcp_presets"

    def test_tool_metadata(self, mcp_server_service, progress_emitter):
        tool = create_get_mcp_presets_tool(mcp_server_service, progress_emitter)
        assert tool.metadata.name == "get_mcp_presets"


class TestManageMCPServerTool:
    @pytest.fixture(autouse=True)
    def _mock_verify(self):
        """Auto-mock verification for all propose tests (no real processes)."""
        with patch(VERIFY_PATCH, return_value=(True, "Server verified: 3 tools")) as m:
            self._verify_mock = m
            yield

    @pytest.mark.asyncio
    async def test_approved_add_applies_and_returns_success(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        _auto_approve(confirmation_service)
        result = await tool.acall(
            action="add",
            name="context7",
            command="npx",
            args=["-y", "@upstash/context7-mcp@latest"],
            summary="Add Context7 server",
        )
        result_str = str(result)
        assert "Successfully" in result_str
        mcp_server_service.add.assert_called_once()
        tool_service.reload.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejected_returns_rejection_message(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        _auto_reject(confirmation_service)
        result = await tool.acall(
            action="add",
            name="test",
            command="node",
            args=["server.js"],
            summary="Add test server",
        )
        assert "rejected" in str(result).lower()
        mcp_server_service.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_emits_confirmation_request_event(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        _auto_approve(confirmation_service)
        await tool.acall(
            action="add",
            name="test",
            command="node",
            args=["server.js"],
            summary="Add test server",
        )
        # Should have emitted a confirmation_request phase
        calls = progress_emitter.call_args_list
        conf_calls = [c for c in calls if c[0][0].phase == "confirmation_request"]
        assert len(conf_calls) == 1
        tp = conf_calls[0][0][0]
        assert tp.metadata["action_type"] == "mcp_add"
        assert tp.metadata["tool_name"] == "manage_mcp_server"
        assert "confirmation_id" in tp.metadata

    @pytest.mark.asyncio
    async def test_validates_invalid_action(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        result = await tool.acall(
            action="invalid",
            name="test",
            summary="Invalid action",
        )
        assert "Error" in str(result)

    @pytest.mark.asyncio
    async def test_validates_stdio_requires_command(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        result = await tool.acall(
            action="add",
            name="test",
            type="stdio",
            summary="Missing command",
        )
        assert "command" in str(result).lower()

    @pytest.mark.asyncio
    async def test_validates_sse_requires_url(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        result = await tool.acall(
            action="add",
            name="test",
            type="sse",
            summary="Missing URL",
        )
        assert "url" in str(result).lower()

    @pytest.mark.asyncio
    async def test_validates_remove_server_exists(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        mcp_server_service.list_all.return_value = []
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        result = await tool.acall(
            action="remove",
            name="nonexistent",
            summary="Remove nonexistent",
        )
        assert "not found" in str(result).lower()

    @pytest.mark.asyncio
    async def test_validates_remove_not_builtin(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        mcp_server_service.list_all.return_value = [
            {"name": "search_web", "builtin": True}
        ]
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        result = await tool.acall(
            action="remove",
            name="search_web",
            summary="Remove built-in",
        )
        assert "built-in" in str(result).lower()

    @pytest.mark.asyncio
    async def test_auto_fills_from_preset_when_fields_missing(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        """When name matches a preset and command/url are omitted, auto-fill."""
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        _auto_approve(confirmation_service)
        result = await tool.acall(
            action="add",
            name="context7",
            summary="Add Context7 server",
        )
        result_str = str(result)
        # Should succeed (auto-filled from preset), not error about missing command
        assert "Successfully" in result_str

        # Verify the confirmation has the preset config in details
        conf_calls = [
            c
            for c in progress_emitter.call_args_list
            if c[0][0].phase == "confirmation_request"
        ]
        assert len(conf_calls) == 1
        details = conf_calls[0][0][0].metadata["details"]
        config = details["config"]
        assert config["command"] == "npx"
        assert "@upstash/context7-mcp@latest" in config["args"]

    @pytest.mark.asyncio
    async def test_no_auto_fill_when_command_provided(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        """When command is explicitly provided, don't override with preset."""
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        _auto_approve(confirmation_service)
        result = await tool.acall(
            action="add",
            name="context7",
            command="custom-cmd",
            args=["--flag"],
            summary="Add with custom command",
        )
        result_str = str(result)
        assert "Successfully" in result_str

        conf_calls = [
            c
            for c in progress_emitter.call_args_list
            if c[0][0].phase == "confirmation_request"
        ]
        details = conf_calls[0][0][0].metadata["details"]
        config = details["config"]
        assert config["command"] == "custom-cmd"
        assert config["args"] == ["--flag"]

    @pytest.mark.asyncio
    async def test_infers_npx_command_from_args_with_y_flag(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        """When LLM passes args=['-y', '<pkg>'] but command=null, infer npx."""
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        _auto_approve(confirmation_service)
        result = await tool.acall(
            action="add",
            name="sequential-thinking",
            args=["-y", "@spences10/mcp-sequential-thinking"],
            summary="Add Sequential Thinking server",
        )
        result_str = str(result)
        assert "Successfully" in result_str

        conf_calls = [
            c
            for c in progress_emitter.call_args_list
            if c[0][0].phase == "confirmation_request"
        ]
        details = conf_calls[0][0][0].metadata["details"]
        config = details["config"]
        assert config["command"] == "npx"
        assert "-y" in config["args"]

    @pytest.mark.asyncio
    async def test_infers_npx_command_from_scoped_package_arg(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        """When LLM passes args=['@org/pkg'] without -y, infer npx -y."""
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        _auto_approve(confirmation_service)
        result = await tool.acall(
            action="add",
            name="some-mcp",
            args=["@org/some-mcp-server"],
            summary="Add some MCP server",
        )
        result_str = str(result)
        assert "Successfully" in result_str

        conf_calls = [
            c
            for c in progress_emitter.call_args_list
            if c[0][0].phase == "confirmation_request"
        ]
        details = conf_calls[0][0][0].metadata["details"]
        config = details["config"]
        assert config["command"] == "npx"
        assert config["args"][0] == "-y"

    @pytest.mark.asyncio
    async def test_blocks_identical_retry_after_failure(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        """Second identical failing call is blocked with a retry error."""
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        # First call: fails (no command, no args to infer from)
        result1 = await tool.acall(
            action="add",
            name="bad-server",
            summary="Will fail",
        )
        assert "Error" in str(result1)

        # Second identical call: blocked as retry
        result2 = await tool.acall(
            action="add",
            name="bad-server",
            summary="Will fail",
        )
        assert "same call" in str(result2).lower()

        # Third call allowed again (reset after block)
        result3 = await tool.acall(
            action="add",
            name="bad-server",
            summary="Will fail",
        )
        assert "command" in str(result3).lower()

    @pytest.mark.asyncio
    async def test_verification_called_for_add(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        """Verification is called when adding a server."""
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        _auto_approve(confirmation_service)
        await tool.acall(
            action="add",
            name="test-server",
            command="npx",
            args=["-y", "test-pkg"],
            summary="Add test server",
        )
        self._verify_mock.assert_awaited_once()
        config = self._verify_mock.call_args[0][0]
        assert config["command"] == "npx"
        assert config["name"] == "test-server"

    @pytest.mark.asyncio
    async def test_verification_failure_blocks_confirmation(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        """When verification fails, no confirmation is created."""
        self._verify_mock.return_value = (
            False,
            "Server 'bad-pkg' timed out after 15s.",
        )
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        result = await tool.acall(
            action="add",
            name="bad-pkg",
            command="npx",
            args=["-y", "bad-pkg"],
            summary="Add bad server",
        )
        result_str = str(result)
        assert "Error" in result_str
        assert "timed out" in result_str

        # No confirmation_request should have been emitted
        conf_calls = [
            c
            for c in progress_emitter.call_args_list
            if c[0][0].phase == "confirmation_request"
        ]
        assert len(conf_calls) == 0

    @pytest.mark.asyncio
    async def test_verification_not_called_for_remove(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        """Verification is skipped for remove actions."""
        mcp_server_service.list_all.return_value = [
            {"name": "my-server", "builtin": False}
        ]
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        _auto_approve(confirmation_service)
        result = await tool.acall(
            action="remove",
            name="my-server",
            summary="Remove my server",
        )
        assert "Successfully" in str(result)
        self._verify_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_approved_add_calls_service(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        """After approval, the tool directly calls mcp_server_service.add()."""
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        _auto_approve(confirmation_service)
        result = await tool.acall(
            action="add",
            name="test-server",
            command="npx",
            args=["-y", "test-pkg"],
            summary="Add test",
        )
        assert "Successfully" in str(result)
        mcp_server_service.add.assert_called_once()

    def test_tool_metadata(
        self, confirmation_service, mcp_server_service, tool_service, progress_emitter
    ):
        tool = create_manage_mcp_server_tool(
            confirmation_service,
            mcp_server_service,
            tool_service,
            progress_emitter,
            "sess-1",
        )
        assert tool.metadata.name == "manage_mcp_server"
        assert "add, update, or remove" in tool.metadata.description.lower()


class TestVerifyMCPServer:
    """Tests for _verify_mcp_server with mocked BasicMCPClient."""

    @pytest.fixture
    def progress_emitter(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_successful_stdio_verification(self, progress_emitter):
        from tensortruth.services.orchestrator_tool_wrappers import _verify_mcp_server

        mock_tool = MagicMock()
        mock_tool.name = "do_thing"
        mock_result = MagicMock()
        mock_result.tools = [mock_tool]

        with patch(
            "tensortruth.services.orchestrator_tool_wrappers.BasicMCPClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.list_tools = AsyncMock(return_value=mock_result)

            ok, msg = await _verify_mcp_server(
                {
                    "name": "test",
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "pkg"],
                },
                progress_emitter,
            )

        assert ok is True
        assert "1 tools" in msg
        assert "do_thing" in msg

    @pytest.mark.asyncio
    async def test_failed_verification_returns_error(self, progress_emitter):
        from tensortruth.services.orchestrator_tool_wrappers import _verify_mcp_server

        with patch(
            "tensortruth.services.orchestrator_tool_wrappers.BasicMCPClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.list_tools = AsyncMock(
                side_effect=RuntimeError("npm ERR! 404 Not Found: @fake/pkg")
            )

            ok, msg = await _verify_mcp_server(
                {
                    "name": "test",
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@fake/pkg"],
                },
                progress_emitter,
            )

        assert ok is False
        assert "404" in msg or "verification failed" in msg.lower()

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, progress_emitter):
        import asyncio as aio

        from tensortruth.services.orchestrator_tool_wrappers import _verify_mcp_server

        async def slow_list():
            await aio.sleep(100)

        with patch(
            "tensortruth.services.orchestrator_tool_wrappers.BasicMCPClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.list_tools = slow_list

            ok, msg = await _verify_mcp_server(
                {
                    "name": "slow",
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "pkg"],
                },
                progress_emitter,
                timeout=0.1,
            )

        assert ok is False
        assert "timed out" in msg.lower()

    @pytest.mark.asyncio
    async def test_file_not_found_returns_error(self, progress_emitter):
        from tensortruth.services.orchestrator_tool_wrappers import _verify_mcp_server

        with patch(
            "tensortruth.services.orchestrator_tool_wrappers.BasicMCPClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.list_tools = AsyncMock(
                side_effect=FileNotFoundError("No such file: 'badcmd'")
            )

            ok, msg = await _verify_mcp_server(
                {"name": "test", "type": "stdio", "command": "badcmd"},
                progress_emitter,
            )

        assert ok is False
        assert "not found" in msg.lower()

    @pytest.mark.asyncio
    async def test_sse_verification(self, progress_emitter):
        from tensortruth.services.orchestrator_tool_wrappers import _verify_mcp_server

        mock_result = MagicMock()
        mock_result.tools = []

        with patch(
            "tensortruth.services.orchestrator_tool_wrappers.BasicMCPClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.list_tools = AsyncMock(return_value=mock_result)

            ok, msg = await _verify_mcp_server(
                {"name": "test", "type": "sse", "url": "http://localhost:3000/sse"},
                progress_emitter,
            )

        assert ok is True
        assert "0 tools" in msg
        MockClient.assert_called_once_with(
            command_or_url="http://localhost:3000/sse", timeout=15
        )
