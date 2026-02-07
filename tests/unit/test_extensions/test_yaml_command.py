"""Tests for YamlCommand execution."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensortruth.extensions.schema import CommandSpec, StepSpec
from tensortruth.extensions.yaml_command import YamlCommand


@pytest.fixture
def websocket():
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


@pytest.fixture
def mock_tool_service():
    svc = MagicMock()
    svc.execute_tool = AsyncMock()
    return svc


class TestYamlCommand:
    def test_init_from_spec(self):
        spec = CommandSpec(
            name="test",
            description="Test cmd",
            usage="/test <arg>",
            aliases=["t"],
            steps=[StepSpec(tool="my-tool", params={"q": "{{args}}"})],
        )
        cmd = YamlCommand(spec)
        assert cmd.name == "test"
        assert cmd.aliases == ["t"]
        assert cmd.usage == "/test <arg>"

    def test_default_usage(self):
        spec = CommandSpec(
            name="foo",
            description="Foo",
            steps=[StepSpec(tool="t")],
        )
        cmd = YamlCommand(spec)
        assert cmd.usage == "/foo <args>"

    @pytest.mark.asyncio
    async def test_single_step_pipeline(self, websocket, mock_tool_service):
        mock_tool_service.execute_tool.return_value = {
            "success": True,
            "data": "result data",
        }

        spec = CommandSpec(
            name="test",
            description="Test",
            steps=[StepSpec(tool="my-tool", params={"q": "{{args}}"})],
        )
        cmd = YamlCommand(spec)

        with patch(
            "tensortruth.api.deps.get_tool_service",
            return_value=mock_tool_service,
        ):
            await cmd.execute("hello world", {}, websocket)

        # Should have called execute_tool with resolved params
        mock_tool_service.execute_tool.assert_awaited_once_with(
            "my-tool", {"q": "hello world"}
        )

        # Should have sent progress and done
        calls = websocket.send_json.call_args_list
        progress_calls = [c for c in calls if c.args[0].get("type") == "agent_progress"]
        done_calls = [c for c in calls if c.args[0].get("type") == "done"]
        assert len(progress_calls) == 1
        assert len(done_calls) == 1
        assert done_calls[0].args[0]["content"] == "result data"

    @pytest.mark.asyncio
    async def test_multi_step_pipeline_with_result_var(
        self, websocket, mock_tool_service
    ):
        """Test a two-step pipeline like Context7."""
        mock_tool_service.execute_tool.side_effect = [
            {
                "success": True,
                "data": json.dumps({"libraryID": "/pytorch/pytorch"}),
            },
            {
                "success": True,
                "data": "Documentation content here",
            },
        ]

        spec = CommandSpec(
            name="c7",
            description="Context7",
            steps=[
                StepSpec(
                    tool="resolve-library-id",
                    params={"libraryName": "{{args.0}}"},
                    result_var="resolved",
                ),
                StepSpec(
                    tool="get-library-docs",
                    params={
                        "context7CompatibleLibraryID": "{{resolved.libraryID}}",
                        "topic": "{{args.rest}}",
                    },
                ),
            ],
        )
        cmd = YamlCommand(spec)

        with patch(
            "tensortruth.api.deps.get_tool_service",
            return_value=mock_tool_service,
        ):
            await cmd.execute("pytorch batch normalization", {}, websocket)

        # Verify first call
        first_call = mock_tool_service.execute_tool.call_args_list[0]
        assert first_call.args == ("resolve-library-id", {"libraryName": "pytorch"})

        # Verify second call with resolved variable
        second_call = mock_tool_service.execute_tool.call_args_list[1]
        assert second_call.args == (
            "get-library-docs",
            {
                "context7CompatibleLibraryID": "/pytorch/pytorch",
                "topic": "batch normalization",
            },
        )

        # Final result
        done_calls = [
            c
            for c in websocket.send_json.call_args_list
            if c.args[0].get("type") == "done"
        ]
        assert done_calls[0].args[0]["content"] == "Documentation content here"

    @pytest.mark.asyncio
    async def test_tool_failure_sends_error(self, websocket, mock_tool_service):
        mock_tool_service.execute_tool.return_value = {
            "success": False,
            "error": "Tool not found",
        }

        spec = CommandSpec(
            name="test",
            description="Test",
            steps=[StepSpec(tool="missing-tool")],
        )
        cmd = YamlCommand(spec)

        with patch(
            "tensortruth.api.deps.get_tool_service",
            return_value=mock_tool_service,
        ):
            await cmd.execute("args", {}, websocket)

        error_calls = [
            c
            for c in websocket.send_json.call_args_list
            if c.args[0].get("type") == "error"
        ]
        assert len(error_calls) == 1
        assert "missing-tool" in error_calls[0].args[0]["detail"]

    @pytest.mark.asyncio
    async def test_template_error_sends_error(self, websocket, mock_tool_service):
        spec = CommandSpec(
            name="test",
            description="Test",
            steps=[StepSpec(tool="t", params={"q": "{{nonexistent}}"})],
        )
        cmd = YamlCommand(spec)

        with patch(
            "tensortruth.api.deps.get_tool_service",
            return_value=mock_tool_service,
        ):
            await cmd.execute("args", {}, websocket)

        error_calls = [
            c
            for c in websocket.send_json.call_args_list
            if c.args[0].get("type") == "error"
        ]
        assert len(error_calls) == 1
        assert "Template error" in error_calls[0].args[0]["detail"]
