"""Tests for YamlCommand and YamlAgentCommand execution."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensortruth.agents.config import AgentResult
from tensortruth.extensions.schema import CommandSpec, StepSpec
from tensortruth.extensions.yaml_command import YamlAgentCommand, YamlCommand


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


class TestYamlAgentCommand:
    """Tests for YamlAgentCommand tool_steps wiring."""

    @pytest.fixture
    def websocket(self):
        ws = AsyncMock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.fixture
    def mock_agent_service(self):
        svc = MagicMock()
        svc.run = AsyncMock()
        return svc

    @pytest.fixture
    def mock_config_service(self):
        config = MagicMock()
        config.llm.default_model = "llama3.2:3b"
        config.ollama.base_url = "http://localhost:11434"
        config.llm.default_context_window = 4096
        svc = MagicMock()
        svc.load.return_value = config
        return svc

    @pytest.mark.asyncio
    async def test_tool_steps_in_done_message(
        self, websocket, mock_agent_service, mock_config_service
    ):
        """tool_steps appears in done message when agent returns tool_steps."""
        tool_steps = [
            {
                "tool": "search",
                "params": {"q": "test"},
                "output": "found it",
                "is_error": False,
            },
        ]

        # The agent service run will capture the callbacks and fire them
        async def fake_run(agent_name, goal, callbacks, session_params):
            # Simulate on_tool_call_result being fired
            if callbacks.on_tool_call_result:
                callbacks.on_tool_call_result(
                    "search", {"q": "test"}, "found it", False
                )
            return AgentResult(
                final_answer="The answer",
                tool_steps=tool_steps,
            )

        mock_agent_service.run = fake_run

        spec = CommandSpec(
            name="research",
            description="Research docs",
            agent="research_docs",
        )
        cmd = YamlAgentCommand(spec)
        session = {"params": {}}

        with (
            patch(
                "tensortruth.api.deps.get_agent_service",
                return_value=mock_agent_service,
            ),
            patch(
                "tensortruth.services.config_service.ConfigService",
                return_value=mock_config_service,
            ),
        ):
            await cmd.execute("pytorch tensors", session, websocket)

        # Find the done message
        done_calls = [
            c
            for c in websocket.send_json.call_args_list
            if isinstance(c.args[0], dict) and c.args[0].get("type") == "done"
        ]
        assert len(done_calls) == 1
        done_msg = done_calls[0].args[0]
        assert done_msg["content"] == "The answer"
        assert "tool_steps" in done_msg
        assert len(done_msg["tool_steps"]) == 1
        assert done_msg["tool_steps"][0]["tool"] == "search"

    @pytest.mark.asyncio
    async def test_no_tool_steps_when_empty(
        self, websocket, mock_agent_service, mock_config_service
    ):
        """tool_steps is omitted from done message when no tools were called."""
        mock_agent_service.run = AsyncMock(
            return_value=AgentResult(final_answer="Direct answer")
        )

        spec = CommandSpec(
            name="research",
            description="Research docs",
            agent="research_docs",
        )
        cmd = YamlAgentCommand(spec)
        session = {"params": {}}

        with (
            patch(
                "tensortruth.api.deps.get_agent_service",
                return_value=mock_agent_service,
            ),
            patch(
                "tensortruth.services.config_service.ConfigService",
                return_value=mock_config_service,
            ),
        ):
            await cmd.execute("pytorch tensors", session, websocket)

        done_calls = [
            c
            for c in websocket.send_json.call_args_list
            if isinstance(c.args[0], dict) and c.args[0].get("type") == "done"
        ]
        assert len(done_calls) == 1
        assert "tool_steps" not in done_calls[0].args[0]

    @pytest.mark.asyncio
    async def test_tool_progress_sent_on_result(
        self, websocket, mock_agent_service, mock_config_service
    ):
        """tool_progress messages are sent when on_tool_call_result fires."""

        async def fake_run(agent_name, goal, callbacks, session_params):
            if callbacks.on_tool_call_result:
                callbacks.on_tool_call_result("search", {"q": "test"}, "results", False)
            return AgentResult(final_answer="Done")

        mock_agent_service.run = fake_run

        spec = CommandSpec(
            name="research",
            description="Research docs",
            agent="research_docs",
        )
        cmd = YamlAgentCommand(spec)
        session = {"params": {}}

        with (
            patch(
                "tensortruth.api.deps.get_agent_service",
                return_value=mock_agent_service,
            ),
            patch(
                "tensortruth.services.config_service.ConfigService",
                return_value=mock_config_service,
            ),
        ):
            await cmd.execute("test query", session, websocket)

        # Find tool_progress messages with action "completed"
        tool_progress_calls = [
            c
            for c in websocket.send_json.call_args_list
            if isinstance(c.args[0], dict)
            and c.args[0].get("type") == "tool_progress"
            and c.args[0].get("action") == "completed"
        ]
        assert len(tool_progress_calls) == 1
        msg = tool_progress_calls[0].args[0]
        assert msg["tool"] == "search"
        assert msg["output"] == "results"
        assert msg["is_error"] is False

    @pytest.mark.asyncio
    async def test_empty_args_sends_error(self, websocket, mock_agent_service):
        """Empty args sends usage error."""
        spec = CommandSpec(
            name="research",
            description="Research docs",
            usage="/research <query>",
            agent="research_docs",
        )
        cmd = YamlAgentCommand(spec)

        await cmd.execute("", {}, websocket)

        error_calls = [
            c
            for c in websocket.send_json.call_args_list
            if isinstance(c.args[0], dict) and c.args[0].get("type") == "error"
        ]
        assert len(error_calls) == 1
