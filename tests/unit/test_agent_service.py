"""Tests for AgentService.

AgentService creates and executes LlamaIndex agents from configuration.
Does NOT subclass or wrap agents - creates FunctionAgent/ReActAgent directly.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensortruth.agents.config import AgentConfig
from tensortruth.services.agent_service import AgentCallbacks, AgentService


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_agent_config_defaults(self):
        """Should have correct default values."""
        config = AgentConfig(
            name="test",
            description="Test agent",
            tools=["tool1"],
            system_prompt="You are a test agent.",
        )

        assert config.name == "test"
        assert config.description == "Test agent"
        assert config.tools == ["tool1"]
        assert config.system_prompt == "You are a test agent."
        assert config.agent_type == "function"
        assert config.model is None
        assert config.max_iterations == 10

    def test_agent_config_custom_values(self):
        """Should accept custom values."""
        config = AgentConfig(
            name="browse",
            description="Browse agent",
            tools=["search_web", "fetch_page"],
            system_prompt="You are a web researcher.",
            agent_type="react",
            model="llama3.1:8b",
            max_iterations=20,
        )

        assert config.agent_type == "react"
        assert config.model == "llama3.1:8b"
        assert config.max_iterations == 20


class TestAgentCallbacks:
    """Test AgentCallbacks dataclass."""

    def test_callbacks_defaults(self):
        """Should have None defaults."""
        callbacks = AgentCallbacks()

        assert callbacks.on_progress is None
        assert callbacks.on_tool_call is None
        assert callbacks.on_token is None

    def test_callbacks_with_functions(self):
        """Should accept callback functions."""

        def progress_fn(msg):
            pass

        def tool_fn(name, params):
            pass

        def token_fn(token):
            pass

        callbacks = AgentCallbacks(
            on_progress=progress_fn,
            on_tool_call=tool_fn,
            on_token=token_fn,
        )

        assert callbacks.on_progress == progress_fn
        assert callbacks.on_tool_call == tool_fn
        assert callbacks.on_token == token_fn


class TestAgentServiceInit:
    """Test AgentService initialization."""

    def test_init_with_tool_service(self):
        """Should initialize with tool service and config."""
        mock_tool_service = MagicMock()
        config = {"ollama_url": "http://localhost:11434"}

        service = AgentService(tool_service=mock_tool_service, config=config)

        assert service._tool_service == mock_tool_service
        assert service._config == config
        assert service._agent_configs == {}

    def test_init_calls_load_builtin_agents(self):
        """Should call _load_builtin_agents on init."""
        mock_tool_service = MagicMock()

        with patch.object(AgentService, "_load_builtin_agents") as mock_load:
            AgentService(tool_service=mock_tool_service, config={})

        mock_load.assert_called_once()


class TestAgentServiceRegisterAgent:
    """Test AgentService.register_agent()."""

    def test_register_agent_adds_config(self):
        """Should add agent config to registry."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="test_agent",
            description="Test",
            tools=["tool1"],
            system_prompt="Test prompt",
        )
        service.register_agent(config)

        assert "test_agent" in service._agent_configs
        assert service._agent_configs["test_agent"] == config

    def test_register_agent_overwrites_existing(self):
        """Should overwrite existing agent with same name."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        config1 = AgentConfig(
            name="test",
            description="First",
            tools=["tool1"],
            system_prompt="First prompt",
        )
        config2 = AgentConfig(
            name="test",
            description="Second",
            tools=["tool2"],
            system_prompt="Second prompt",
        )

        service.register_agent(config1)
        service.register_agent(config2)

        assert service._agent_configs["test"].description == "Second"


class TestAgentServiceListAgents:
    """Test AgentService.list_agents()."""

    def test_list_agents_empty(self):
        """Should return empty list when no agents registered."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        result = service.list_agents()

        assert result == []

    def test_list_agents_returns_metadata(self):
        """Should return agent metadata for API."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="browse",
            description="Browse the web",
            tools=["search_web", "fetch_page"],
            system_prompt="You are a researcher.",
        )
        service.register_agent(config)

        result = service.list_agents()

        assert len(result) == 1
        assert result[0]["name"] == "browse"
        assert result[0]["description"] == "Browse the web"
        assert result[0]["tools"] == ["search_web", "fetch_page"]


class TestAgentServiceCreateLLM:
    """Test AgentService._create_llm()."""

    def test_create_llm_with_default_url(self):
        """Should create Ollama LLM with default URL."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        with patch("tensortruth.services.agent_service.Ollama") as mock_ollama:
            service._create_llm("llama3.1:8b")

        mock_ollama.assert_called_once_with(
            model="llama3.1:8b",
            base_url="http://localhost:11434",
            temperature=0.2,
            request_timeout=120.0,
        )

    def test_create_llm_with_custom_url(self):
        """Should use custom Ollama URL from config."""
        mock_tool_service = MagicMock()
        config = {"ollama_url": "http://custom:11434"}
        service = AgentService(tool_service=mock_tool_service, config=config)

        with patch("tensortruth.services.agent_service.Ollama") as mock_ollama:
            service._create_llm("llama3.1:8b")

        mock_ollama.assert_called_once_with(
            model="llama3.1:8b",
            base_url="http://custom:11434",
            temperature=0.2,
            request_timeout=120.0,
        )


class TestAgentServiceCreateAgent:
    """Test AgentService._create_agent()."""

    def test_create_function_agent(self):
        """Should create FunctionAgent for function type."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="test",
            description="Test",
            tools=["tool1"],
            system_prompt="Test prompt",
            agent_type="function",
        )
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        with patch("tensortruth.services.agent_service.FunctionAgent") as mock_fn_agent:
            service._create_agent(config, mock_tools, mock_llm)

        mock_fn_agent.assert_called_once_with(
            tools=mock_tools,
            llm=mock_llm,
            system_prompt="Test prompt",
        )

    def test_create_react_agent(self):
        """Should create ReActAgent for react type."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="test",
            description="Test",
            tools=["tool1"],
            system_prompt="Test prompt",
            agent_type="react",
        )
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        with patch("tensortruth.services.agent_service.ReActAgent") as mock_react:
            service._create_agent(config, mock_tools, mock_llm)

        mock_react.assert_called_once_with(
            tools=mock_tools,
            llm=mock_llm,
            system_prompt="Test prompt",
            verbose=True,
        )


class TestAgentServiceRun:
    """Test AgentService.run()."""

    @pytest.mark.asyncio
    async def test_run_unknown_agent(self):
        """Should return error for unknown agent."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        result = await service.run(
            agent_name="nonexistent",
            goal="test goal",
            callbacks=AgentCallbacks(),
            session_params={},
        )

        assert result.error is not None
        assert "Unknown agent" in result.error

    @pytest.mark.asyncio
    async def test_run_missing_tools(self):
        """Should return error when required tools are missing."""
        mock_tool_service = MagicMock()
        mock_tool_service.get_tools_by_names.return_value = []
        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="test",
            description="Test",
            tools=["missing_tool"],
            system_prompt="Test",
        )
        service.register_agent(config)

        result = await service.run(
            agent_name="test",
            goal="test goal",
            callbacks=AgentCallbacks(),
            session_params={},
        )

        assert result.error is not None
        assert "Missing tools" in result.error

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Should execute agent and return result."""
        mock_tool_service = MagicMock()
        mock_tool = MagicMock()
        mock_tool.metadata.name = "tool1"
        mock_tool_service.get_tools_by_names.return_value = [mock_tool]

        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="test",
            description="Test",
            tools=["tool1"],
            system_prompt="Test",
        )
        service.register_agent(config)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="Agent response")

        with (
            patch.object(service, "_create_llm") as mock_create_llm,
            patch.object(service, "_create_agent", return_value=mock_agent),
        ):
            mock_create_llm.return_value = MagicMock()

            result = await service.run(
                agent_name="test",
                goal="test goal",
                callbacks=AgentCallbacks(),
                session_params={"model": "llama3.1:8b"},
            )

        assert result.final_answer == "Agent response"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_run_uses_config_model(self):
        """Should use model from config if specified."""
        mock_tool_service = MagicMock()
        mock_tool = MagicMock()
        mock_tool.metadata.name = "tool1"
        mock_tool_service.get_tools_by_names.return_value = [mock_tool]

        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="test",
            description="Test",
            tools=["tool1"],
            system_prompt="Test",
            model="custom-model",  # Specific model override
        )
        service.register_agent(config)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="response")

        with (
            patch.object(service, "_create_llm") as mock_create_llm,
            patch.object(service, "_create_agent", return_value=mock_agent),
        ):
            mock_create_llm.return_value = MagicMock()

            await service.run(
                agent_name="test",
                goal="goal",
                callbacks=AgentCallbacks(),
                session_params={"model": "session-model"},
            )

        # Should use config model, not session model
        mock_create_llm.assert_called_once_with("custom-model")

    @pytest.mark.asyncio
    async def test_run_calls_progress_callback(self):
        """Should call on_progress callback."""
        mock_tool_service = MagicMock()
        mock_tool = MagicMock()
        mock_tool.metadata.name = "tool1"
        mock_tool_service.get_tools_by_names.return_value = [mock_tool]

        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="test",
            description="Test",
            tools=["tool1"],
            system_prompt="Test",
        )
        service.register_agent(config)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value="response")

        progress_calls = []

        def on_progress(msg):
            progress_calls.append(msg)

        callbacks = AgentCallbacks(on_progress=on_progress)

        with (
            patch.object(service, "_create_llm"),
            patch.object(service, "_create_agent", return_value=mock_agent),
        ):
            await service.run(
                agent_name="test",
                goal="goal",
                callbacks=callbacks,
                session_params={},
            )

        assert len(progress_calls) > 0
        assert "test" in progress_calls[0]

    @pytest.mark.asyncio
    async def test_run_handles_exception(self):
        """Should return error when agent execution fails."""
        mock_tool_service = MagicMock()
        mock_tool = MagicMock()
        mock_tool.metadata.name = "tool1"
        mock_tool_service.get_tools_by_names.return_value = [mock_tool]

        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="test",
            description="Test",
            tools=["tool1"],
            system_prompt="Test",
        )
        service.register_agent(config)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("Agent failed"))

        with (
            patch.object(service, "_create_llm"),
            patch.object(service, "_create_agent", return_value=mock_agent),
        ):
            result = await service.run(
                agent_name="test",
                goal="goal",
                callbacks=AgentCallbacks(),
                session_params={},
            )

        assert result.error is not None
        assert "Agent failed" in result.error


class TestAgentServiceWrapToolsForCallbacks:
    """Test AgentService._wrap_tools_for_callbacks()."""

    def test_wrap_tools_no_callback(self):
        """Should return tools unchanged when no callback."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        tools = [MagicMock(), MagicMock()]
        callbacks = AgentCallbacks()  # No on_tool_call

        result = service._wrap_tools_for_callbacks(tools, callbacks)

        # When no callback, should return original tools
        assert result == tools

    @pytest.mark.asyncio
    async def test_wrap_tools_with_callback(self):
        """Should wrap tools to emit callback on call."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        # Create mock tool
        mock_tool = MagicMock()
        mock_tool.metadata.name = "search_web"
        mock_tool.metadata.description = "Search the web"
        mock_tool.fn = None
        mock_tool.async_fn = AsyncMock(return_value="result")

        tool_calls = []

        def on_tool_call(name, params):
            tool_calls.append((name, params))

        callbacks = AgentCallbacks(on_tool_call=on_tool_call)

        wrapped = service._wrap_tools_for_callbacks([mock_tool], callbacks)

        # Should have created new wrapped tool
        assert len(wrapped) == 1

        # Call the wrapped tool
        result = await wrapped[0].async_fn(query="test")

        # Should have called callback
        assert len(tool_calls) == 1
        assert tool_calls[0][0] == "search_web"
        assert tool_calls[0][1] == {"query": "test"}

        # Should have returned original result
        assert result == "result"
