"""Tests for refactored AgentService using factory registry."""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.agents.base import Agent
from tensortruth.agents.config import AgentCallbacks, AgentConfig, AgentResult
from tensortruth.services.agent_service import AgentService


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, name: str):
        self.name = name
        self.run_called = False
        self.run_query = None

    async def run(self, query, callbacks, **kwargs):
        self.run_called = True
        self.run_query = query
        return AgentResult(final_answer=f"Answer from {self.name}")

    def get_metadata(self):
        return {"name": self.name, "type": "mock"}


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_agent_config_defaults(self):
        """Should have correct default values."""
        config = AgentConfig(
            name="test",
            description="Test agent",
            tools=["tool1"],
        )

        assert config.name == "test"
        assert config.description == "Test agent"
        assert config.tools == ["tool1"]
        assert config.system_prompt == ""
        assert config.agent_type == "router"  # Changed default
        assert config.model is None
        assert config.max_iterations == 10
        assert config.factory_params == {}

    def test_agent_config_with_factory_params(self):
        """Should accept factory_params."""
        config = AgentConfig(
            name="test",
            description="Test",
            tools=["tool1"],
            factory_params={"param1": "value1"},
        )

        assert config.factory_params == {"param1": "value1"}

    def test_agent_type_is_extensible(self):
        """Should accept any string for agent_type."""
        config = AgentConfig(
            name="test",
            description="Test",
            tools=[],
            agent_type="custom_type",
        )

        assert config.agent_type == "custom_type"


class TestAgentServiceInit:
    """Test AgentService initialization."""

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    def test_init_with_tool_service(self, mock_import, mock_load):
        """Should initialize with tool service and config."""
        mock_tool_service = MagicMock()
        config = {"ollama_url": "http://localhost:11434"}

        service = AgentService(tool_service=mock_tool_service, config=config)

        assert service._tool_service == mock_tool_service
        assert service._config == config
        assert service._factory_registry is not None
        mock_import.assert_called_once()
        mock_load.assert_called_once()

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    def test_init_imports_factories(self, mock_import, mock_load):
        """Should import factories on init."""
        mock_tool_service = MagicMock()
        _ = AgentService(tool_service=mock_tool_service, config={})

        mock_import.assert_called_once()


class TestLoadBuiltinAgents:
    """Test AgentService._load_builtin_agents()."""

    @patch.object(AgentService, "_import_factories")
    def test_load_builtin_agents_registers_browse(self, mock_import):
        """Should register browse agent with router type."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        assert "browse" in service._agent_configs
        browse_config = service._agent_configs["browse"]
        assert browse_config.name == "browse"
        assert browse_config.tools == [
            "search_web",
            "fetch_pages_batch",
            "search_focused",
        ]
        assert browse_config.agent_type == "router"
        assert browse_config.model is None

    @patch.object(AgentService, "_import_factories")
    def test_load_builtin_agents_registers_research_alias(self, mock_import):
        """Should register research as alias for browse."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        assert "research" in service._agent_configs
        research_config = service._agent_configs["research"]
        assert research_config.name == "research"
        assert research_config.agent_type == "router"

    @patch.object(AgentService, "_import_factories")
    def test_load_builtin_agents_uses_config_values(self, mock_import):
        """Should use config values for max_iterations and min_pages."""
        mock_tool_service = MagicMock()
        config = {"agent": {"max_iterations": 20, "min_pages_required": 7}}
        service = AgentService(tool_service=mock_tool_service, config=config)

        browse_config = service._agent_configs["browse"]
        assert browse_config.max_iterations == 20
        assert browse_config.factory_params["min_pages_required"] == 7

    @patch.object(AgentService, "_import_factories")
    def test_load_builtin_agents_uses_defaults_when_no_config(self, mock_import):
        """Should use default values when config section missing."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        browse_config = service._agent_configs["browse"]
        assert browse_config.max_iterations == 10
        assert browse_config.factory_params["min_pages_required"] == 3

    @patch.object(AgentService, "_import_factories")
    def test_list_agents_includes_builtin_agents(self, mock_import):
        """Should list browse and research agents after initialization."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        agents = service.list_agents()
        agent_names = [agent["name"] for agent in agents]

        assert "browse" in agent_names
        assert "research" in agent_names
        assert len(agents) == 2

    @patch.object(AgentService, "_import_factories")
    def test_list_agents_includes_agent_type(self, mock_import):
        """Should include agent_type in list_agents output."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        agents = service.list_agents()
        browse_agent = next(a for a in agents if a["name"] == "browse")

        assert "agent_type" in browse_agent
        assert browse_agent["agent_type"] == "router"


class TestAgentServiceRegisterAgent:
    """Test AgentService.register_agent()."""

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    def test_register_agent_adds_config(self, mock_import, mock_load):
        """Should add agent config to registry."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="custom",
            description="Custom agent",
            tools=["tool1"],
        )

        service.register_agent(config)

        assert "custom" in service._agent_configs
        assert service._agent_configs["custom"] == config


@pytest.mark.asyncio
class TestAgentServiceRun:
    """Test AgentService.run() with factory pattern."""

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    async def test_run_unknown_agent_returns_error(self, mock_import, mock_load):
        """Should return error for unknown agent."""
        mock_tool_service = MagicMock()
        service = AgentService(tool_service=mock_tool_service, config={})

        result = await service.run(
            agent_name="nonexistent",
            goal="test",
            callbacks=AgentCallbacks(),
            session_params={},
        )

        assert result.error == "Unknown agent: nonexistent"
        assert result.final_answer == ""

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    async def test_run_missing_tools_returns_error(self, mock_import, mock_load):
        """Should return error when tools are missing."""
        mock_tool_service = MagicMock()
        mock_tool_service.get_tools_by_names.return_value = []
        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="test",
            description="Test",
            tools=["tool1", "tool2"],
        )
        service.register_agent(config)

        result = await service.run(
            agent_name="test",
            goal="test query",
            callbacks=AgentCallbacks(),
            session_params={},
        )

        assert "Missing tools" in result.error
        assert "tool1" in result.error or "tool2" in result.error

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    async def test_run_creates_agent_via_factory(self, mock_import, mock_load):
        """Should create agent via factory registry."""
        from llama_index.core.tools import FunctionTool

        mock_tool_service = MagicMock()
        tool = FunctionTool.from_defaults(fn=lambda: "test", name="test_tool")
        mock_tool_service.get_tools_by_names.return_value = [tool]

        service = AgentService(tool_service=mock_tool_service, config={})

        # Register a config
        config = AgentConfig(
            name="test",
            description="Test",
            tools=["test_tool"],
            agent_type="mock_type",
        )
        service.register_agent(config)

        # Mock the factory
        mock_agent = MockAgent("test")

        def mock_factory(config, tools, llm, params):
            return mock_agent

        service._factory_registry.register("mock_type", mock_factory)

        # Run
        result = await service.run(
            agent_name="test",
            goal="test query",
            callbacks=AgentCallbacks(),
            session_params={},
        )

        assert mock_agent.run_called
        assert mock_agent.run_query == "test query"
        assert result.final_answer == "Answer from test"

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    async def test_run_function_agent_uses_function_agent_model(
        self, mock_import, mock_load
    ):
        """Function agents should use function_agent_model when config.model is None."""
        from llama_index.core.tools import FunctionTool

        mock_tool_service = MagicMock()
        tool = FunctionTool.from_defaults(fn=lambda: "test", name="test_tool")
        mock_tool_service.get_tools_by_names.return_value = [tool]

        service = AgentService(
            tool_service=mock_tool_service,
            config={"agent": {"function_agent_model": "qwen2.5:7b"}},
        )

        config = AgentConfig(
            name="test_func",
            description="Test function agent",
            tools=["test_tool"],
            agent_type="function",
            model=None,  # No explicit model
        )
        service.register_agent(config)

        created_llm_model = None

        def capturing_factory(config, tools, llm, params):
            nonlocal created_llm_model
            created_llm_model = llm.model
            return MockAgent("test_func")

        # Overwrite directly since "function" may already be registered globally
        service._factory_registry._factories["function"] = capturing_factory

        await service.run(
            agent_name="test_func",
            goal="test",
            callbacks=AgentCallbacks(),
            session_params={},
        )

        assert created_llm_model == "qwen2.5:7b"

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    async def test_run_function_agent_with_none_session_param(
        self, mock_import, mock_load
    ):
        """Should fall back to config when session param is explicitly None.

        This mirrors the real-world case where yaml_command.py passes
        {"function_agent_model": None} from params.get().
        """
        from llama_index.core.tools import FunctionTool

        mock_tool_service = MagicMock()
        tool = FunctionTool.from_defaults(fn=lambda: "test", name="test_tool")
        mock_tool_service.get_tools_by_names.return_value = [tool]

        service = AgentService(
            tool_service=mock_tool_service,
            config={"agent": {"function_agent_model": "qwen2.5:7b"}},
        )

        config = AgentConfig(
            name="test_func_none",
            description="Test function agent",
            tools=["test_tool"],
            agent_type="function",
            model=None,
        )
        service.register_agent(config)

        created_llm_model = None

        def capturing_factory(config, tools, llm, params):
            nonlocal created_llm_model
            created_llm_model = llm.model
            return MockAgent("test_func_none")

        service._factory_registry._factories["function"] = capturing_factory

        await service.run(
            agent_name="test_func_none",
            goal="test",
            callbacks=AgentCallbacks(),
            # Key present but None — exactly what yaml_command.py does
            session_params={"function_agent_model": None},
        )

        assert created_llm_model == "qwen2.5:7b"

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    async def test_run_function_agent_session_override(self, mock_import, mock_load):
        """Session params should override global function_agent_model."""
        from llama_index.core.tools import FunctionTool

        mock_tool_service = MagicMock()
        tool = FunctionTool.from_defaults(fn=lambda: "test", name="test_tool")
        mock_tool_service.get_tools_by_names.return_value = [tool]

        service = AgentService(
            tool_service=mock_tool_service,
            config={"agent": {"function_agent_model": "qwen2.5:7b"}},
        )

        config = AgentConfig(
            name="test_func2",
            description="Test function agent",
            tools=["test_tool"],
            agent_type="function",
            model=None,
        )
        service.register_agent(config)

        created_llm_model = None

        def capturing_factory(config, tools, llm, params):
            nonlocal created_llm_model
            created_llm_model = llm.model
            return MockAgent("test_func2")

        # Overwrite directly since "function" may already be registered
        service._factory_registry._factories["function"] = capturing_factory

        await service.run(
            agent_name="test_func2",
            goal="test",
            callbacks=AgentCallbacks(),
            session_params={"function_agent_model": "mistral:7b"},
        )

        assert created_llm_model == "mistral:7b"

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    async def test_run_passes_factory_params(self, mock_import, mock_load):
        """Should pass factory_params to factory."""
        from llama_index.core.tools import FunctionTool

        mock_tool_service = MagicMock()
        tool = FunctionTool.from_defaults(fn=lambda: "test", name="test_tool")
        mock_tool_service.get_tools_by_names.return_value = [tool]

        service = AgentService(
            tool_service=mock_tool_service,
            config={"agent": {"router_model": "llama3.2:3b", "min_pages_required": 5}},
        )

        config = AgentConfig(
            name="test",
            description="Test",
            tools=["test_tool"],
            agent_type="mock_type_params",  # Unique type name
            factory_params={"custom_param": "custom_value"},
        )
        service.register_agent(config)

        received_params = {}

        def capturing_factory(config, tools, llm, params):
            received_params.update(params)
            return MockAgent("test")

        service._factory_registry.register("mock_type_params", capturing_factory)

        await service.run(
            agent_name="test",
            goal="test",
            callbacks=AgentCallbacks(),
            session_params={"context_window": 8192},
        )

        # Check params were passed correctly
        assert received_params["router_model"] == "llama3.2:3b"
        assert received_params["function_agent_model"] is not None
        assert received_params["min_pages_required"] == 5
        assert received_params["custom_param"] == "custom_value"
        assert received_params["context_window"] == 8192

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    async def test_run_handles_exceptions(self, mock_import, mock_load):
        """Should handle exceptions and return error."""
        from llama_index.core.tools import FunctionTool

        mock_tool_service = MagicMock()
        tool = FunctionTool.from_defaults(fn=lambda: "test", name="test_tool")
        mock_tool_service.get_tools_by_names.return_value = [tool]

        service = AgentService(tool_service=mock_tool_service, config={})

        config = AgentConfig(
            name="test",
            description="Test",
            tools=["test_tool"],
            agent_type="mock_type_exception",  # Unique type name
        )
        service.register_agent(config)

        def failing_factory(config, tools, llm, params):
            raise ValueError("Factory failed")

        service._factory_registry.register("mock_type_exception", failing_factory)

        result = await service.run(
            agent_name="test",
            goal="test",
            callbacks=AgentCallbacks(),
            session_params={},
        )

        assert result.error == "Factory failed"
        assert result.final_answer == ""


class TestAgentServiceLLMCreation:
    """Test LLM creation methods."""

    def test_create_llm_static(self):
        """Should create Ollama LLM with correct params."""
        llm = AgentService._create_llm_static(
            "llama3.1:8b", context_window=8192, ollama_url="http://test:11434"
        )

        assert llm.model == "llama3.1:8b"
        assert llm.base_url == "http://test:11434"
        assert llm.context_window == 8192

    def test_create_llm_static_uses_defaults(self):
        """Should use default values when not provided."""
        llm = AgentService._create_llm_static("llama3.1:8b")

        assert llm.base_url == "http://localhost:11434"
        assert llm.context_window == 16384

    @patch.object(AgentService, "_load_builtin_agents")
    @patch.object(AgentService, "_import_factories")
    def test_create_llm_instance_method(self, mock_import, mock_load):
        """Should create LLM using instance method."""
        mock_tool_service = MagicMock()
        config = {"ollama_url": "http://custom:11434"}
        service = AgentService(tool_service=mock_tool_service, config=config)

        llm = service._create_llm("llama3.1:8b", context_window=8192)

        assert llm.model == "llama3.1:8b"
        assert llm.base_url == "http://custom:11434"
        assert llm.context_window == 8192
