"""Service for creating and executing LlamaIndex agents.

AgentService creates and executes LlamaIndex agents from configuration.
Does NOT subclass or wrap agents - creates FunctionAgent/ReActAgent directly.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow.function_agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import BaseTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.config import AgentConfig, AgentResult
from tensortruth.services.tool_service import ToolService

logger = logging.getLogger(__name__)


@dataclass
class AgentCallbacks:
    """Streaming callbacks for WebSocket progress.

    Attributes:
        on_progress: Called with status messages during execution.
        on_tool_call: Called when a tool is invoked with tool name and params.
        on_token: Called with streaming tokens during generation.
    """

    on_progress: Optional[Callable[[str], None]] = None
    on_tool_call: Optional[Callable[[str, Dict], None]] = None
    on_token: Optional[Callable[[str], None]] = None


class AgentService:
    """Creates and executes LlamaIndex agents from configuration.

    Does NOT subclass or wrap agents - creates FunctionAgent/ReActAgent directly.
    Agents are registered via AgentConfig which defines tools, prompts, and behavior.
    """

    def __init__(self, tool_service: ToolService, config: Dict[str, Any]):
        """Initialize AgentService.

        Args:
            tool_service: ToolService for getting FunctionTools.
            config: Application configuration dictionary.
        """
        self._tool_service = tool_service
        self._config = config
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._load_builtin_agents()

    def _load_builtin_agents(self) -> None:
        """Register built-in agent configurations.

        NOTE: In this phase, no built-in agents are registered.
        Concrete agents (browse, research, etc.) come in a subsequent phase.
        """
        pass  # Framework only - agents registered in later phase

    def register_agent(self, config: AgentConfig) -> None:
        """Register an agent configuration.

        Called by concrete agent implementations or user-defined agent loaders.

        Args:
            config: AgentConfig defining the agent's behavior.
        """
        self._agent_configs[config.name] = config
        logger.info(f"Registered agent: {config.name}")

    def list_agents(self) -> List[Dict[str, Any]]:
        """List available agents for API.

        Returns:
            List of dictionaries with agent metadata.
        """
        return [
            {
                "name": cfg.name,
                "description": cfg.description,
                "tools": cfg.tools,
            }
            for cfg in self._agent_configs.values()
        ]

    def _create_llm(self, model: str) -> Ollama:
        """Create Ollama LLM instance.

        Args:
            model: Model name to use.

        Returns:
            Configured Ollama LLM instance.
        """
        return Ollama(
            model=model,
            base_url=self._config.get("ollama_url", "http://localhost:11434"),
            temperature=0.2,
            request_timeout=120.0,
        )

    def _create_agent(
        self,
        config: AgentConfig,
        tools: Sequence[Union[BaseTool, Callable[..., Any]]],
        llm: Ollama,
    ) -> Union[FunctionAgent, ReActAgent]:
        """Create a LlamaIndex agent from configuration.

        Args:
            config: Agent configuration.
            tools: List of tools for the agent.
            llm: Ollama LLM instance.

        Returns:
            FunctionAgent or ReActAgent instance.
        """
        tools_list = list(tools)
        if config.agent_type == "function":
            return FunctionAgent(
                tools=tools_list,
                llm=llm,
                system_prompt=config.system_prompt,
            )
        else:  # react
            return ReActAgent(
                tools=tools_list,
                llm=llm,
                system_prompt=config.system_prompt,
                verbose=True,
            )

    def _wrap_tools_for_callbacks(
        self, tools: List[FunctionTool], callbacks: AgentCallbacks
    ) -> List[FunctionTool]:
        """Wrap tools to emit progress callbacks.

        Creates new FunctionTool instances that call the callback before
        delegating to the original tool.

        Args:
            tools: Original tools to wrap.
            callbacks: Callbacks to emit on tool calls.

        Returns:
            List of wrapped tools (or original if no callback).
        """
        if not callbacks.on_tool_call:
            return tools

        wrapped = []
        for tool in tools:
            # Get the callable - prefer async_fn if available
            original_fn = tool.async_fn if tool.async_fn is not None else tool.fn

            # Create closure with explicit binding
            def make_tracked_call(
                t: FunctionTool, orig: Callable[..., Any]
            ) -> Callable[..., Any]:
                async def tracked_call(**kwargs: Any) -> Any:
                    if callbacks.on_tool_call:
                        tool_name = t.metadata.name or "unknown"
                        callbacks.on_tool_call(tool_name, kwargs)
                    return await orig(**kwargs)

                return tracked_call

            tracked_fn = make_tracked_call(tool, original_fn)

            wrapped_tool = FunctionTool.from_defaults(
                async_fn=tracked_fn,
                name=tool.metadata.name,
                description=tool.metadata.description,
            )
            wrapped.append(wrapped_tool)

        return wrapped

    async def run(
        self,
        agent_name: str,
        goal: str,
        callbacks: AgentCallbacks,
        session_params: Dict[str, Any],
    ) -> AgentResult:
        """Execute an agent by name.

        Args:
            agent_name: Name of the registered agent to run.
            goal: The user's goal/query for the agent.
            callbacks: Callbacks for progress updates.
            session_params: Session parameters including model.

        Returns:
            AgentResult with final answer or error.
        """
        config = self._agent_configs.get(agent_name)
        if not config:
            return AgentResult(final_answer="", error=f"Unknown agent: {agent_name}")

        # Get required tools
        tools = self._tool_service.get_tools_by_names(config.tools)
        loaded_names = {t.metadata.name for t in tools}
        missing = set(config.tools) - loaded_names
        if missing:
            return AgentResult(
                final_answer="", error=f"Missing tools: {', '.join(missing)}"
            )

        # Wrap tools for progress tracking
        if callbacks.on_tool_call:
            tools = self._wrap_tools_for_callbacks(tools, callbacks)

        # Determine model
        model = config.model or session_params.get("model", "llama3.1:8b")
        llm = self._create_llm(model)

        # Create LlamaIndex agent
        agent = self._create_agent(config, tools, llm)

        # Execute
        try:
            if callbacks.on_progress:
                callbacks.on_progress(f"Starting {agent_name} agent...")

            response = await agent.run(
                user_msg=goal,
            )

            return AgentResult(
                final_answer=str(response),
                iterations=config.max_iterations,  # Approximate
                tools_called=[],  # TODO: track from wrapper
            )
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return AgentResult(final_answer="", error=str(e))
