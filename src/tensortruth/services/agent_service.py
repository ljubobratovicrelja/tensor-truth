"""Service for creating and executing agents via factory registry.

AgentService uses the agent factory registry to create and execute agents.
Supports both built-in agents (router, function) and custom user agents.
"""

import logging

# TYPE_CHECKING avoids circular import with ConfigService
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llama_index.llms.ollama import Ollama

from tensortruth.agents.config import AgentCallbacks, AgentConfig, AgentResult
from tensortruth.agents.factory import get_agent_factory_registry
from tensortruth.core.constants import DEFAULT_FUNCTION_AGENT_MODEL
from tensortruth.services.tool_service import ToolService

if TYPE_CHECKING:
    from tensortruth.services.config_service import ConfigService

logger = logging.getLogger(__name__)


class AgentService:
    """Creates and executes agents via factory registry.

    Uses the agent factory registry pattern to support built-in and custom agents.
    Agents are registered via AgentConfig and created through registered factories.
    """

    def __init__(
        self,
        tool_service: ToolService,
        config: Dict[str, Any],
        config_service: Optional["ConfigService"] = None,
    ):
        """Initialize AgentService.

        Args:
            tool_service: ToolService for getting FunctionTools.
            config: Application configuration dictionary (used at init time).
            config_service: Optional ConfigService for live config reads at runtime.
        """
        self._tool_service = tool_service
        self._config = config
        self._config_service = config_service
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._factory_registry = get_agent_factory_registry()

        # Import factories to trigger self-registration
        self._import_factories()

    def _import_factories(self) -> None:
        """Import agent factories to trigger self-registration."""
        try:
            from tensortruth.agents.function import (  # noqa: F401
                factory as function_factory,
            )

            logger.info("Agent factories imported and registered")
        except ImportError as e:
            logger.warning(f"Failed to import agent factories: {e}")

    def _get_live_config(self) -> Dict[str, Any]:
        """Get fresh config from ConfigService, falling back to init snapshot."""
        if self._config_service:
            return self._config_service.load().to_dict()
        return self._config

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
                "agent_type": cfg.agent_type,
            }
            for cfg in self._agent_configs.values()
        ]

    @staticmethod
    def _create_llm_static(
        model: str,
        context_window: Optional[int] = None,
        ollama_url: Optional[str] = None,
    ) -> Ollama:
        """Create Ollama LLM (static for factory usage).

        Args:
            model: Model name to use.
            context_window: Context window size (num_ctx).
            ollama_url: Ollama base URL.

        Returns:
            Configured Ollama LLM instance.
        """
        base_url = ollama_url or "http://localhost:11434"
        ctx_window = context_window or 16384

        return Ollama(
            model=model,
            base_url=base_url,
            temperature=0.2,
            context_window=ctx_window,
            additional_kwargs={"num_ctx": ctx_window},
            request_timeout=120.0,
        )

    def _create_llm(
        self,
        model: str,
        context_window: Optional[int] = None,
        ollama_url: Optional[str] = None,
    ) -> Ollama:
        """Create Ollama LLM instance (instance method wrapping static version).

        Args:
            model: Model name to use.
            context_window: Context window size (num_ctx).
            ollama_url: Ollama base URL.

        Returns:
            Configured Ollama LLM instance.
        """
        ollama_url = ollama_url or self._config.get(
            "ollama_url", "http://localhost:11434"
        )
        return self._create_llm_static(model, context_window, ollama_url)

    async def run(
        self,
        agent_name: str,
        goal: str,
        callbacks: AgentCallbacks,
        session_params: Dict[str, Any],
    ) -> AgentResult:
        """Execute agent via factory registry.

        Creates agent through registered factory and executes it.
        All agents (router, function) handle their own execution logic.

        Args:
            agent_name: Name of the registered agent to run.
            goal: The user's goal/query for the agent.
            callbacks: Callbacks for progress updates.
            session_params: Session parameters including model.

        Returns:
            AgentResult with final answer or error.
        """
        # Get config
        config = self._agent_configs.get(agent_name)
        if not config:
            return AgentResult(final_answer="", error=f"Unknown agent: {agent_name}")

        # Get tools
        tools = self._tool_service.get_tools_by_names(config.tools)
        loaded_names = {t.metadata.name for t in tools}
        missing = set(config.tools) - loaded_names
        if missing:
            return AgentResult(
                final_answer="", error=f"Missing tools: {', '.join(missing)}"
            )

        # Read live config (not stale startup snapshot)
        live_config = self._get_live_config()
        agent_cfg = live_config.get("agent", {})

        # Determine model (use `or` to skip None values from session params)
        if config.model:
            model = config.model
        elif config.agent_type == "function":
            model = (
                session_params.get("function_agent_model")
                or agent_cfg.get("function_agent_model")
                or DEFAULT_FUNCTION_AGENT_MODEL
            )
        else:
            model = session_params.get("model") or DEFAULT_FUNCTION_AGENT_MODEL
        context_window = session_params.get("context_window", 16384)
        ollama_url = session_params.get("ollama_url", "http://localhost:11434")

        # Create LLM
        llm = self._create_llm(model, context_window, ollama_url)

        # Build factory params: live config defaults, then session overrides
        factory_params = {
            "router_model": agent_cfg.get("router_model", "llama3.2:3b"),
            "function_agent_model": agent_cfg.get(
                "function_agent_model", DEFAULT_FUNCTION_AGENT_MODEL
            ),
            "min_pages_required": agent_cfg.get("min_pages_required", 3),
            **session_params,
            **config.factory_params,
        }

        try:
            # Progress
            if callbacks.on_progress:
                callbacks.on_progress(f"Starting {agent_name} agent...")

            logger.info(
                f"Creating agent: type={config.agent_type}, model={model}, "
                f"context_window={context_window}"
            )

            # Create agent via factory
            agent = self._factory_registry.create(
                agent_type=config.agent_type,
                config=config,
                tools=tools,
                llm=llm,
                session_params=factory_params,
            )

            # Execute
            result = await agent.run(query=goal, callbacks=callbacks)
            return result

        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            return AgentResult(final_answer="", error=str(e))
