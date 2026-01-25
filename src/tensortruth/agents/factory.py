"""Factory registry for agent plugins."""

from typing import Any, Callable, Dict, Sequence

from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.base import Agent
from tensortruth.agents.config import AgentConfig

AgentFactory = Callable[
    [AgentConfig, Sequence[FunctionTool], Ollama, Dict[str, Any]], Agent
]


class AgentFactoryRegistry:
    """Registry of agent factories for plugin-style extensibility.

    Modeled after ToolService pattern: built-in agents + custom agents
    registered via factories.
    """

    def __init__(self):
        self._factories: Dict[str, AgentFactory] = {}

    def register(self, agent_type: str, factory: AgentFactory) -> None:
        """Register an agent factory.

        Args:
            agent_type: Type identifier (e.g., "router", "function")
            factory: Callable that creates Agent instances

        Raises:
            ValueError: If agent_type already registered
        """
        if agent_type in self._factories:
            raise ValueError(f"Agent type '{agent_type}' already registered")
        self._factories[agent_type] = factory

    def create(
        self,
        agent_type: str,
        config: AgentConfig,
        tools: Sequence[FunctionTool],
        llm: Ollama,
        session_params: Dict[str, Any],
    ) -> Agent:
        """Create agent via registered factory.

        Args:
            agent_type: Type of agent to create
            config: Agent configuration
            tools: Tools available to the agent
            llm: Ollama LLM instance
            session_params: Session-specific parameters

        Returns:
            Agent instance

        Raises:
            ValueError: If agent_type not registered
        """
        if agent_type not in self._factories:
            available = ", ".join(self._factories.keys())
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available: {available}"
            )
        factory = self._factories[agent_type]
        return factory(config, tools, llm, session_params)

    def list_types(self) -> list[str]:
        """List all registered agent types.

        Returns:
            List of agent type identifiers
        """
        return list(self._factories.keys())


# Global registry singleton
_agent_factory_registry = AgentFactoryRegistry()


def register_agent_factory(agent_type: str, factory: AgentFactory) -> None:
    """Register an agent factory globally.

    Args:
        agent_type: Type identifier for the agent
        factory: Factory function that creates agents
    """
    _agent_factory_registry.register(agent_type, factory)


def get_agent_factory_registry() -> AgentFactoryRegistry:
    """Get the global agent factory registry.

    Returns:
        The singleton AgentFactoryRegistry instance
    """
    return _agent_factory_registry
