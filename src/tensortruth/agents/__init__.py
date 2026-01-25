"""Agent framework for TensorTruth.

Provides a plugin-style agent registry system modeled after ToolService.
Agents implement the Agent interface and are registered via factories.
"""

from .base import Agent
from .config import AgentCallbacks, AgentConfig, AgentResult
from .factory import (
    AgentFactoryRegistry,
    get_agent_factory_registry,
    register_agent_factory,
)

__all__ = [
    "Agent",
    "AgentCallbacks",
    "AgentConfig",
    "AgentResult",
    "AgentFactoryRegistry",
    "get_agent_factory_registry",
    "register_agent_factory",
]
