"""Router-based agents for TensorTruth.

Router agents use a two-phase architecture:
1. Small LLM routes workflow decisions
2. Larger LLM synthesizes final answer

This package provides:
- RouterAgent: Abstract base class for router-based agents
- RouterState: Abstract base class for agent state
"""

from .base import RouterAgent
from .state import RouterState

__all__ = ["RouterAgent", "RouterState"]
