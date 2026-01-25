"""FunctionAgent support for TensorTruth."""

from .factory import create_function_agent
from .wrapper import FunctionAgentWrapper

__all__ = ["FunctionAgentWrapper", "create_function_agent"]
