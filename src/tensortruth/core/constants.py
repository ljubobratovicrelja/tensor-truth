"""Core constants for TensorTruth.

This module defines shared constants used across the application to ensure
consistency and avoid hardcoded values in multiple locations.
"""

# Default Ollama configuration
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
"""Default base URL for Ollama API server."""

# Default model configurations
DEFAULT_RAG_MODEL = "deepseek-r1:8b"
"""Default model for RAG engine."""

DEFAULT_AGENT_REASONING_MODEL = "llama3.2:3b"
"""Default model for autonomous agent reasoning.

llama3.2:3b is highly optimized for tool calling and consistently
generates multi-query searches. This model excels at following structured
tool-calling instructions compared to larger general-purpose models.

This model is used for:
- Browse agent reasoning and action selection (/browse command)
- Web search agent operations
- Any autonomous agent that requires structured reasoning
"""

DEFAULT_ROUTER_MODEL = "llama3.2:3b"
"""Default model for router-based agent routing decisions.

llama3.2:3b is ideal for fast, structured routing decisions in router-based
agents. This model is used specifically for action routing, while synthesis
uses the main session model.
"""

DEFAULT_FUNCTION_AGENT_MODEL = "llama3.1:8b"
"""Default model for function-type tool-calling agents.

llama3.1:8b provides reliable tool-calling capabilities for function agents
(e.g., doc_researcher, custom YAML agents with agent_type: function).
This model is used when the agent config does not specify a model explicitly.
"""
