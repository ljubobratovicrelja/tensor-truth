"""Core constants for TensorTruth.

This module defines shared constants used across the application to ensure
consistency and avoid hardcoded values in multiple locations.
"""

# Default model configurations
DEFAULT_RAG_MODEL = "deepseek-r1:14b"
"""Default model for RAG engine."""

DEFAULT_AGENT_REASONING_MODEL = "llama3.1:8b"
"""Default model for autonomous agent reasoning.

This model is used for:
- Browse agent reasoning and action selection (/browse command)
- Web search agent operations
- Any autonomous agent that requires structured reasoning

NOTE: Agent requires 7b+ model for reliable reasoning.
3b models struggle with structured output and logical action selection.
"""
