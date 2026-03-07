"""Core constants for TensorTruth.

This module defines shared constants used across the application to ensure
consistency and avoid hardcoded values in multiple locations.
"""

# Default Ollama configuration
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
"""Default base URL for Ollama API server."""

# Default model configurations
DEFAULT_MODEL = ""
"""Default LLM model (empty = use first available Ollama model)."""
