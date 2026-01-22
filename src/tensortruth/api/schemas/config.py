"""Configuration-related schemas."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class OllamaConfigSchema(BaseModel):
    """Ollama service configuration."""

    base_url: str = "http://localhost:11434"
    timeout: int = 300


class UIConfigSchema(BaseModel):
    """User interface preferences."""

    default_temperature: float = 0.1
    default_context_window: int = 16384
    default_max_tokens: int = 4096
    default_top_n: int = 5
    default_confidence_threshold: float = 0.4
    default_confidence_cutoff_hard: float = 0.1


class RAGConfigSchema(BaseModel):
    """RAG pipeline configuration."""

    default_device: str = "cpu"
    default_balance_strategy: str = "top_k_per_index"
    default_embedding_model: str = "BAAI/bge-m3"
    default_reranker: str = "BAAI/bge-reranker-v2-m3"


class ModelsConfigSchema(BaseModel):
    """Default model configurations."""

    default_rag_model: str = "deepseek-r1:14b"
    default_fallback_model: str = "llama3.1:8b"
    default_agent_reasoning_model: str = "llama3.1:8b"


class AgentConfigSchema(BaseModel):
    """Autonomous agent configuration."""

    max_iterations: int = 10
    min_pages_required: int = 3
    reasoning_model: str = "llama3.1:8b"
    enable_natural_language_agents: bool = True
    intent_classifier_model: str = "llama3.2:3b"


class ConfigResponse(BaseModel):
    """Full configuration response."""

    ollama: OllamaConfigSchema
    ui: UIConfigSchema
    rag: RAGConfigSchema
    models: ModelsConfigSchema
    agent: AgentConfigSchema


class ConfigUpdateRequest(BaseModel):
    """Request body for updating configuration.

    Supports nested updates using prefixed keys:
    - ollama_*: Updates ollama config
    - ui_*: Updates UI config
    - rag_*: Updates RAG config
    - agent_*: Updates agent config
    - models_*: Updates models config
    """

    updates: Dict[str, Any] = Field(
        ...,
        description="Config updates with prefixed keys (e.g., ollama_base_url)",
    )
