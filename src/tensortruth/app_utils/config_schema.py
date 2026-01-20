"""Configuration schema and default values for Tensor-Truth."""

from dataclasses import asdict, dataclass

from tensortruth.core.constants import (
    DEFAULT_AGENT_REASONING_MODEL,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_RAG_MODEL,
)


@dataclass
class OllamaConfig:
    """Ollama service configuration."""

    base_url: str = "http://localhost:11434"
    timeout: int = 300


@dataclass
class UIConfig:
    """User interface preferences."""

    default_temperature: float = 0.1
    default_context_window: int = 16384
    default_max_tokens: int = 4096
    default_reranker: str = "BAAI/bge-reranker-v2-m3"
    default_top_n: int = 5
    default_confidence_threshold: float = 0.4
    default_confidence_cutoff_hard: float = 0.1


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""

    default_device: str = "cpu"  # Will be auto-detected on first run


@dataclass
class ModelsConfig:
    """Default model configurations."""

    # Default model for RAG engine
    default_rag_model: str = DEFAULT_RAG_MODEL

    # Default fallback model when no models are available
    default_fallback_model: str = DEFAULT_FALLBACK_MODEL

    # Default agent reasoning model (used in browse commands and autonomous agents)
    default_agent_reasoning_model: str = DEFAULT_AGENT_REASONING_MODEL


@dataclass
class AgentConfig:
    """Autonomous agent configuration."""

    # Maximum iterations for agent execution
    max_iterations: int = 10

    # Minimum pages agent must fetch during web research
    # Higher values = more thorough research, but slower
    min_pages_required: int = 3

    # Model to use for agent reasoning (fast model for decisions)
    # Falls back to main chat model if not specified
    reasoning_model: str = "llama3.1:8b"

    # Natural language agent routing
    # When enabled, messages with trigger words are classified to route to agents
    enable_natural_language_agents: bool = True

    # Model for intent classification (should be fast, small model)
    intent_classifier_model: str = "llama3.2:3b"

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )
        if self.min_pages_required < 1:
            raise ValueError(
                f"min_pages_required must be at least 1, got {self.min_pages_required}"
            )
        if not self.reasoning_model or not isinstance(self.reasoning_model, str):
            raise ValueError(
                f"reasoning_model must be a non-empty string, got {self.reasoning_model!r}"
            )


@dataclass
class TensorTruthConfig:
    """Main configuration for Tensor-Truth application."""

    ollama: OllamaConfig
    ui: UIConfig
    rag: RAGConfig
    models: ModelsConfig
    agent: AgentConfig

    def to_dict(self) -> dict:
        """Convert config to dictionary for YAML serialization."""
        return {
            "ollama": asdict(self.ollama),
            "ui": asdict(self.ui),
            "rag": asdict(self.rag),
            "models": asdict(self.models),
            "agent": asdict(self.agent),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TensorTruthConfig":
        """Create config from dictionary (loaded from YAML)."""
        ollama_data = data.get("ollama", {})
        ui_data = data.get("ui", {})
        rag_data = data.get("rag", {})
        models_data = data.get("models", {})
        agent_data = data.get("agent", {})

        return cls(
            ollama=OllamaConfig(**ollama_data),
            ui=UIConfig(**ui_data),
            rag=RAGConfig(**rag_data),
            models=ModelsConfig(**models_data),
            agent=AgentConfig(**agent_data),
        )

    @classmethod
    def create_default(cls) -> "TensorTruthConfig":
        """Create default configuration with smart device detection."""
        # Detect best default device for RAG
        default_device = cls._detect_default_device()

        return cls(
            ollama=OllamaConfig(),
            ui=UIConfig(),
            rag=RAGConfig(default_device=default_device),
            models=ModelsConfig(),
            agent=AgentConfig(),
        )

    @staticmethod
    def _detect_default_device() -> str:
        """Detect the best default device for this machine."""
        try:
            import torch

            # Check MPS (Apple Silicon) - prefer this on Mac
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"

            # Check CUDA (NVIDIA GPU) - prefer this on Windows/Linux
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass

        # Fallback to CPU
        return "cpu"
