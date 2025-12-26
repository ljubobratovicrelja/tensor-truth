"""Configuration schema and default values for Tensor-Truth."""

from dataclasses import asdict, dataclass


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
    # Thinking model controls
    default_thinking_level: str = "low"  # "low", "medium", "high", or "disabled"
    default_thinking_top_k: int = 20
    default_thinking_stop_sequences: str = "</think>,\\n\\nFinal Answer:,\\n\\nAnswer:"


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""

    default_device: str = "cpu"  # Will be auto-detected on first run


@dataclass
class TensorTruthConfig:
    """Main configuration for Tensor-Truth application."""

    ollama: OllamaConfig
    ui: UIConfig
    rag: RAGConfig

    def to_dict(self) -> dict:
        """Convert config to dictionary for YAML serialization."""
        return {
            "ollama": asdict(self.ollama),
            "ui": asdict(self.ui),
            "rag": asdict(self.rag),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TensorTruthConfig":
        """Create config from dictionary (loaded from YAML)."""
        ollama_data = data.get("ollama", {})
        ui_data = data.get("ui", {})
        rag_data = data.get("rag", {})

        return cls(
            ollama=OllamaConfig(**ollama_data),
            ui=UIConfig(**ui_data),
            rag=RAGConfig(**rag_data),
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
