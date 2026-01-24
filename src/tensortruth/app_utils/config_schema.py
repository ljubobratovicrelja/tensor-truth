"""Configuration schema and default values for Tensor-Truth."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from tensortruth.core.constants import (
    DEFAULT_AGENT_REASONING_MODEL,
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

    default_temperature: float = 0.7
    default_context_window: int = 8192
    default_max_tokens: int = 4096
    default_top_n: int = 5
    default_confidence_threshold: float = 0.35
    default_confidence_cutoff_hard: float = 0.05


@dataclass
class EmbeddingModelConfig:
    """Per-embedding-model configuration for HuggingFace models.

    These settings optimize memory usage and performance for specific models.
    See: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B for Qwen3 recommendations.
    """

    # Batch sizes for embedding (smaller = less VRAM, slower)
    batch_size_cuda: int = 128
    batch_size_cpu: int = 16

    # PyTorch dtype: null (default), "float16", "bfloat16", "float32"
    torch_dtype: Optional[str] = None

    # Tokenizer padding side: null (default), "left", "right"
    # Qwen3 models recommend "left" padding
    padding_side: Optional[str] = None

    # Enable Flash Attention 2 if available (requires flash-attn package)
    flash_attention: bool = False

    # Trust remote code from HuggingFace (required for some models)
    trust_remote_code: bool = True


# Default embedding model configurations
# These are written to config.yaml on first run and can be customized
DEFAULT_EMBEDDING_MODEL_CONFIGS: Dict[str, Dict] = {
    # BGE-M3: High-quality multilingual embeddings, works well with defaults
    "BAAI/bge-m3": {
        "batch_size_cuda": 128,
        "batch_size_cpu": 16,
        "torch_dtype": None,
        "padding_side": None,
        "flash_attention": False,
        "trust_remote_code": True,
    },
    # Qwen3 Embedding models: Need special handling due to memory spikes
    # See: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/discussions/38
    "Qwen/Qwen3-Embedding-0.6B": {
        "batch_size_cuda": 8,
        "batch_size_cpu": 4,
        "torch_dtype": "float16",
        "padding_side": "left",
        "flash_attention": True,
        "trust_remote_code": True,
    },
    "Qwen/Qwen3-Embedding-4B": {
        "batch_size_cuda": 4,
        "batch_size_cpu": 2,
        "torch_dtype": "float16",
        "padding_side": "left",
        "flash_attention": True,
        "trust_remote_code": True,
    },
    "Qwen/Qwen3-Embedding-8B": {
        "batch_size_cuda": 2,
        "batch_size_cpu": 1,
        "torch_dtype": "float16",
        "padding_side": "left",
        "flash_attention": True,
        "trust_remote_code": True,
    },
}

# Default config for unknown models
DEFAULT_EMBEDDING_MODEL_CONFIG = EmbeddingModelConfig()


# Default reranker models available out of the box
DEFAULT_RERANKER_MODELS = [
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""

    default_device: str = "cpu"  # Will be auto-detected on first run
    default_balance_strategy: str = "top_k_per_index"  # Multi-index balancing
    default_embedding_model: str = "BAAI/bge-m3"  # HuggingFace embedding model
    default_reranker: str = "BAAI/bge-reranker-v2-m3"  # Default reranker model

    # Number of recent conversation turns to include in prompt (limits context size)
    # A turn = one user query + one assistant response (2 messages)
    max_history_turns: int = 3

    # Token limit for chat memory buffer (safety backstop)
    # With 8k context, 5 retrievals (~1.5k), system prompt (~1k), this leaves ~5k for history
    memory_token_limit: int = 4000

    # Per-model configurations (model_name -> config dict)
    # On first run, this is populated from DEFAULT_EMBEDDING_MODEL_CONFIGS
    embedding_model_configs: Dict[str, Dict] = field(default_factory=dict)

    # Available reranker models (user can add custom HuggingFace rerankers)
    # On first run, this is populated from DEFAULT_RERANKER_MODELS
    reranker_models: list = field(default_factory=list)

    def get_embedding_model_config(self, model_name: str) -> EmbeddingModelConfig:
        """Get configuration for a specific embedding model.

        Looks up model-specific config, falling back to defaults if not found.

        Args:
            model_name: HuggingFace model path (e.g., "BAAI/bge-m3")

        Returns:
            EmbeddingModelConfig with settings for this model
        """
        # Check user config first
        if model_name in self.embedding_model_configs:
            return EmbeddingModelConfig(**self.embedding_model_configs[model_name])

        # Fall back to built-in defaults
        if model_name in DEFAULT_EMBEDDING_MODEL_CONFIGS:
            return EmbeddingModelConfig(**DEFAULT_EMBEDDING_MODEL_CONFIGS[model_name])

        # Unknown model - use generic defaults
        return DEFAULT_EMBEDDING_MODEL_CONFIG

    def get_reranker_models(self) -> list:
        """Get available reranker models with fallback to defaults.

        Returns:
            List of HuggingFace model paths for rerankers
        """
        if self.reranker_models:
            return self.reranker_models
        return list(DEFAULT_RERANKER_MODELS)


@dataclass
class ModelsConfig:
    """Default model configurations."""

    # Default model for RAG engine
    default_rag_model: str = DEFAULT_RAG_MODEL

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


# Default filler phrases for history cleaning (regex patterns, case-insensitive)
DEFAULT_FILLER_PHRASES = [
    r"(?i)^(great|good|excellent)\s+(question|point)[!.]*\s*",
    r"(?i)^i['']?d be happy to help[!.]*\s*",
    r"(?i)^let me (think|see|help)[^.]*[.!]*\s*",
    r"(?i)^(sure|certainly|absolutely)[!.,]*\s*",
    r"(?i)if you have any (more |other )?questions[^.]*[.!]*\s*$",
    r"(?i)feel free to ask[^.]*[.!]*\s*$",
    r"(?i)hope this helps[!.]*\s*$",
]


@dataclass
class WebSearchConfig:
    """Web search pipeline configuration."""

    # DDG search settings
    ddg_max_results: int = 10  # Max results to fetch from DuckDuckGo
    max_pages_to_fetch: int = 5  # Max pages to actually download and process

    # Reranking thresholds (0.0-1.0)
    rerank_title_threshold: float = 0.1  # Min score after title/snippet reranking
    rerank_content_threshold: float = 0.1  # Min score after content reranking

    # Context fitting (percentages of context window)
    max_source_context_pct: float = 0.15  # Max % of context per source
    input_context_pct: float = 0.6  # % of context window for input (rest for output)


@dataclass
class HistoryCleaningConfig:
    """Chat history cleaning configuration.

    Preprocesses chat history to reduce token usage without losing semantic meaning.
    All settings are enabled by default.
    """

    # Master switch for history cleaning
    enabled: bool = True

    # Remove emoji characters from messages
    remove_emojis: bool = True

    # Remove common filler phrases (e.g., "Great question!", "Hope this helps!")
    remove_filler_phrases: bool = True

    # Normalize multiple inline spaces to single space (preserves indentation)
    normalize_whitespace: bool = True

    # Collapse 3+ consecutive newlines to 2
    collapse_newlines: bool = True

    # Configurable filler phrases (regex patterns, case-insensitive)
    filler_phrases: List[str] = field(
        default_factory=lambda: list(DEFAULT_FILLER_PHRASES)
    )


@dataclass
class TensorTruthConfig:
    """Main configuration for Tensor-Truth application."""

    ollama: OllamaConfig
    ui: UIConfig
    rag: RAGConfig
    models: ModelsConfig
    agent: AgentConfig
    history_cleaning: HistoryCleaningConfig
    web_search: WebSearchConfig

    def to_dict(self) -> dict:
        """Convert config to dictionary for YAML serialization."""
        return {
            "ollama": asdict(self.ollama),
            "ui": asdict(self.ui),
            "rag": asdict(self.rag),
            "models": asdict(self.models),
            "agent": asdict(self.agent),
            "history_cleaning": asdict(self.history_cleaning),
            "web_search": asdict(self.web_search),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TensorTruthConfig":
        """Create config from dictionary (loaded from YAML)."""
        ollama_data = data.get("ollama", {})
        ui_data = data.get("ui", {})
        rag_data = data.get("rag", {})
        models_data = data.get("models", {})
        agent_data = data.get("agent", {})
        history_cleaning_data = data.get("history_cleaning", {})
        web_search_data = data.get("web_search", {})

        return cls(
            ollama=OllamaConfig(**ollama_data),
            ui=UIConfig(**ui_data),
            rag=RAGConfig(**rag_data),
            models=ModelsConfig(**models_data),
            agent=AgentConfig(**agent_data),
            history_cleaning=HistoryCleaningConfig(**history_cleaning_data),
            web_search=WebSearchConfig(**web_search_data),
        )

    @classmethod
    def create_default(cls) -> "TensorTruthConfig":
        """Create default configuration with smart device detection."""
        # Detect best default device for RAG
        default_device = cls._detect_default_device()

        return cls(
            ollama=OllamaConfig(),
            ui=UIConfig(),
            rag=RAGConfig(
                default_device=default_device,
                # Populate with default embedding model configs
                embedding_model_configs=dict(DEFAULT_EMBEDDING_MODEL_CONFIGS),
                # Populate with default reranker models
                reranker_models=list(DEFAULT_RERANKER_MODELS),
            ),
            models=ModelsConfig(),
            agent=AgentConfig(),
            history_cleaning=HistoryCleaningConfig(),
            web_search=WebSearchConfig(),
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
