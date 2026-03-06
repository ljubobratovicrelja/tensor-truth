"""Configuration schema and default values for Tensor-Truth."""

from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional

from tensortruth.core.constants import (
    DEFAULT_FUNCTION_AGENT_MODEL,
    DEFAULT_MODEL,
    DEFAULT_ROUTER_MODEL,
)


@dataclass
class OllamaConfig:
    """Ollama service configuration."""

    base_url: str = "http://localhost:11434"
    timeout: int = 300


@dataclass
class LLMConfig:
    """LLM generation defaults."""

    default_model: str = DEFAULT_MODEL
    default_temperature: float = 0.7
    default_context_window: int = 8192
    default_max_tokens: int = 4096


@dataclass
class EmbeddingModelConfig:
    """Per-embedding-model configuration for HuggingFace models.

    These settings optimize memory usage and performance for specific models.
    """

    # Batch sizes for embedding (smaller = less VRAM, slower)
    batch_size_cuda: int = 128
    batch_size_cpu: int = 16

    # PyTorch dtype: null (default), "float16", "bfloat16", "float32"
    torch_dtype: Optional[str] = None

    # Tokenizer padding side: null (default), "left", "right"
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
}

# Default config for unknown models
DEFAULT_EMBEDDING_MODEL_CONFIG = EmbeddingModelConfig()


# Default reranker models available out of the box
DEFAULT_RERANKER_MODELS = [
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
    default_top_n: int = 5
    default_confidence_threshold: float = 0.35
    default_confidence_cutoff_hard: float = 0.05

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
class ConversationConfig:
    """Conversation history settings."""

    # Number of recent conversation turns to include in prompt (limits context size)
    # A turn = one user query + one assistant response (2 messages)
    max_history_turns: int = 3

    # Token limit for chat memory buffer (safety backstop)
    # With 8k context, 5 retrievals (~1.5k), system prompt (~1k), this leaves ~5k for history
    memory_token_limit: int = 4000


@dataclass
class AgentConfig:
    """Autonomous agent configuration."""

    # Maximum iterations for agent execution
    max_iterations: int = 10

    # Minimum pages agent must fetch during web research
    # Higher values = more thorough research, but slower
    min_pages_required: int = 5

    # Router-based agent configuration
    router_model: str = DEFAULT_ROUTER_MODEL
    function_agent_model: str = DEFAULT_FUNCTION_AGENT_MODEL
    enable_search_reranking: bool = True

    # Natural language agent routing
    # When enabled, messages with trigger words are classified to route to agents
    enable_natural_language_agents: bool = True

    # Model for intent classification (should be fast, small model)
    intent_classifier_model: str = "llama3.2:3b"

    # Orchestrator (agentic chat) — when enabled, messages are routed through
    # the orchestrator agent which can call tools (RAG, web search, etc.).
    # Session-level setting overrides this global default.
    # Hard-disabled if the active model lacks tool-calling capability.
    orchestrator_enabled: bool = True

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
        if not self.router_model or not isinstance(self.router_model, str):
            raise ValueError(
                f"router_model must be a non-empty string, got {self.router_model!r}"
            )
        if not self.function_agent_model or not isinstance(
            self.function_agent_model, str
        ):
            raise ValueError(
                "function_agent_model must be a non-empty string, "
                f"got {self.function_agent_model!r}"
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
    llm: LLMConfig
    rag: RAGConfig
    conversation: ConversationConfig
    agent: AgentConfig
    history_cleaning: HistoryCleaningConfig
    web_search: WebSearchConfig

    def to_dict(self) -> dict:
        """Convert config to dictionary for YAML serialization."""
        return {
            "ollama": asdict(self.ollama),
            "llm": asdict(self.llm),
            "rag": asdict(self.rag),
            "conversation": asdict(self.conversation),
            "agent": asdict(self.agent),
            "history_cleaning": asdict(self.history_cleaning),
            "web_search": asdict(self.web_search),
        }

    @staticmethod
    def _migrate_config_data(data: dict) -> dict:
        """Migrate old config format to new.

        Handles the transition from ui/models groups to llm/rag/conversation.
        """
        ui = data.pop("ui", {})
        models = data.pop("models", {})

        # ui → llm (temperature, context_window, max_tokens)
        llm = data.get("llm", {})
        for key in [
            "default_temperature",
            "default_context_window",
            "default_max_tokens",
        ]:
            if key in ui and key not in llm:
                llm[key] = ui[key]

        # models.default_rag_model → llm.default_model
        if "default_rag_model" in models and "default_model" not in llm:
            llm["default_model"] = models["default_rag_model"]
        data["llm"] = llm

        # ui → rag (top_n, confidence thresholds)
        rag = data.get("rag", {})
        for key in [
            "default_top_n",
            "default_confidence_threshold",
            "default_confidence_cutoff_hard",
        ]:
            if key in ui and key not in rag:
                rag[key] = ui[key]

        # rag → conversation (max_history_turns, memory_token_limit)
        conversation = data.get("conversation", {})
        for key in ["max_history_turns", "memory_token_limit"]:
            if key in rag and key not in conversation:
                conversation[key] = rag.pop(key)
        data["rag"] = rag
        data["conversation"] = conversation

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TensorTruthConfig":
        """Create config from dictionary (loaded from YAML)."""
        data = cls._migrate_config_data(data)

        ollama_data = data.get("ollama", {})
        llm_data = data.get("llm", {})
        rag_data = data.get("rag", {})
        conversation_data = data.get("conversation", {})
        agent_data = data.get("agent", {})
        history_cleaning_data = data.get("history_cleaning", {})
        web_search_data = data.get("web_search", {})

        def _filter(cls_, data_):
            """Filter dict to only include known dataclass fields."""
            known = {f.name for f in fields(cls_)}
            return {k: v for k, v in data_.items() if k in known}

        return cls(
            ollama=OllamaConfig(**_filter(OllamaConfig, ollama_data)),
            llm=LLMConfig(**_filter(LLMConfig, llm_data)),
            rag=RAGConfig(**_filter(RAGConfig, rag_data)),
            conversation=ConversationConfig(
                **_filter(ConversationConfig, conversation_data)
            ),
            agent=AgentConfig(**_filter(AgentConfig, agent_data)),
            history_cleaning=HistoryCleaningConfig(
                **_filter(HistoryCleaningConfig, history_cleaning_data)
            ),
            web_search=WebSearchConfig(**_filter(WebSearchConfig, web_search_data)),
        )

    @classmethod
    def create_default(cls) -> "TensorTruthConfig":
        """Create default configuration with smart device detection."""
        # Detect best default device for RAG
        default_device = cls._detect_default_device()

        return cls(
            ollama=OllamaConfig(),
            llm=LLMConfig(),
            rag=RAGConfig(
                default_device=default_device,
                # Populate with default embedding model configs
                embedding_model_configs=dict(DEFAULT_EMBEDDING_MODEL_CONFIGS),
                # Populate with default reranker models
                reranker_models=list(DEFAULT_RERANKER_MODELS),
            ),
            conversation=ConversationConfig(),
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
