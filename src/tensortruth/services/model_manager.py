"""Singleton model manager for embedder and reranker lifecycle.

This module provides centralized management of embedding and reranking models
with lazy loading and memory-efficient swapping between models.
"""

import gc
import logging
import threading
from typing import Any, Dict, Optional

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from tensortruth.app_utils.config_schema import (
    DEFAULT_EMBEDDING_MODEL_CONFIG,
    DEFAULT_EMBEDDING_MODEL_CONFIGS,
    EmbeddingModelConfig,
)

logger = logging.getLogger(__name__)

# Default models
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class ModelManager:
    """Singleton managing embedder and reranker with lazy loading and swapping.

    Only one embedder and one reranker are loaded in memory at a time.
    When a different model is requested, the old one is dropped and the
    new one loads on-demand.
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        """Ensure only one instance exists (thread-safe singleton)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the model manager (only once)."""
        if getattr(self, "_initialized", False):
            return

        self._embedder: Optional[HuggingFaceEmbedding] = None
        self._embedder_model_name: Optional[str] = None
        self._embedder_device: Optional[str] = None

        self._reranker: Optional[SentenceTransformerRerank] = None
        self._reranker_model_name: Optional[str] = None
        self._reranker_top_n: Optional[int] = None
        self._reranker_device: Optional[str] = None

        self._default_device = "cpu"
        self._model_lock = threading.Lock()
        self._initialized = True

        logger.debug("ModelManager singleton initialized")

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Get the singleton instance.

        Returns:
            The ModelManager singleton instance
        """
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for testing).

        This unloads all models and resets the singleton.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.unload_all()
                cls._instance._initialized = False
                cls._instance = None

    def set_default_device(self, device: str) -> None:
        """Set the default device for model loading.

        Args:
            device: Device to use ("cpu", "cuda", "mps")
        """
        self._default_device = device
        logger.debug(f"Default device set to: {device}")

    def get_embedder(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> HuggingFaceEmbedding:
        """Get embedder, loading or swapping if needed.

        Args:
            model_name: HuggingFace model path (default: BAAI/bge-m3)
            device: Device to load on (default: class default device)

        Returns:
            HuggingFaceEmbedding instance
        """
        model_name = model_name or DEFAULT_EMBEDDING_MODEL
        device = device or self._default_device

        with self._model_lock:
            # Check if we need to swap models
            needs_reload = (
                self._embedder is None
                or self._embedder_model_name != model_name
                or self._embedder_device != device
            )

            if needs_reload:
                self._unload_embedder()
                self._load_embedder(model_name, device)

            return self._embedder  # type: ignore[return-value]

    def get_reranker(
        self,
        model_name: Optional[str] = None,
        top_n: int = 5,
        device: Optional[str] = None,
    ) -> SentenceTransformerRerank:
        """Get reranker, loading or swapping if needed.

        Args:
            model_name: HuggingFace model path (default: BAAI/bge-reranker-v2-m3)
            top_n: Number of top results to return after reranking
            device: Device to load on (default: class default device)

        Returns:
            SentenceTransformerRerank instance
        """
        model_name = model_name or DEFAULT_RERANKER_MODEL
        device = device or self._default_device

        with self._model_lock:
            # Check if we need to swap models (top_n change doesn't require reload)
            needs_reload = (
                self._reranker is None
                or self._reranker_model_name != model_name
                or self._reranker_device != device
            )

            if needs_reload:
                self._unload_reranker()
                self._load_reranker(model_name, top_n, device)
            elif self._reranker_top_n != top_n and self._reranker is not None:
                # Just update top_n without full reload
                self._reranker.top_n = top_n
                self._reranker_top_n = top_n

            return self._reranker  # type: ignore[return-value]

    def _load_embedder(self, model_name: str, device: str) -> None:
        """Load embedder model with config-driven settings.

        Loads model-specific configuration from config.yaml (or built-in defaults)
        and applies optimizations like batch size, dtype, flash attention, etc.

        Args:
            model_name: HuggingFace model path
            device: Device to load on
        """
        logger.info(f"Loading embedder: {model_name} on {device.upper()}")
        print(f"Loading Embedder: {model_name} on {device.upper()}")

        # Get model-specific config
        model_config = self._get_embedding_model_config(model_name)

        # Determine batch size based on device
        batch_size = (
            model_config.batch_size_cuda
            if device == "cuda"
            else model_config.batch_size_cpu
        )

        # Build model_kwargs
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": model_config.trust_remote_code
        }

        # Add torch_dtype if specified
        if model_config.torch_dtype:
            import torch

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if model_config.torch_dtype in dtype_map:
                model_kwargs["torch_dtype"] = dtype_map[model_config.torch_dtype]

        # Try to enable flash attention if configured
        if model_config.flash_attention:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled for embedding model")
            except ImportError:
                logger.debug("Flash Attention not available, using default attention")

        # Build tokenizer_kwargs
        tokenizer_kwargs = None
        if model_config.padding_side:
            tokenizer_kwargs = {"padding_side": model_config.padding_side}

        logger.info(
            f"Creating embedding model: {model_name} "
            f"(batch_size={batch_size}, dtype={model_config.torch_dtype or 'default'})"
        )

        self._embedder = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            embed_batch_size=batch_size,
        )
        self._embedder_model_name = model_name
        self._embedder_device = device

        logger.info(f"Embedder loaded: {model_name}")

    def _get_embedding_model_config(self, model_name: str) -> EmbeddingModelConfig:
        """Get configuration for a specific embedding model.

        Tries to load from config.yaml first, falls back to built-in defaults.

        Args:
            model_name: HuggingFace model path (e.g., "BAAI/bge-m3")

        Returns:
            EmbeddingModelConfig with settings for this model
        """
        try:
            from tensortruth.app_utils.config import load_config

            config = load_config()
            return config.rag.get_embedding_model_config(model_name)
        except Exception:
            # Fall back to built-in defaults if config loading fails
            if model_name in DEFAULT_EMBEDDING_MODEL_CONFIGS:
                return EmbeddingModelConfig(
                    **DEFAULT_EMBEDDING_MODEL_CONFIGS[model_name]
                )
            return DEFAULT_EMBEDDING_MODEL_CONFIG

    def _unload_embedder(self) -> None:
        """Release embedder memory."""
        if self._embedder is not None:
            logger.info(f"Unloading embedder: {self._embedder_model_name}")

            # Delete the model
            del self._embedder

            # Clear GPU cache if available
            self._clear_gpu_cache()

            # Garbage collect
            gc.collect()

            logger.debug("Embedder unloaded and memory cleared")

        self._embedder = None
        self._embedder_model_name = None
        self._embedder_device = None

    def _load_reranker(self, model_name: str, top_n: int, device: str) -> None:
        """Load reranker model.

        Args:
            model_name: HuggingFace model path
            top_n: Number of top results to return
            device: Device to load on
        """
        logger.info(f"Loading reranker: {model_name} on {device.upper()}")
        print(f"Loading Reranker: {model_name} on {device.upper()}")

        self._reranker = SentenceTransformerRerank(
            model=model_name,
            top_n=top_n,
            device=device,
        )
        self._reranker_model_name = model_name
        self._reranker_top_n = top_n
        self._reranker_device = device

        logger.info(f"Reranker loaded: {model_name}")

    def _unload_reranker(self) -> None:
        """Release reranker memory."""
        if self._reranker is not None:
            logger.info(f"Unloading reranker: {self._reranker_model_name}")

            # Delete the model
            del self._reranker

            # Clear GPU cache if available
            self._clear_gpu_cache()

            # Garbage collect
            gc.collect()

            logger.debug("Reranker unloaded and memory cleared")

        self._reranker = None
        self._reranker_model_name = None
        self._reranker_top_n = None
        self._reranker_device = None

    def unload_all(self) -> None:
        """Unload all models and release memory.

        Call this before shutting down to ensure clean memory release.
        """
        with self._model_lock:
            self._unload_embedder()
            self._unload_reranker()

        logger.info("All models unloaded")

    def _clear_gpu_cache(self) -> None:
        """Clear GPU cache if torch is available and GPU is in use."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't have explicit cache clearing, but gc helps
                pass
        except ImportError:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Get current model manager status.

        Returns:
            Dict with loaded model information
        """
        return {
            "embedder": {
                "loaded": self._embedder is not None,
                "model_name": self._embedder_model_name,
                "device": self._embedder_device,
            },
            "reranker": {
                "loaded": self._reranker is not None,
                "model_name": self._reranker_model_name,
                "top_n": self._reranker_top_n,
                "device": self._reranker_device,
            },
            "default_device": self._default_device,
        }

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage of loaded models.

        Calculates memory by summing parameter sizes of the underlying PyTorch models.

        Returns:
            Dict with memory info for embedder and reranker:
            {
                "embedder": {"model_name": str, "device": str, "memory_gb": float} | None,
                "reranker": {"model_name": str, "device": str, "memory_gb": float} | None,
                "total_gb": float
            }
        """
        result: Dict[str, Any] = {
            "embedder": None,
            "reranker": None,
            "total_gb": 0.0,
        }

        # Calculate embedder memory
        if self._embedder is not None:
            embedder_mem = self._get_model_memory_gb(self._embedder)
            if embedder_mem is not None:
                result["embedder"] = {
                    "model_name": self._embedder_model_name,
                    "device": self._embedder_device,
                    "memory_gb": embedder_mem,
                }
                result["total_gb"] += embedder_mem

        # Calculate reranker memory
        if self._reranker is not None:
            reranker_mem = self._get_model_memory_gb(self._reranker)
            if reranker_mem is not None:
                result["reranker"] = {
                    "model_name": self._reranker_model_name,
                    "device": self._reranker_device,
                    "memory_gb": reranker_mem,
                }
                result["total_gb"] += reranker_mem

        return result

    def _get_model_memory_gb(self, wrapper: Any) -> Optional[float]:
        """Calculate memory usage of a model wrapper in GB.

        Accesses the underlying PyTorch model and sums parameter memory.

        Args:
            wrapper: LlamaIndex model wrapper (HuggingFaceEmbedding or SentenceTransformerRerank)

        Returns:
            Memory usage in GB, or None if unable to calculate
        """
        try:
            model = None

            # HuggingFaceEmbedding stores model in _model attribute
            if hasattr(wrapper, "_model"):
                model = wrapper._model
            # SentenceTransformerRerank stores it in model attribute
            elif hasattr(wrapper, "model"):
                model = wrapper.model
            # sentence-transformers model has _first_module()
            if model is not None and hasattr(model, "_first_module"):
                model = model._first_module()
            # Or it might have an 'auto_model' attribute
            if model is not None and hasattr(model, "auto_model"):
                model = model.auto_model
            # Or access via [0] for sequential models
            if model is not None and hasattr(model, "__getitem__"):
                try:
                    first = model[0]
                    if hasattr(first, "auto_model"):
                        model = first.auto_model
                except (IndexError, TypeError):
                    pass

            if model is None:
                return None

            # Sum parameter memory
            total_bytes = 0
            if hasattr(model, "parameters"):
                for param in model.parameters():
                    total_bytes += param.numel() * param.element_size()

            return total_bytes / (1024**3)
        except Exception as e:
            logger.debug(f"Could not calculate model memory: {e}")
            return None

    @property
    def current_embedding_model(self) -> Optional[str]:
        """Get the currently loaded embedding model name."""
        return self._embedder_model_name

    @property
    def current_reranker_model(self) -> Optional[str]:
        """Get the currently loaded reranker model name."""
        return self._reranker_model_name

    def is_embedder_loaded(self, model_name: Optional[str] = None) -> bool:
        """Check if a specific embedder is loaded.

        Args:
            model_name: Model to check (if None, checks if any embedder is loaded)

        Returns:
            True if the specified model (or any model) is loaded
        """
        if model_name is None:
            return self._embedder is not None
        return self._embedder_model_name == model_name

    def is_reranker_loaded(self, model_name: Optional[str] = None) -> bool:
        """Check if a specific reranker is loaded.

        Args:
            model_name: Model to check (if None, checks if any reranker is loaded)

        Returns:
            True if the specified model (or any model) is loaded
        """
        if model_name is None:
            return self._reranker is not None
        return self._reranker_model_name == model_name
