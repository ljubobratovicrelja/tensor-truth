"""Background preloading of the RAG embedding model and reranker.

The first RAG query otherwise pays the full HuggingFace load cost (and the API
request that triggers it blocks on it). This module warms both models in a
daemon thread at startup / on the first frontend status request, records a
per-model state machine for the UI, and makes repeated triggers idempotent so
that two callers can never allocate two copies of the same model.

Loading goes through :class:`~tensortruth.services.model_manager.ModelManager`,
so a preloaded model is the same instance the RAG engine later receives.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# State machine values
IDLE = "idle"
LOADING = "loading"
READY = "ready"
ERROR = "error"

WARMUP_TEXT = "TensorTruth warm-up probe."


@dataclass
class ModelLoadState:
    """State of a single model's background load."""

    kind: str
    model_name: Optional[str] = None
    device: Optional[str] = None
    state: str = IDLE
    step: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Serialize for the status endpoint (cheap, no model access)."""
        now = time.time() if now is None else now
        elapsed = 0.0
        if self.started_at is not None:
            end = self.finished_at if self.finished_at is not None else now
            elapsed = max(0.0, end - self.started_at)
        return {
            "kind": self.kind,
            "state": self.state,
            "model_name": self.model_name,
            "device": self.device,
            "step": self.step,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(elapsed, 1),
            "error": self.error,
        }


class ModelPreloadManager:
    """Singleton that warms the configured embedder and reranker off-thread.

    ``start()`` is idempotent: while a load thread is alive, further calls
    return the current snapshot and join that load instead of starting a second
    one. After a successful load, calls are a no-op unless ``force=True``.
    """

    _instance: Optional["ModelPreloadManager"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        config_loader: Optional[Callable[[], Any]] = None,
        manager_factory: Optional[Callable[[], Any]] = None,
        device_resolver: Optional[Callable[[Any], str]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._embedder = ModelLoadState(kind="embedder")
        self._reranker = ModelLoadState(kind="reranker")
        self._config_loader = config_loader
        self._manager_factory = manager_factory
        self._device_resolver = device_resolver

    # -- singleton ---------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "ModelPreloadManager":
        """Get the process-wide preload manager."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the singleton (tests only). Does not unload models."""
        with cls._instance_lock:
            inst = cls._instance
            if inst is not None:
                thread = inst._thread
                if thread is not None and thread.is_alive():
                    thread.join(timeout=30)
            cls._instance = None

    # -- public API --------------------------------------------------------

    def start(self, force: bool = False) -> Dict[str, Any]:
        """Kick off a background preload if one is not already running.

        Never blocks on model work and never raises: a failure is recorded in
        the per-model state for the UI, and the lazy load path in RAGService is
        untouched.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                snapshot = self._snapshot_locked()
            else:
                both_ready = (
                    self._embedder.state == READY and self._reranker.state == READY
                )
                if both_ready and not force:
                    snapshot = self._snapshot_locked()
                else:
                    self._embedder = ModelLoadState(kind="embedder", state=IDLE)
                    self._reranker = ModelLoadState(kind="reranker", state=IDLE)
                    self._thread = threading.Thread(
                        target=self._run,
                        name="tensortruth-model-preload",
                        daemon=True,
                    )
                    self._thread.start()
                    snapshot = self._snapshot_locked()

        logger.info(
            "Model preload snapshot: embedder=%s, reranker=%s",
            snapshot["embedder"]["state"],
            snapshot["reranker"]["state"],
        )
        return snapshot

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for the current background load to finish (tests/ops helper)."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def status(self) -> Dict[str, Any]:
        """Return the current snapshot. Cheap: no model or GPU calls."""
        with self._lock:
            return self._snapshot_locked()

    # -- internals ---------------------------------------------------------

    def _snapshot_locked(self) -> Dict[str, Any]:
        now = time.time()
        embedder = self._embedder.to_dict(now)
        reranker = self._reranker.to_dict(now)
        both_ready = embedder["state"] == READY and reranker["state"] == READY
        any_loading = LOADING in (embedder["state"], reranker["state"])
        return {
            "embedder": embedder,
            "reranker": reranker,
            "any_loading": any_loading,
            "all_ready": both_ready,
        }

    def _update(self, kind: str, **fields: Any) -> None:
        with self._lock:
            state = self._embedder if kind == "embedder" else self._reranker
            for key, value in fields.items():
                setattr(state, key, value)

    def _run(self) -> None:
        """Load both models sequentially; failures are recorded, not raised."""
        try:
            config = self._load_config()
            manager = self._manager_factory_instance()
            device = self._resolve_device(config)
            embed_model, rerank_model, top_n = self._targets(config)

            self._load_embedder(manager, embed_model, device)
            self._load_reranker(manager, rerank_model, top_n, device)
        except Exception as e:  # defensive: startup must never see this thread
            logger.exception("Model preload thread failed unexpectedly: %s", e)
            with self._lock:
                for state in (self._embedder, self._reranker):
                    if state.state in (IDLE, LOADING):
                        state.state = ERROR
                        state.step = "failed"
                        state.finished_at = time.time()
                        state.error = f"Preload failed: {e}"

    def _load_config(self) -> Any:
        if self._config_loader is not None:
            return self._config_loader()
        from tensortruth.app_utils.config import load_config

        return load_config()

    def _manager_factory_instance(self) -> Any:
        if self._manager_factory is not None:
            return self._manager_factory()
        from tensortruth.services.model_manager import ModelManager

        return ModelManager.get_instance()

    def _targets(self, config: Any) -> tuple:
        rag = config.rag
        return (
            rag.default_embedding_model,
            rag.default_reranker,
            int(rag.default_top_n),
        )

    def _resolve_device(self, config: Any) -> str:
        if self._device_resolver is not None:
            return self._device_resolver(config)
        device = getattr(getattr(config, "rag", None), "default_device", None)
        if device in ("cpu", "cuda", "mps"):
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_embedder(self, manager: Any, model_name: str, device: str) -> None:
        self._update(
            "embedder",
            model_name=model_name,
            device=device,
            state=LOADING,
            step="resolving device",
            started_at=time.time(),
            finished_at=None,
            error=None,
        )
        try:
            self._update("embedder", step="loading weights")
            embedder = manager.get_embedder(model_name=model_name, device=device)
            self._update("embedder", step="warm-up")
            _warm_up_embedder(embedder)
            self._update(
                "embedder",
                state=READY,
                step="ready",
                finished_at=time.time(),
            )
            logger.info("Embedder preloaded: %s on %s", model_name, device)
        except Exception as e:
            logger.warning("Embedder preload failed (%s): %s", model_name, e)
            self._update(
                "embedder",
                state=ERROR,
                step="failed",
                finished_at=time.time(),
                error=str(e),
            )

    def _load_reranker(
        self, manager: Any, model_name: str, top_n: int, device: str
    ) -> None:
        self._update(
            "reranker",
            model_name=model_name,
            device=device,
            state=LOADING,
            step="resolving device",
            started_at=time.time(),
            finished_at=None,
            error=None,
        )
        try:
            self._update("reranker", step="loading weights")
            reranker = manager.get_reranker(
                model_name=model_name, top_n=top_n, device=device
            )
            self._update("reranker", step="warm-up")
            _warm_up_reranker(reranker)
            self._update(
                "reranker",
                state=READY,
                step="ready",
                finished_at=time.time(),
            )
            logger.info("Reranker preloaded: %s on %s", model_name, device)
        except Exception as e:
            logger.warning("Reranker preload failed (%s): %s", model_name, e)
            self._update(
                "reranker",
                state=ERROR,
                step="failed",
                finished_at=time.time(),
                error=str(e),
            )


def _warm_up_embedder(embedder: Any) -> None:
    """Run one tiny inference so the first real query skips kernel init."""
    embedder.get_text_embedding(WARMUP_TEXT)


def _warm_up_reranker(reranker: Any) -> None:
    """Score one trivial pair so the reranker's first call is fast."""
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    node = NodeWithScore(node=TextNode(text=WARMUP_TEXT), score=1.0)
    reranker.postprocess_nodes([node], query_bundle=QueryBundle(query_str=WARMUP_TEXT))


def get_preload_manager() -> ModelPreloadManager:
    """Return the process-wide preload manager."""
    return ModelPreloadManager.get_instance()


def trigger_model_preload(force: bool = False) -> Dict[str, Any]:
    """Idempotently start warming the RAG models and return the snapshot."""
    return get_preload_manager().start(force=force)
