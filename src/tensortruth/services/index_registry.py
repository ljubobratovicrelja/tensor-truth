"""Process-wide registry of loaded module indexes.

RAG sessions used to share one engine, so loading a module for one session
rebuilt the engine for every session. This registry owns the expensive part --
the Chroma handle and VectorStoreIndex per module -- and refcounts it, so two
sessions using the same module share exactly one copy. Sessions keep only a
thin retrieval set (see RAGService) on top.

Embedding models are not owned here: the embedder instance comes from
ModelManager and is passed in per call. Entries are keyed by
(embedding_model_id, module_name) so a session on a different embedder can
never reuse an index built with another one.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.retrievers import BaseRetriever

from tensortruth.rag_engine import load_module_index

logger = logging.getLogger(__name__)

DEFAULT_MAX_CACHED_INDEXES = 8

EntryKey = Tuple[str, str]


@dataclass
class IndexEntry:
    """One loaded module index plus its bookkeeping."""

    key: EntryKey
    index: Any
    retriever: BaseRetriever
    refcount: int = 0
    last_used: float = field(default_factory=time.monotonic)
    doc_count: int = 0


class ModuleIndexRegistry:
    """Refcounted, LRU-capped store of module indexes.

    Loading happens while holding the lock. That serializes first opens of
    different modules, but it guarantees concurrent acquires of the same module
    cannot open the same Chroma directory twice -- which is the failure mode
    that matters (duplicate handles and double memory).
    """

    def __init__(self, max_entries: Optional[int] = None):
        self._lock = threading.RLock()
        self._entries: Dict[EntryKey, IndexEntry] = {}
        self._max_entries_override = max_entries

    @property
    def max_entries(self) -> int:
        if self._max_entries_override is not None:
            return max(0, self._max_entries_override)
        try:
            from tensortruth.app_utils.config import load_config

            return max(0, int(load_config().rag.max_cached_indexes))
        except Exception:
            return DEFAULT_MAX_CACHED_INDEXES

    def acquire(
        self,
        modules: List[str],
        embedding_model: str,
        embed_model: Any,
        similarity_top_k: int,
        skipped: Optional[List[str]] = None,
    ) -> List[BaseRetriever]:
        """Load missing modules and increment their refcounts.

        Args:
            modules: Module names requested by a session.
            embedding_model: Canonical embedding model id for the index path.
            embed_model: Shared embedder instance (ModelManager).
            similarity_top_k: Candidate pool size for the inner retriever.
            skipped: Optional out-parameter receiving names with no index on
                disk. They are logged, reported here, and never registered.

        Returns:
            Retrievers in the same order as the loadable subset of ``modules``.
        """
        retrievers: List[BaseRetriever] = []
        with self._lock:
            for module in dict.fromkeys(modules):
                key = (embedding_model, module)
                entry = self._entries.get(key)
                if entry is None:
                    loaded = load_module_index(
                        module, embedding_model, embed_model, similarity_top_k
                    )
                    if loaded is None:
                        if skipped is not None:
                            skipped.append(module)
                        continue
                    retriever, index = loaded
                    entry = IndexEntry(
                        key=key,
                        index=index,
                        retriever=retriever,
                        doc_count=_count_docs(index),
                    )
                    self._entries[key] = entry
                entry.refcount += 1
                entry.last_used = time.monotonic()
                retrievers.append(entry.retriever)

            self._evict_locked()
        return retrievers

    def release(self, modules: List[str], embedding_model: str) -> None:
        """Decrement refcounts, then evict idle entries over the cap."""
        with self._lock:
            now = time.monotonic()
            for module in dict.fromkeys(modules):
                entry = self._entries.get((embedding_model, module))
                if entry is None:
                    continue
                entry.refcount = max(0, entry.refcount - 1)
                if entry.refcount == 0:
                    entry.last_used = now
            self._evict_locked()

    def peek(self, module: str, embedding_model: str) -> Optional[IndexEntry]:
        """Return an entry without changing its refcount (diagnostics/tests)."""
        with self._lock:
            return self._entries.get((embedding_model, module))

    def stats(self) -> Dict[str, Any]:
        """Snapshot of the registry for logging and tests."""
        with self._lock:
            entries = [
                {
                    "module": module,
                    "embedding_model": model,
                    "refcount": entry.refcount,
                    "doc_count": entry.doc_count,
                    "last_used": entry.last_used,
                }
                for (model, module), entry in self._entries.items()
            ]
            return {
                "count": len(entries),
                "max_entries": self.max_entries,
                "total_refs": sum(e["refcount"] for e in entries),
                "entries": entries,
            }

    def clear(self) -> None:
        """Drop every handle regardless of refcount."""
        with self._lock:
            self._entries.clear()

    def _evict_locked(self) -> None:
        """Evict least-recently-used idle entries beyond the cap.

        Entries with refcount > 0 are never touched: an active session would
        otherwise lose its retriever mid-turn.
        """
        cap = self.max_entries
        while len(self._entries) > cap:
            idle = [e for e in self._entries.values() if e.refcount == 0]
            if not idle:
                return
            victim = min(idle, key=lambda e: e.last_used)
            del self._entries[victim.key]
            logger.info(
                "Evicted idle module index %s (docs=%s)",
                victim.key[1],
                victim.doc_count,
            )


def _count_docs(index: Any) -> int:
    """Approximate entry size: number of nodes in the index docstore."""
    try:
        return len(index.docstore.docs)
    except Exception:
        return 0


_registry: Optional[ModuleIndexRegistry] = None
_registry_lock = threading.Lock()


def get_instance() -> ModuleIndexRegistry:
    """Get the process-wide registry singleton."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ModuleIndexRegistry()
        return _registry


def reset_instance(registry: Optional[ModuleIndexRegistry] = None) -> None:
    """Replace the singleton (tests, or explicit reconfiguration)."""
    global _registry
    with _registry_lock:
        _registry = registry
