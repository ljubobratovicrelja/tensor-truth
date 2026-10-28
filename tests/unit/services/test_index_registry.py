"""Unit tests for the refcounted module index registry."""

import threading
import time
from unittest.mock import Mock, patch

import pytest

from tensortruth.app_utils.config_schema import RAGConfig
from tensortruth.services.index_registry import ModuleIndexRegistry

LOADER = "tensortruth.services.index_registry.load_module_index"


def _fake_load(module, embedding_model, embed_model, similarity_top_k):
    """Stand-in loader returning a distinct retriever per module."""
    retriever = Mock(name=f"retriever:{module}")
    index = Mock(name=f"index:{module}")
    index.docstore.docs = {f"{module}-{i}": i for i in range(3)}
    return retriever, index


@pytest.mark.unit
class TestModuleIndexRegistry:
    def test_same_module_acquired_by_two_owners_loads_once(self):
        registry = ModuleIndexRegistry(max_entries=8)
        with patch(LOADER, side_effect=_fake_load) as loader:
            first = registry.acquire(["pytorch"], "m1", Mock(), 5)
            second = registry.acquire(["pytorch"], "m1", Mock(), 5)

        assert loader.call_count == 1
        assert first[0] is second[0]
        entry = registry.peek("pytorch", "m1")
        assert entry is not None
        assert entry.refcount == 2
        assert entry.doc_count == 3

    def test_release_to_zero_unloads_when_cap_is_zero(self):
        registry = ModuleIndexRegistry(max_entries=0)
        with patch(LOADER, side_effect=_fake_load):
            registry.acquire(["pytorch"], "m1", Mock(), 5)
        assert registry.peek("pytorch", "m1") is not None

        registry.release(["pytorch"], "m1")

        assert registry.peek("pytorch", "m1") is None
        assert registry.stats()["count"] == 0

    def test_idle_entries_are_kept_and_evicted_lru(self):
        registry = ModuleIndexRegistry(max_entries=1)
        with patch(LOADER, side_effect=_fake_load):
            registry.acquire(["a"], "m1", Mock(), 5)
            registry.release(["a"], "m1")
            assert registry.peek("a", "m1") is not None

            registry.acquire(["b"], "m1", Mock(), 5)

        # Cap 1 with both idle: least recently used ("a") is evicted.
        assert registry.peek("a", "m1") is None
        assert registry.peek("b", "m1") is not None

    def test_eviction_never_touches_active_entries(self):
        registry = ModuleIndexRegistry(max_entries=1)
        with patch(LOADER, side_effect=_fake_load):
            registry.acquire(["a"], "m1", Mock(), 5)
            registry.acquire(["b"], "m1", Mock(), 5)

        # Both active (refcount 1) despite cap 1: nothing evictable.
        assert registry.peek("a", "m1") is not None
        assert registry.peek("b", "m1") is not None
        assert registry.stats()["count"] == 2

        registry.release(["a"], "m1")

        assert registry.peek("a", "m1") is None
        assert registry.peek("b", "m1") is not None

    def test_concurrent_acquire_of_same_module_loads_once(self):
        registry = ModuleIndexRegistry(max_entries=8)

        def slow_load(module, embedding_model, embed_model, similarity_top_k):
            time.sleep(0.05)
            return _fake_load(module, embedding_model, embed_model, similarity_top_k)

        results = []
        errors = []

        def worker():
            try:
                results.append(registry.acquire(["pytorch"], "m1", Mock(), 5))
            except Exception as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        with patch(LOADER, side_effect=slow_load) as loader:
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert errors == []
        assert loader.call_count == 1
        assert registry.peek("pytorch", "m1").refcount == 2
        assert results[0][0] is results[1][0]

    def test_unknown_module_reported_without_poisoning_registry(self):
        registry = ModuleIndexRegistry(max_entries=8)

        def load(module, embedding_model, embed_model, similarity_top_k):
            if module == "ghost":
                return None
            return _fake_load(module, embedding_model, embed_model, similarity_top_k)

        skipped = []
        with patch(LOADER, side_effect=load):
            retrievers = registry.acquire(
                ["ghost", "pytorch"], "m1", Mock(), 5, skipped=skipped
            )
            again = registry.acquire(["pytorch"], "m1", Mock(), 5)

        assert skipped == ["ghost"]
        assert len(retrievers) == 1
        assert again[0] is retrievers[0]
        assert registry.peek("ghost", "m1") is None
        assert registry.peek("pytorch", "m1").refcount == 2

    def test_entries_are_partitioned_by_embedding_model(self):
        registry = ModuleIndexRegistry(max_entries=8)
        with patch(LOADER, side_effect=_fake_load) as loader:
            registry.acquire(["pytorch"], "m1", Mock(), 5)
            registry.acquire(["pytorch"], "m2", Mock(), 5)

        assert loader.call_count == 2
        assert registry.peek("pytorch", "m1") is not registry.peek("pytorch", "m2")
        assert registry.stats()["count"] == 2

    def test_release_unknown_module_is_noop(self):
        registry = ModuleIndexRegistry(max_entries=8)
        registry.release(["never-loaded"], "m1")
        assert registry.stats()["count"] == 0

    def test_clear_drops_everything(self):
        registry = ModuleIndexRegistry(max_entries=8)
        with patch(LOADER, side_effect=_fake_load):
            registry.acquire(["a", "b"], "m1", Mock(), 5)

        registry.clear()

        assert registry.stats()["count"] == 0


@pytest.mark.unit
def test_rag_config_max_cached_indexes_validation():
    assert RAGConfig().max_cached_indexes == 8
    assert RAGConfig(max_cached_indexes=0).max_cached_indexes == 0
    with pytest.raises(ValueError):
        RAGConfig(max_cached_indexes=-1)
