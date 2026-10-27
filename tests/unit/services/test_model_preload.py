"""Unit tests for the background RAG model preload manager.

No real weights are loaded and CUDA is never touched: the ModelManager factory
and model factories are faked.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from tensortruth.services import model_preload
from tensortruth.services.model_preload import ModelPreloadManager


class FakeEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.warmups: list = []

    def get_text_embedding(self, text: str) -> list:
        if self.fail:
            raise RuntimeError("embedder warm-up blew up")
        self.warmups.append(text)
        return [0.0, 1.0]


class FakeReranker:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list = []

    def postprocess_nodes(self, nodes, query_bundle=None):
        if self.fail:
            raise RuntimeError("reranker warm-up blew up")
        self.calls.append((nodes, query_bundle))
        return nodes


class FakeModelManager:
    def __init__(
        self,
        embedder=None,
        reranker=None,
        embed_gate: threading.Event | None = None,
        embed_error: Exception | None = None,
    ) -> None:
        self.embedder = embedder or FakeEmbedder()
        self.reranker = reranker or FakeReranker()
        self.embed_gate = embed_gate
        self.embed_error = embed_error
        self.embed_calls: list = []
        self.rerank_calls: list = []

    def get_embedder(self, model_name=None, device=None):
        self.embed_calls.append((model_name, device))
        if self.embed_gate is not None:
            assert self.embed_gate.wait(timeout=5), "gate was never released"
        if self.embed_error is not None:
            raise self.embed_error
        return self.embedder

    def get_reranker(self, model_name=None, top_n=5, device=None):
        self.rerank_calls.append((model_name, top_n, device))
        return self.reranker


def make_config() -> SimpleNamespace:
    rag = SimpleNamespace(
        default_embedding_model="fake/embedder",
        default_reranker="fake/reranker",
        default_top_n=4,
        default_device="cuda",
    )
    return SimpleNamespace(rag=rag)


def make_manager(fake: FakeModelManager) -> ModelPreloadManager:
    return ModelPreloadManager(
        config_loader=lambda: make_config(),
        manager_factory=lambda: fake,
        device_resolver=lambda config: config.rag.default_device,
    )


@pytest.mark.unit
class TestModelPreloadManager:
    def test_loads_both_models_and_warms_them(self):
        fake = FakeModelManager()
        manager = make_manager(fake)

        manager.start()
        manager.wait(timeout=10)
        status = manager.status()

        assert fake.embed_calls == [("fake/embedder", "cuda")]
        assert fake.rerank_calls == [("fake/reranker", 4, "cuda")]
        assert fake.embedder.warmups == [model_preload.WARMUP_TEXT]
        assert len(fake.reranker.calls) == 1

        for kind in ("embedder", "reranker"):
            state = status[kind]
            assert state["state"] == "ready"
            assert state["step"] == "ready"
            assert state["model_name"].startswith("fake/")
            assert state["device"] == "cuda"
            assert state["error"] is None
            assert state["started_at"] is not None
            assert state["finished_at"] is not None
            assert state["elapsed_seconds"] >= 0

        assert status["all_ready"] is True
        assert status["any_loading"] is False

    def test_concurrent_trigger_joins_in_flight_load(self):
        gate = threading.Event()
        fake = FakeModelManager(embed_gate=gate)
        manager = make_manager(fake)

        first = manager.start()
        assert first["embedder"]["state"] in ("idle", "loading")

        # Second and third triggers must not spawn another load.
        second = manager.start()
        third = manager.start()
        assert second["embedder"]["state"] in ("idle", "loading")
        assert third["embedder"]["state"] in ("idle", "loading")

        gate.set()
        manager.wait(timeout=10)

        assert len(fake.embed_calls) == 1
        assert len(fake.rerank_calls) == 1
        assert manager.status()["all_ready"] is True

    def test_repeated_trigger_after_ready_is_noop(self):
        fake = FakeModelManager()
        manager = make_manager(fake)

        manager.start()
        manager.wait(timeout=10)
        manager.start()
        manager.wait(timeout=10)

        assert len(fake.embed_calls) == 1
        assert len(fake.rerank_calls) == 1
        assert manager.status()["all_ready"] is True

    def test_force_reloads(self):
        fake = FakeModelManager()
        manager = make_manager(fake)

        manager.start()
        manager.wait(timeout=10)
        manager.start(force=True)
        manager.wait(timeout=10)

        assert len(fake.embed_calls) == 2
        assert len(fake.rerank_calls) == 2

    def test_embedder_failure_is_recorded_and_reranker_still_loads(self):
        fake = FakeModelManager(embed_error=RuntimeError("CUDA out of memory"))
        manager = make_manager(fake)

        manager.start()  # must not raise
        manager.wait(timeout=10)
        status = manager.status()

        assert status["embedder"]["state"] == "error"
        assert status["embedder"]["step"] == "failed"
        assert "CUDA out of memory" in status["embedder"]["error"]
        assert status["embedder"]["finished_at"] is not None
        assert status["reranker"]["state"] == "ready"
        assert status["all_ready"] is False
        assert status["any_loading"] is False

    def test_warmup_failure_marks_model_error(self):
        fake = FakeModelManager(embedder=FakeEmbedder(fail=True))
        manager = make_manager(fake)

        manager.start()
        manager.wait(timeout=10)

        status = manager.status()
        assert status["embedder"]["state"] == "error"
        assert "warm-up" in status["embedder"]["error"]
        assert status["reranker"]["state"] == "ready"

    def test_unexpected_thread_failure_does_not_escape(self):
        def exploding_factory():
            raise RuntimeError("config is broken")

        manager = ModelPreloadManager(
            config_loader=lambda: make_config(),
            manager_factory=exploding_factory,
            device_resolver=lambda config: "cpu",
        )

        manager.start()
        manager.wait(timeout=10)
        status = manager.status()

        assert status["embedder"]["state"] == "error"
        assert status["reranker"]["state"] == "error"
        assert "config is broken" in status["embedder"]["error"]

    def test_status_is_cheap_and_reports_elapsed_while_loading(self):
        gate = threading.Event()
        fake = FakeModelManager(embed_gate=gate)
        manager = make_manager(fake)

        manager.start()
        time.sleep(0.05)
        status = manager.status()
        assert status["embedder"]["state"] == "loading"
        assert status["embedder"]["step"] in (
            "resolving device",
            "loading weights",
            "warm-up",
        )
        assert status["embedder"]["elapsed_seconds"] >= 0
        gate.set()
        manager.wait(timeout=10)

    def test_idle_state_shape_before_any_trigger(self):
        manager = make_manager(FakeModelManager())
        status = manager.status()

        assert status["embedder"]["state"] == "idle"
        assert status["reranker"]["state"] == "idle"
        assert status["any_loading"] is False
        assert status["all_ready"] is False
        assert status["embedder"]["elapsed_seconds"] == 0.0
