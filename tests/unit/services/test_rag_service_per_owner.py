"""Per-owner (per-session) RAG retrieval set tests."""

import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from tensortruth.services.index_registry import ModuleIndexRegistry
from tensortruth.services.rag_service import RAGService

LOADER = "tensortruth.services.index_registry.load_module_index"
COMPOSE = "tensortruth.services.rag_service.compose_chat_engine"


class FakeRetriever:
    def __init__(self, label):
        self.label = label

    def retrieve(self, query):
        return [NodeWithScore(node=TextNode(text=f"{self.label}:{query}"), score=0.9)]


class FakeComposite:
    def __init__(self, retrievers):
        self.retrievers = list(retrievers)

    def retrieve(self, query):
        nodes = []
        for retriever in self.retrievers:
            nodes.extend(retriever.retrieve(query))
        return nodes

    def clear_cache(self):
        pass


def _loader(module, embedding_model, embed_model, similarity_top_k):
    return FakeRetriever(module), Mock(name=f"index:{module}")


def _compose(retrievers, params, rag_device=None):
    return SimpleNamespace(
        _llm=Mock(),
        _retriever=FakeComposite(retrievers),
        _node_postprocessors=[],
        _condense_prompt_template=None,
        _skip_condense=True,
        memory=Mock(),
    )


def _config():
    config = Mock()
    config.history_cleaning.enabled = False
    config.conversation.max_history_turns = 3
    return config


@pytest.fixture
def service():
    registry = ModuleIndexRegistry(max_entries=8)
    with (
        patch(LOADER, side_effect=_loader),
        patch(COMPOSE, side_effect=_compose),
        patch(
            "tensortruth.services.rag_service.resolve_engine_settings",
            return_value=("BAAI/bge-m3", "cpu"),
        ),
        patch("tensortruth.services.rag_service.get_similarity_top_k", return_value=5),
        patch("tensortruth.services.rag_service.get_embed_model", return_value=Mock()),
    ):
        yield RAGService(_config(), index_registry=registry)


def _text(result):
    return " ".join(node.get_content() for node in result.source_nodes)


@pytest.mark.unit
class TestPerOwnerSets:
    def test_two_owners_retrieve_from_their_own_modules(self, service):
        service.load_engine(["pytorch"], {}, owner="A")
        service.load_engine(["opencv"], {}, owner="B")

        assert _text(service.retrieve("q", owner="A")) == "pytorch:q"
        assert _text(service.retrieve("q", owner="B")) == "opencv:q"

    def test_owner_a_set_is_unchanged_after_b_loads(self, service):
        service.load_engine(["pytorch"], {}, owner="A")
        a_engine = service._sets["A"].chat_engine

        service.load_engine(["pytorch", "opencv"], {}, owner="B")

        a_set = service._sets["A"]
        assert a_set.chat_engine is a_engine
        assert a_set.modules == ["pytorch"]
        assert service.get_loaded_modules("A") == ["pytorch"]
        assert service.get_loaded_modules("B") == ["pytorch", "opencv"]
        assert _text(service.retrieve("q", owner="A")) == "pytorch:q"

    def test_reload_for_one_owner_does_not_touch_other(self, service):
        service.load_engine(["pytorch"], {}, owner="A")
        service.load_engine(["opencv"], {}, owner="B")

        service.load_engine(["pytorch", "opencv"], {}, owner="A")

        assert service.get_loaded_modules("B") == ["opencv"]
        assert _text(service.retrieve("q", owner="B")) == "opencv:q"
        assert len(service.retrieve("q", owner="A").source_nodes) == 2

    def test_release_decrements_shared_module_exactly_once(self, service):
        registry = service._registry
        service.load_engine(["pytorch"], {}, owner="A")
        service.load_engine(["pytorch"], {}, owner="B")
        assert registry.peek("pytorch", "BAAI/bge-m3").refcount == 2

        service.release("A")

        assert "A" not in service._sets
        assert registry.peek("pytorch", "BAAI/bge-m3").refcount == 1
        assert _text(service.retrieve("q", owner="B")) == "pytorch:q"

        service.release("A")  # idempotent: no second decrement

        assert registry.peek("pytorch", "BAAI/bge-m3").refcount == 1

        service.release("B")

        assert registry.peek("pytorch", "BAAI/bge-m3").refcount == 0

    def test_clear_releases_everything(self, service):
        service.load_engine(["pytorch"], {}, owner="A")
        service.load_engine(["opencv"], {}, owner="B")

        service.clear()

        assert service._sets == {}
        assert service._registry.stats()["count"] == 0
        assert not service.is_loaded()

    def test_needs_reload_is_per_owner(self, service):
        service.load_engine(["pytorch"], {"reranker_top_n": 3}, owner="A")

        assert not service.needs_reload(["pytorch"], {"reranker_top_n": 3}, owner="A")
        assert service.needs_reload(
            ["pytorch", "opencv"], {"reranker_top_n": 3}, owner="A"
        )
        # Owner B has nothing loaded, so its first turn must build a set.
        assert service.needs_reload(["pytorch"], {"reranker_top_n": 3}, owner="B")

    def test_is_loaded_any_owner_vs_specific_owner(self, service):
        assert not service.is_loaded()
        service.load_engine(["pytorch"], {}, owner="A")

        assert service.is_loaded()
        assert service.is_loaded("A")
        assert not service.is_loaded("B")

    def test_additional_index_paths_stay_per_owner(self, service):
        with patch(
            "tensortruth.services.rag_service.load_index_from_path",
            side_effect=lambda path, embed_model, k: (FakeRetriever("pdf"), Mock()),
        ) as loader:
            service.load_engine(["pytorch"], {}, ["/tmp/session.pdf"], owner="A")

        assert loader.call_count == 1
        assert service._sets["A"].additional_index_paths == ["/tmp/session.pdf"]
        # PDF indexes are not shared through the module registry.
        assert service._registry.peek("/tmp/session.pdf", "BAAI/bge-m3") is None
        assert len(service.retrieve("q", owner="A").source_nodes) == 2

        service.release("A")
        assert service._registry.stats()["count"] == 1
        assert service._registry.peek("pytorch", "BAAI/bge-m3").refcount == 0

    def test_sweep_idle_releases_only_stale_sets(self, service):
        service.load_engine(["pytorch"], {}, owner="old")
        service.load_engine(["opencv"], {}, owner="fresh")
        service._sets["old"].last_used = time.time() - 3600

        released = service.sweep_idle(60)

        assert released == ["old"]
        assert "old" not in service._sets
        assert "fresh" in service._sets
        assert _text(service.retrieve("q", owner="fresh")) == "opencv:q"

    def test_retrieve_without_set_returns_none_confidence(self, service):
        result = service.retrieve("q", owner="missing")
        assert result.confidence_level == "none"


@pytest.mark.unit
class TestToolLayerOwnership:
    def test_load_modules_tool_loads_only_its_session(self, service):
        from tensortruth.services.orchestrator_tool_wrappers import (
            create_load_modules_tool,
        )

        service.load_engine(["pytorch"], {}, owner="A")
        service.load_engine(["opencv"], {}, owner="B")

        catalog = [
            SimpleNamespace(name="pytorch", display_name="PyTorch", doc_count=10),
            SimpleNamespace(name="opencv", display_name="OpenCV", doc_count=20),
            SimpleNamespace(name="numpy", display_name="NumPy", doc_count=30),
        ]
        tool = create_load_modules_tool(
            service,
            progress_emitter=lambda *a, **k: None,
            module_catalog=catalog,
            loaded_modules=["pytorch"],
            session_params={},
            additional_index_paths=[],
            session_id="A",
            session_service=Mock(),
        )

        import asyncio

        output = asyncio.run(tool.acall(modules=["numpy"]))

        assert "Loaded: numpy" in str(output)
        assert service.get_loaded_modules("A") == ["pytorch", "numpy"]
        assert service.get_loaded_modules("B") == ["opencv"]
        assert _text(service.retrieve("q", owner="B")) == "opencv:q"

    def test_rag_tool_queries_the_owners_set(self, service):
        from tensortruth.services.orchestrator_tool_wrappers import create_rag_tool

        service.load_engine(["pytorch"], {}, owner="A")
        service.load_engine(["opencv"], {}, owner="B")

        tool = create_rag_tool(
            service,
            progress_emitter=lambda *a, **k: None,
            session_params={},
            session_messages=[],
            session_id="B",
        )

        import asyncio

        output = str(asyncio.run(tool.acall(query="tensor")))

        assert "opencv:tensor" in output
        assert "pytorch:tensor" not in output
