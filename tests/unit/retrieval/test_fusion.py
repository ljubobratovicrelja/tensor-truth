"""Unit tests for lexical retrieval and RRF fusion."""

import pytest
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from tensortruth.indexing.lexical import LexicalIndex
from tensortruth.retrieval.fusion import LexicalRetriever, RRFusionRetriever


class FakeRetriever(BaseRetriever):
    """Returns a fixed ranking, so fusion arithmetic is testable."""

    def __init__(self, node_ids, score=0.5, fail=False):
        self.node_ids = node_ids
        self.fixed_score = score
        self.fail = fail
        # Build nodes once so identity checks can tell which object was kept.
        self.nodes = [
            TextNode(text=f"text {node_id}", id_=node_id) for node_id in node_ids
        ]
        super().__init__()

    def _retrieve(self, query_bundle):
        if self.fail:
            raise RuntimeError("backend is down")
        return [NodeWithScore(node=node, score=self.fixed_score) for node in self.nodes]


@pytest.mark.unit
class TestRRFusionRetriever:
    def test_overlapping_hit_wins(self):
        first = FakeRetriever(["A", "B"])
        second = FakeRetriever(["B", "C"])

        fused = RRFusionRetriever([first, second], rrf_k=60)._retrieve(
            QueryBundle(query_str="q")
        )

        ids = [item.node.node_id for item in fused]
        assert ids == ["B", "A", "C"]
        by_id = {item.node.node_id: item.score for item in fused}
        assert by_id["B"] == pytest.approx(1 / 62 + 1 / 61)
        assert by_id["A"] == pytest.approx(1 / 61)
        assert by_id["C"] == pytest.approx(1 / 62)

    def test_duplicate_keeps_highest_scoring_node_object(self):
        high = FakeRetriever(["X"], score=0.9)
        low = FakeRetriever(["X"], score=0.1)

        fused = RRFusionRetriever([high, low])._retrieve(QueryBundle(query_str="q"))

        assert len(fused) == 1
        assert fused[0].score == pytest.approx(2 / 61)
        # Same node object, not a copy of the low-score one.
        assert fused[0].node is high.nodes[0]

    def test_weights_shift_the_ranking(self):
        first = FakeRetriever(["A"])
        second = FakeRetriever(["B"])

        fused = RRFusionRetriever([first, second], weights=[1.0, 3.0])._retrieve(
            QueryBundle(query_str="q")
        )

        assert [item.node.node_id for item in fused] == ["B", "A"]

    def test_failing_child_does_not_kill_fusion(self):
        good = FakeRetriever(["A"])
        broken = FakeRetriever([], fail=True)

        fused = RRFusionRetriever([broken, good])._retrieve(QueryBundle(query_str="q"))

        assert [item.node.node_id for item in fused] == ["A"]

    def test_every_child_failing_returns_empty(self):
        broken = FakeRetriever([], fail=True)
        assert RRFusionRetriever([broken])._retrieve(QueryBundle(query_str="q")) == []

    def test_weight_count_must_match_retrievers(self):
        with pytest.raises(ValueError):
            RRFusionRetriever([FakeRetriever(["A"])], weights=[1.0, 2.0])


@pytest.mark.unit
class TestLexicalRetriever:
    @pytest.fixture()
    def lexical(self, tmp_path):
        index = LexicalIndex(tmp_path)
        index.build(
            [
                ("c1", "Stochastic gradient descent with minibatches.", "doc-1"),
                ("c2", "Neural networks stack affine layers.", "doc-2"),
            ],
            embedding_model="bge-m3",
        )
        return index

    def test_synthesises_nodes_with_stored_text(self, lexical):
        retriever = LexicalRetriever(lexical=lexical, top_k=2)

        results = retriever.retrieve(QueryBundle(query_str="gradient descent"))

        assert [item.node.node_id for item in results] == ["c1"]
        assert "minibatches" in results[0].node.get_content()
        assert results[0].node.metadata["doc_id"] == "doc-1"
        assert 0.0 < results[0].score <= 1.0

    def test_node_lookup_prefers_docstore_node(self, lexical):
        docstore_node = TextNode(text="canonical text", id_="c1")
        retriever = LexicalRetriever(
            lexical=lexical,
            top_k=2,
            node_lookup=lambda node_id: docstore_node,
        )

        results = retriever.retrieve(QueryBundle(query_str="gradient descent"))

        assert results[0].node is docstore_node

    def test_missing_node_falls_back_to_index_text(self, lexical):
        def broken_lookup(node_id):
            raise ValueError("not in docstore")

        retriever = LexicalRetriever(
            lexical=lexical, top_k=2, node_lookup=broken_lookup
        )

        results = retriever.retrieve(QueryBundle(query_str="gradient descent"))

        assert "minibatches" in results[0].node.get_content()

    def test_empty_index_returns_no_nodes(self, tmp_path):
        retriever = LexicalRetriever(lexical=LexicalIndex(tmp_path), top_k=2)
        assert retriever.retrieve(QueryBundle(query_str="anything")) == []
