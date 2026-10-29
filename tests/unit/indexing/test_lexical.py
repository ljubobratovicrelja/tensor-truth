"""Unit tests for the FTS5 lexical index."""

import pytest
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from tensortruth.indexing.lexical import (
    LEXICAL_DB_FILENAME,
    LexicalIndex,
    build_match_query,
)

CHUNKS = [
    (
        "chunk-descent",
        "Stochastic gradient descent updates the weights with a minibatch estimate "
        "of the gradient.",
        "doc-nn",
    ),
    (
        "chunk-layers",
        "Neural networks stack affine layers with elementwise nonlinearities.",
        "doc-nn",
    ),
    (
        "chunk-algebra",
        "Linear algebra supplies the notation for vectors and matrices.",
        "doc-math",
    ),
]


@pytest.fixture()
def lexical(tmp_path):
    index = LexicalIndex(tmp_path)
    index.build(CHUNKS, embedding_model="bge-m3")
    return index


@pytest.mark.unit
class TestLexicalIndexBuild:
    def test_db_lives_beside_the_vector_store(self, tmp_path, lexical):
        assert lexical.db_path == tmp_path / LEXICAL_DB_FILENAME
        assert lexical.exists()
        assert lexical.count() == 3

    def test_meta_records_build_context(self, lexical):
        import sqlite3

        conn = sqlite3.connect(str(lexical.db_path))
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        finally:
            conn.close()
        assert meta["embedding_model"] == "bge-m3"
        assert meta["built_at"]

    def test_rebuild_is_idempotent(self, tmp_path, lexical):
        lexical.build(CHUNKS, embedding_model="bge-m3")
        assert lexical.count() == 3

    def test_missing_db_degrades_to_empty(self, tmp_path):
        index = LexicalIndex(tmp_path / "nope")
        assert index.exists() is False
        assert index.count() == 0
        assert index.search("gradient descent") == []
        assert index.get_chunk("anything") is None


@pytest.mark.unit
class TestLexicalIndexSearch:
    def test_phrase_match_ranks_first(self, lexical):
        results = lexical.search("stochastic gradient descent", top_k=3)

        assert results
        assert results[0][0] == "chunk-descent"
        assert 0.0 < results[0][1] <= 1.0
        # Scores must be ordered, best first.
        assert results == sorted(results, key=lambda item: item[1], reverse=True)

    def test_token_match_works_when_phrase_absent(self, lexical):
        results = lexical.search("matrices vectors algebra", top_k=3)

        assert results
        assert results[0][0] == "chunk-algebra"

    def test_nonsense_query_returns_empty(self, lexical):
        assert lexical.search("quantum chromodynamics zzzznotaterm") == []

    def test_empty_query_returns_empty(self, lexical):
        assert lexical.search("") == []
        assert lexical.search("   ") == []

    def test_fts5_operators_are_stripped(self, lexical):
        # Must not raise sqlite3.OperationalError on FTS5 syntax.
        assert build_match_query('"neural*" AND (foo:bar)^ -baz') is not None
        results = lexical.search('"neural*" AND (networks)', top_k=3)
        assert results
        assert results[0][0] == "chunk-layers"

    def test_stopword_only_query_returns_empty(self, lexical):
        assert build_match_query("the and of to") is None
        assert lexical.search("the and of") == []

    def test_terms_are_or_ed_not_and_ed(self, lexical):
        # A natural-language question must not require every token in one chunk.
        expr = build_match_query("momentum changes the update direction")
        assert expr is not None
        assert " AND " not in expr
        results = lexical.search("matrices nonlinearities", top_k=3)
        assert {node_id for node_id, _ in results} == {
            "chunk-algebra",
            "chunk-layers",
        }

    def test_top_k_is_respected(self, tmp_path):
        index = LexicalIndex(tmp_path)
        index.build(
            [
                ("d1", "Regularisation shrinks weights in a linear model.", "doc-1"),
                ("d2", "Regularisation by dropout trains a linear model.", "doc-2"),
                ("d3", "A linear model with no regularisation overfits.", "doc-3"),
            ]
        )

        assert len(index.search("regularisation linear model", top_k=1)) == 1
        assert len(index.search("regularisation linear model", top_k=3)) == 3


@pytest.mark.unit
class TestBuildFromStore:
    def test_only_leaves_are_indexed(self, tmp_path):
        parent = TextNode(text="Parent section text about descent methods.", id_="p1")
        child_a = TextNode(text="Child one repeats the descent discussion.", id_="c1")
        child_b = TextNode(text="Child two continues the descent discussion.", id_="c2")
        for child in (child_a, child_b):
            child.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id="p1")
        parent.relationships[NodeRelationship.CHILD] = [
            RelatedNodeInfo(node_id="c1"),
            RelatedNodeInfo(node_id="c2"),
        ]
        nodes = [parent, child_a, child_b]
        assert len(get_leaf_nodes(nodes)) == 2

        index = LexicalIndex.build_from_store(tmp_path, nodes, embedding_model="bge-m3")

        assert index.count() == 2
        assert index.get_chunk("p1") is None
        assert index.get_chunk("c2") is not None

    def test_doc_id_comes_from_ref_doc_id(self, tmp_path):
        node = TextNode(text="A chunk that mentions backpropagation.", id_="c1")
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="doc-42")

        index = LexicalIndex.build_from_store(tmp_path, [node])

        text, doc_id = index.get_chunk("c1")
        assert doc_id == "doc-42"
        assert "backpropagation" in text
