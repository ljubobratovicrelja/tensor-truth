"""Tests for ChunkContextService and the rag_query chunk pointers.

The fixtures build the same two-level hierarchy indexing produces with
llama_index's HierarchicalNodeParser (document/section -> leaves) so the walks
are exercised against real node relationships, not hand-rolled dicts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

from tensortruth.agents.tool_output import extract_tool_text
from tensortruth.services.chunk_context import (
    ChunkContextService,
    build_for_session,
    format_char_size,
)
from tensortruth.services.models import RAGRetrievalResult
from tensortruth.services.orchestrator_tool_wrappers import (
    _format_retrieval_result,
    create_chunk_context_tools,
)

DOC1 = "doc1aaaabbbbccccddddeeeeffff0000"
DOC2 = "doc2aaaabbbbccccddddeeeeffff1111"
ROOT1 = "a1a1a1" + "0" * 26
ROOT2 = "b2b2b2" + "0" * 26
SEC_A = "c3c3c3" + "0" * 26
SEC_B = "d4d4d4" + "0" * 26
SEC_C = "e5e5e5" + "0" * 26
LEAF1 = "f6f6f6" + "0" * 26
LEAF2 = "f7f7f7" + "0" * 26
LEAF3 = "f8f8f8" + "0" * 26
LEAF4 = "f9f9f9" + "0" * 26
LEAF5 = "aa0aa0" + "0" * 26
LEAF6 = "ab1ab1" + "0" * 26


def _node(node_id, text, doc_id, *, heading_path=None, section_title=None):
    metadata = {"file_name": f"{doc_id[:4]}.md", "display_name": f"Doc {doc_id[:4]}"}
    if heading_path:
        metadata["heading_path"] = heading_path
    if section_title:
        metadata["section_title"] = section_title
    return TextNode(text=text, id_=node_id, metadata=metadata)


def _set_relationship(node, rel, other_id):
    node.relationships[rel] = RelatedNodeInfo(node_id=other_id)


def _append_child(parent, child):
    existing = list(parent.relationships.get(NodeRelationship.CHILD, []) or [])
    existing.append(RelatedNodeInfo(node_id=child.node_id))
    parent.relationships[NodeRelationship.CHILD] = existing
    _set_relationship(child, NodeRelationship.PARENT, parent.node_id)


def _link_siblings(nodes):
    for prev_node, next_node in zip(nodes, nodes[1:]):
        _set_relationship(prev_node, NodeRelationship.NEXT, next_node.node_id)
        _set_relationship(next_node, NodeRelationship.PREVIOUS, prev_node.node_id)


def _mark_source(node, doc_id):
    _set_relationship(node, NodeRelationship.SOURCE, doc_id)


class _DocstoreRetriever(BaseRetriever):
    """Stands in for AutoMergingRetriever: docstore reachable via _storage_context."""

    def __init__(self, storage_context):
        self._storage_context = storage_context
        super().__init__()

    def _retrieve(self, query_bundle):  # pragma: no cover - never called
        return []


@pytest.fixture()
def hierarchy():
    root1 = _node(ROOT1, "Title: Book One\n" + ("root body " * 1800), DOC1)
    sec_a = _node(
        SEC_A,
        "Section A: Introduction\n" + ("alpha body " * 60),
        DOC1,
        heading_path="Chapter 1 > Introduction",
        section_title="Introduction",
    )
    sec_b = _node(
        SEC_B,
        "Section B: Methods\n" + ("beta body " * 60),
        DOC1,
        heading_path="Chapter 1 > Methods",
        section_title="Methods",
    )
    leaf1 = _node(LEAF1, "Alpha leaf one.", DOC1)
    leaf2 = _node(LEAF2, "Alpha leaf two.", DOC1)
    leaf3 = _node(LEAF3, "Beta leaf one.", DOC1)
    leaf4 = _node(LEAF4, "Beta leaf two.", DOC1)

    root2 = _node(ROOT2, "Title: Book Two\n" + ("other root body " * 900), DOC2)
    sec_c = _node(
        SEC_C,
        "Section C: Other\n" + ("gamma body " * 60),
        DOC2,
        heading_path="Other > Section C",
        section_title="Other",
    )
    leaf5 = _node(LEAF5, "Gamma leaf one.", DOC2)
    leaf6 = _node(LEAF6, "Gamma leaf two.", DOC2)

    nodes = [
        root1,
        sec_a,
        sec_b,
        leaf1,
        leaf2,
        leaf3,
        leaf4,
        root2,
        sec_c,
        leaf5,
        leaf6,
    ]
    for node, doc_id in zip(nodes, [DOC1] * 7 + [DOC2] * 4):
        _mark_source(node, doc_id)

    _append_child(root1, sec_a)
    _append_child(root1, sec_b)
    _append_child(sec_a, leaf1)
    _append_child(sec_a, leaf2)
    _append_child(sec_b, leaf3)
    _append_child(sec_b, leaf4)
    _link_siblings([leaf1, leaf2, leaf3, leaf4])

    _append_child(root2, sec_c)
    _append_child(sec_c, leaf5)
    _append_child(sec_c, leaf6)
    _link_siblings([leaf5, leaf6])
    # Corruption seen in merged stores: the last chunk of doc 1 points at the
    # first chunk of doc 2. Expansion must not follow it.
    _set_relationship(leaf4, NodeRelationship.NEXT, leaf5.node_id)

    store = SimpleDocumentStore()
    store.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=store)
    return {
        "service": ChunkContextService([_DocstoreRetriever(storage_context)]),
        "store": store,
        "storage_context": storage_context,
    }


def test_resolve_exact_and_unique_prefix(hierarchy):
    service = hierarchy["service"]
    assert service.resolve(LEAF2).node_id == LEAF2
    assert service.resolve(LEAF2[:8]).node_id == LEAF2
    assert service.node_count == 11


def test_resolve_rejects_short_and_ambiguous_prefixes(hierarchy):
    service = hierarchy["service"]
    assert service.resolve(LEAF2[:4]) is None
    # Both doc roots share the "a1a1a1"/"b2b2b2" only up to 5 chars; force an
    # ambiguous six-char prefix by asking for one that matches two leaves.
    assert service.resolve("f") is None
    assert service.resolve("f6f6f6") == service.resolve(LEAF1)
    assert service.resolve("f6f6f") is None  # five chars, below the minimum
    assert service.resolve("") is None
    assert service.resolve("does-not-exist") is None


def test_resolve_ambiguous_prefix_returns_none(hierarchy):
    store = hierarchy["store"]
    store.add_documents(
        [_node("zzzzzz1", "dup a", DOC1), _node("zzzzzz2", "dup b", DOC1)]
    )
    service = hierarchy["service"]
    # Nodes added after construction are picked up by a refresh, which is what
    # the tools do on every call (they rebuild the service from the session).
    service.refresh()
    assert service.resolve("zzzzzz") is None
    assert service.resolve("zzzzzz1").node_id == "zzzzzz1"


def test_neighbours_walk_both_directions(hierarchy):
    service = hierarchy["service"]
    pairs = service.neighbours(LEAF3, before=1, after=1)
    assert [(label, node.node_id) for label, node in pairs] == [
        ("previous", LEAF2),
        ("next", LEAF4),
    ]


def test_neighbours_do_not_cross_document_boundary(hierarchy):
    service = hierarchy["service"]
    pairs = service.neighbours(LEAF4, before=3, after=3)
    ids = [node.node_id for _, node in pairs]
    assert LEAF5 not in ids
    assert ids == [LEAF1, LEAF2, LEAF3]


def test_section_picks_largest_ancestor_under_cap(hierarchy):
    service = hierarchy["service"]
    section, ancestors = service.section(LEAF1, max_chars=6000)
    assert section.node_id == SEC_A
    assert [a.node_id for a in ancestors] == [SEC_A, ROOT1]

    # With a cap below the section size only the leaf itself fits.
    section, ancestors = service.section(LEAF1, max_chars=100)
    assert section.node_id == LEAF1
    assert [a.node_id for a in ancestors] == [SEC_A]


def test_section_gives_up_when_chunk_exceeds_cap(hierarchy):
    service = hierarchy["service"]
    assert service.section(LEAF1, max_chars=5) == (None, [])
    assert service.section("nope") == (None, [])


def test_heading_path_prefers_metadata_then_ancestors(hierarchy):
    service = hierarchy["service"]
    leaf1 = service.resolve(LEAF1)
    assert service.heading_path(leaf1) == "Chapter 1 > Introduction"
    # Section node without explicit heading_path falls back to ancestor titles
    # and finally to the document display name.
    plain = _node("cd1cd1" + "0" * 26, "Unnamed heading\nbody", DOC1)
    assert service.heading_path(plain) == "Doc doc1"


def test_outline_lists_only_this_documents_sections(hierarchy):
    service = hierarchy["service"]
    entries = service.outline(LEAF2)
    assert [title for _, title in entries] == ["Introduction", "Methods"]
    assert [index for index, _ in entries] == [1, 2]
    assert [title for _, title in service.outline(LEAF5)] == ["Other"]


def test_unknown_retriever_shapes_are_skipped():
    service = ChunkContextService([object(), SimpleNamespace(retrievers=[object()])])
    assert service.docstore_count == 0
    assert service.resolve("anything") is None


def test_unwraps_composite_retrievers(hierarchy):
    from tensortruth.rag_engine import MultiIndexRetriever

    composite = MultiIndexRetriever(
        [_DocstoreRetriever(hierarchy["storage_context"])], enable_cache=False
    )
    service = ChunkContextService([composite])
    assert service.docstore_count == 1
    assert service.resolve(LEAF1).node_id == LEAF1


def test_build_for_session_reads_the_sessions_retriever(hierarchy):
    retriever = _DocstoreRetriever(hierarchy["storage_context"])
    rag_service = SimpleNamespace(_sets={"s1": SimpleNamespace(retriever=retriever)})

    built = build_for_session(rag_service, "s1")
    assert built is not None and built.resolve(LEAF1).node_id == LEAF1
    assert build_for_session(rag_service, "missing-session") is None
    assert build_for_session(SimpleNamespace(_sets={}), "s1") is None
    assert build_for_session(None) is None


def test_format_char_size():
    assert format_char_size(320) == "~320 chars"
    assert format_char_size(1400) == "~1.4k chars"


def test_format_retrieval_result_emits_chunk_pointers(hierarchy):
    service = hierarchy["service"]
    leaf1 = service.resolve(LEAF1)
    result = RAGRetrievalResult(
        source_nodes=[NodeWithScore(node=leaf1, score=0.8123)],
        confidence_level="high",
        condensed_query="alpha",
        num_sources=1,
    )

    output = _format_retrieval_result(result, max_chars=4000, context_service=service)

    assert "[s1] Doc doc1" in output
    assert "§Chapter 1 > Introduction" in output
    assert f"chunk node_id={LEAF1[:8]}" in output
    assert "section ~684 chars available" in output
    assert "score 0.812" in output
    # Without a service the excerpt budget behaviour is unchanged.
    lean = _format_retrieval_result(result, max_chars=4000)
    assert "chunk node_id=" not in lean
    assert "Alpha leaf one." in lean


def test_context_tools_return_actionable_text(hierarchy):
    service = hierarchy["service"]
    tools = {
        tool.metadata.name: tool
        for tool in create_chunk_context_tools(context_service=service)
    }
    assert set(tools) == {"expand_section", "read_neighbours", "document_outline"}

    expanded = extract_tool_text(tools["expand_section"](node_id=LEAF1[:8]))
    assert "Section: Chapter 1 > Introduction" in expanded
    assert "Section A: Introduction" in expanded
    assert LEAF1[:8] in expanded and LEAF2[:8] in expanded

    neighbours = extract_tool_text(
        tools["read_neighbours"](node_id=LEAF3[:8], before=1, after=1)
    )
    assert "--- previous: node_id=" in neighbours
    assert "Alpha leaf two." in neighbours and "Beta leaf two." in neighbours

    outline = extract_tool_text(tools["document_outline"](node_id=LEAF1[:8]))
    assert "1. Introduction" in outline and "2. Methods" in outline
    assert f"node_id={SEC_A[:8]}" in outline

    for name in tools:
        bad = extract_tool_text(tools[name](node_id="missingid"))
        assert "not found" in bad.lower()
