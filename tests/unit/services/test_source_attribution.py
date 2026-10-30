"""Cross-module source attribution: headers, prompt rule, per-module cap."""

from pathlib import Path

from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode

from tensortruth.api.deps import get_config_service
from tensortruth.api.main import app
from tensortruth.app_utils.config_schema import RAGConfig, TensorTruthConfig
from tensortruth.rag_engine import CUSTOM_CONTEXT_PROMPT_TEMPLATE
from tensortruth.services.config_service import ConfigService
from tensortruth.services.models import RAGRetrievalResult
from tensortruth.services.orchestrator_tool_wrappers import _format_retrieval_result
from tensortruth.services.rag_service import (
    _cap_sources_per_module,
    _source_module_id,
)


def _node(
    text: str,
    score: float,
    *,
    source_index: int = 0,
    title: str = "A Book",
    heading: str = "Some Section",
    doc_type: str = "book",
) -> NodeWithScore:
    node = TextNode(
        text=text,
        metadata={
            "display_name": title,
            "heading_path": heading,
            "doc_type": doc_type,
            "_source_index": source_index,
        },
    )
    return NodeWithScore(node=node, score=score)


def test_header_names_document_section_and_score():
    node = _node(
        "The decomposition splits expected error into bias and variance.",
        0.913,
        source_index=1,
        title="Dive into Deep Learning Ch.25",
        heading="The Bias-Variance Trade-off",
    )
    result = RAGRetrievalResult(
        source_nodes=[node],
        confidence_level="normal",
        condensed_query="bias variance decomposition",
        num_sources=1,
    )

    output = _format_retrieval_result(result, max_chars=4000)
    header = next(line for line in output.splitlines() if line.startswith("[s1]"))

    assert header == (
        "[s1] Dive into Deep Learning Ch.25 (book) "
        "· §The Bias-Variance Trade-off · score 0.913"
    )
    assert "Found 1 sources (confidence: normal)" in output
    assert "The decomposition splits expected error" in output


def test_header_index_is_stable_across_sources():
    nodes = [
        _node("first", 0.9, source_index=0, title="Bishop", heading="Chapter 7"),
        _node("second", 0.5, source_index=1, title="D2L", heading="Chapter 25"),
    ]
    result = RAGRetrievalResult(
        source_nodes=nodes,
        confidence_level="normal",
        condensed_query="q",
        num_sources=2,
    )

    output = _format_retrieval_result(result, max_chars=4000)

    assert "[s1] Bishop (book) · §Chapter 7 · score 0.900" in output
    assert "[s2] D2L (book) · §Chapter 25 · score 0.500" in output
    # Each header sits immediately above its own excerpt, not in a title list.
    assert output.index("[s1] Bishop") < output.index("first")
    assert output.index("[s2] D2L") < output.index("second")


def test_cap_keeps_one_per_module_and_backfills_from_under_represented():
    module_labels = ["module_a", "module_b"]
    nodes = [
        _node("a1", 0.91, source_index=0),
        _node("a2", 0.84, source_index=0),
        _node("a3", 0.77, source_index=0),
        _node("a4", 0.64, source_index=0),
        _node("b1", 0.48, source_index=1),
    ]

    capped = _cap_sources_per_module(
        nodes, max_per_module=3, limit=5, module_labels=module_labels
    )

    texts = [n.node.text for n in capped]
    assert texts == ["a1", "a2", "a3", "b1"]
    assert [n.score for n in capped] == sorted([n.score for n in capped], reverse=True)
    assert len(texts) == len(set(texts))  # no duplicate excerpts


def test_cap_backfills_beyond_top_n_cutoff():
    module_labels = ["module_a", "module_b"]
    nodes = [
        _node("a1", 0.95, source_index=0),
        _node("a2", 0.90, source_index=0),
        _node("a3", 0.85, source_index=0),
        _node("a4", 0.80, source_index=0),
        # Below a top_n=3 cutoff, but the only candidate from module B.
        _node("b1", 0.40, source_index=1),
    ]

    capped = _cap_sources_per_module(
        nodes, max_per_module=3, limit=3, module_labels=module_labels
    )

    assert [n.node.text for n in capped] == ["a1", "a2", "a3"]
    assert _source_module_id(capped[0], module_labels) == "module_a"

    # Limit 5 lets the backfill use the freed slot.
    capped5 = _cap_sources_per_module(
        nodes, max_per_module=3, limit=5, module_labels=module_labels
    )
    assert [n.node.text for n in capped5] == ["a1", "a2", "a3", "b1"]


def test_cap_zero_disables():
    nodes = [_node(f"a{i}", 0.9 - i / 10, source_index=0) for i in range(5)]
    nodes.append(_node("b1", 0.1, source_index=1))
    module_labels = ["module_a", "module_b"]

    assert (
        _cap_sources_per_module(
            nodes, max_per_module=0, limit=5, module_labels=module_labels
        )
        == nodes
    )


def test_module_identity_falls_back_to_document_metadata():
    node = _node("x", 0.5, source_index=7, title="Bishop", heading="Chapter 7")
    # Stale/out-of-range _source_index must not collapse every node to one id.
    assert _source_module_id(node) == "Bishop"


def test_prompt_contains_attribution_rule():
    prompt = " ".join(CUSTOM_CONTEXT_PROMPT_TEMPLATE.split()).lower()

    assert "cite only the document named in that excerpt's header" in prompt
    assert (
        "never attribute a passage to a document that is not named in its own header"
        in prompt
    )
    assert (
        "if two documents disagree" in prompt or "when two documents disagree" in prompt
    )


def test_config_defaults_include_module_cap():
    assert RAGConfig().max_sources_per_module == 3
    assert TensorTruthConfig.create_default().rag.max_sources_per_module == 3


def test_config_patch_then_get_round_trips_module_cap(tmp_path: Path):
    service = ConfigService(config_file=tmp_path / "config.yaml")
    app.dependency_overrides[get_config_service] = lambda: service
    try:
        client = TestClient(app)
        response = client.patch(
            "/api/config", json={"updates": {"rag_max_sources_per_module": 5}}
        )
        assert response.status_code == 200, response.text
        assert response.json()["rag"]["max_sources_per_module"] == 5

        response = client.get("/api/config")
        assert response.status_code == 200, response.text
        assert response.json()["rag"]["max_sources_per_module"] == 5
    finally:
        app.dependency_overrides.pop(get_config_service, None)
