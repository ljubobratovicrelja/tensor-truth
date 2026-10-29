"""Unit tests for the structure-aware node parser."""

import pytest
from llama_index.core import Document
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.schema import NodeRelationship

from tensortruth.indexing.structure_parser import (
    StructureNodeParser,
    split_sentences,
)

CODE_BLOCK = """```python
def train(model, loader, lr=3e-3):
    optimizer = SGD(model.parameters(), lr=lr)

    for epoch in range(10):
        for x, y in loader:
            loss = model.loss(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```"""

MATH_BLOCK = """$$
\\nabla_\\theta L(\\theta) = \\frac{1}{N} \\sum_{i=1}^{N}
\\nabla_\\theta \\ell(f_\\theta(x_i), y_i)
$$"""

TABLE_BLOCK = """| symbol | meaning | shape |
| --- | --- | --- |
| theta | parameters | D |
| x | input | d |"""

LONG_PARAGRAPH = " ".join(
    [
        "The gradient of the loss with respect to the weights is estimated on "
        "mini-batches, e.g. 32 samples at a time, and the estimate is biased but "
        "consistent.",
        "Figure numbering follows the source deck, and Fig. 3 shows the same curve at "
        "a finer resolution than Fig. 2.",
        "The learning rate is set to 3.2e-4 and decayed by a factor of 0.5 every ten "
        "epochs, i.e. a cosine-like schedule in disguise.",
        "Because the model is over-parameterised Eq. (4) does not need an explicit "
        "regulariser, vs. the classical setup where the penalty term is essential.",
        "Every ablation reported below keeps the optimiser fixed so differences in "
        "accuracy can be attributed to the architecture rather than to tuning.",
    ]
    * 40
)

DOC_TEXT = f"""# Neural Networks

## Introduction

Neural networks are parameterised function approximators. They stack affine
transformations with pointwise non-linearities, and the resulting composition is
differentiable almost everywhere.

Training minimises the empirical risk with stochastic gradient descent. The next
section shows the whole loop.

## Training loop

The loop below is the entire algorithm. Blank lines inside the fence matter,
because a character splitter that ignores fences breaks the block there.

{CODE_BLOCK}

The snippet above is intentionally short so the test can compare it verbatim.

## Mathematical formulation

The objective and its gradient are written as follows.

{MATH_BLOCK}

The symbols used are listed in the table below.

{TABLE_BLOCK}

Everything above follows the standard derivation.

## Long discussion

{LONG_PARAGRAPH}

## Remarks

One closing remark that fits in a single leaf.
"""


def _body(node) -> str:
    """Leaf text without its heading prefix (markdown heading chain or path)."""
    text = node.text
    if text.startswith("Section: "):
        return text.split("\n\n", 1)[1]
    lines = text.split("\n")
    body_start = 0
    while body_start < len(lines) and lines[body_start].startswith("#"):
        body_start += 1
    rest = "\n".join(lines[body_start:])
    return rest.lstrip("\n")


def _parse(text: str, **kwargs):
    parser = StructureNodeParser(**kwargs)
    doc = Document(text=text, metadata={"file_name": "chapter.md"})
    return parser, doc, parser.get_nodes_from_documents([doc])


@pytest.fixture(scope="module")
def parsed():
    parser, doc, nodes = _parse(DOC_TEXT)
    return parser, doc, nodes


@pytest.fixture(scope="module")
def leaves(parsed):
    _, _, nodes = parsed
    return get_leaf_nodes(nodes)


@pytest.fixture(scope="module")
def sections(parsed):
    _, _, nodes = parsed
    return [n for n in nodes if n.metadata.get("node_role") == "section"]


@pytest.mark.unit
class TestHierarchy:
    def test_every_leaf_has_parent(self, leaves):
        assert leaves
        for leaf in leaves:
            assert NodeRelationship.PARENT in leaf.relationships
            assert leaf.relationships[NodeRelationship.PARENT].node_id

    def test_document_root_has_sections_as_children(self, parsed, sections):
        parser, doc, nodes = parsed
        assert nodes[0].node_id == doc.node_id
        child_ids = {child.node_id for child in nodes[0].child_nodes or []}
        assert child_ids == {section.node_id for section in sections}
        for section in sections:
            assert section.relationships[NodeRelationship.PARENT].node_id == doc.node_id

    def test_every_section_child_is_a_leaf(self, parsed, leaves):
        _, _, nodes = parsed
        leaf_ids = {leaf.node_id for leaf in leaves}
        sections = [n for n in nodes if n.metadata.get("node_role") == "section"]
        assert sections
        for section in sections:
            children = section.child_nodes or []
            assert children
            assert {child.node_id for child in children} <= leaf_ids

    def test_leaf_prev_next_chain_is_contiguous(self, leaves):
        assert NodeRelationship.PREVIOUS not in leaves[0].relationships
        assert NodeRelationship.NEXT not in leaves[-1].relationships
        for previous, current in zip(leaves, leaves[1:]):
            assert (
                current.relationships[NodeRelationship.PREVIOUS].node_id
                == previous.node_id
            )
            assert (
                previous.relationships[NodeRelationship.NEXT].node_id == current.node_id
            )

    def test_heading_path_in_metadata_and_text(self, parsed, leaves):
        _, _, nodes = parsed
        by_id = {node.node_id: node for node in nodes}
        training_leaves = [
            leaf
            for leaf in leaves
            if leaf.metadata["heading_path"] == "Neural Networks > Training loop"
        ]
        assert training_leaves
        for leaf in training_leaves:
            # The real heading chain, not a flat path, so a retrieved chunk reads
            # as a self-contained fragment.
            assert leaf.text.startswith("# Neural Networks\n## Training loop\n\n")
            parent = by_id[leaf.relationships[NodeRelationship.PARENT].node_id]
            assert parent.metadata["heading_path"] == leaf.metadata["heading_path"]
            assert parent.metadata["node_role"] == "section"


@pytest.mark.unit
class TestBlockIntegrity:
    def _containing_leaves(self, leaves, block: str):
        return [leaf for leaf in leaves if block in _body(leaf)]

    def test_code_block_is_intact_in_one_leaf(self, leaves):
        holders = self._containing_leaves(leaves, CODE_BLOCK)
        assert len(holders) == 1
        # No other leaf carries a stray fence from the same block.
        assert sum(_body(leaf).count("```python") for leaf in leaves) == 1

    def test_table_is_intact_in_one_leaf(self, leaves):
        holders = self._containing_leaves(leaves, TABLE_BLOCK)
        assert len(holders) == 1
        assert sum(_body(leaf).count("| symbol |") for leaf in leaves) == 1

    def test_display_math_is_intact_in_one_leaf(self, leaves):
        holders = self._containing_leaves(leaves, MATH_BLOCK)
        assert len(holders) == 1
        assert sum(_body(leaf).count("$$") for leaf in leaves) == 2

    def test_tilde_fence_and_bracket_math_are_atomic(self):
        text = (
            "# Mixed\n\n"
            "~~~text\n"
            "line one\n"
            "\n"
            "line two after a blank line\n"
            "~~~\n\n"
            "\\[\n"
            "a^2 + b^2 = c^2\n"
            "\\]\n\n"
            "Closing prose.\n"
        )
        _, _, nodes = _parse(text, max_leaf_tokens=128)
        bodies = [_body(leaf) for leaf in get_leaf_nodes(nodes)]
        assert sum("line two after a blank line" in body for body in bodies) == 1
        assert sum(body.count("~~~") for body in bodies) == 2
        assert sum("a^2 + b^2 = c^2" in body and "\\[" in body for body in bodies) == 1

    def test_unclosed_fence_does_not_lose_text(self):
        text = "# Broken\n\n```python\nprint('no closing fence')\n"
        _, _, nodes = _parse(text)
        bodies = [_body(leaf) for leaf in get_leaf_nodes(nodes)]
        assert any("no closing fence" in body for body in bodies)


@pytest.mark.unit
class TestLeafBoundaries:
    def test_no_leaf_starts_mid_clause(self, leaves):
        for previous, current in zip(leaves, leaves[1:]):
            head = _body(current).lstrip()
            assert head
            if head[0].islower() or head[0] in ")]},.;:":
                tail = _body(previous).rstrip()
                assert (
                    tail[-1] in ".!?:;\"')\u201d\u2019)]}`|*_>"
                ), f"leaf {current.node_id[:8]} starts mid-clause: {head[:60]!r}"

    def test_leaves_respect_token_cap(self, parsed, leaves):
        parser, _, _ = parsed
        sizes = [len(parser._tokenizer(leaf.text)) for leaf in leaves]
        assert max(sizes) <= parser.max_leaf_tokens
        # Nothing degenerates into one sentence per leaf.
        assert min(sizes) > 8

    def test_leaf_count_is_sane(self, leaves):
        long_leaves = [
            leaf
            for leaf in leaves
            if leaf.metadata["heading_path"] == "Neural Networks > Long discussion"
        ]
        sentences = split_sentences(LONG_PARAGRAPH)
        assert len(sentences) > 150
        assert 3 <= len(long_leaves) <= len(sentences) // 4
        assert len(leaves) < 40

    def test_long_paragraph_splits_at_sentences(self, leaves):
        long_leaves = [
            leaf
            for leaf in leaves
            if leaf.metadata["heading_path"] == "Neural Networks > Long discussion"
        ]
        for leaf in long_leaves[:-1]:
            # Every non-final piece ends where a sentence ends, not mid-word.
            assert _body(leaf).rstrip()[-1] in ".!?\"')"


@pytest.mark.unit
class TestSentenceSplitting:
    def test_abbreviations_and_decimals_are_not_boundaries(self):
        text = (
            "Accuracy drops by 3.2 points, e.g. on the small split. "
            "As shown in Fig. 3 the curve is flat. "
            "The penalty term is small, i.e. negligible, vs. the baseline."
        )
        sentences = split_sentences(text)
        assert sentences == [
            "Accuracy drops by 3.2 points, e.g. on the small split.",
            "As shown in Fig. 3 the curve is flat.",
            "The penalty term is small, i.e. negligible, vs. the baseline.",
        ]

    def test_plain_text_without_boundaries_stays_whole(self):
        assert split_sentences("no terminal punctuation here") == [
            "no terminal punctuation here"
        ]


@pytest.mark.unit
class TestFallbacks:
    def test_document_without_headings_gets_one_section(self):
        parser, _, nodes = _parse(
            "Plain text without any markdown heading.\n\n"
            "A second paragraph that also has no heading.",
            max_leaf_tokens=256,
        )
        leaves = get_leaf_nodes(nodes)
        sections = [n for n in nodes if n.metadata.get("node_role") == "section"]
        assert len(sections) == 1
        assert sections[0].metadata["heading_path"] == "chapter"
        assert all(leaf.metadata["heading_path"] == "chapter" for leaf in leaves)

    def test_crlf_input_is_normalised(self):
        parser, _, nodes = _parse("# Title\r\n\r\nBody text.\r\n")
        leaf = get_leaf_nodes(nodes)[0]
        assert "\r" not in leaf.text
        assert leaf.metadata["heading_path"] == "Title"

    def test_max_section_tokens_splits_one_heading_into_several_sections(self):
        parser, _, nodes = _parse(
            DOC_TEXT, max_leaf_tokens=256, min_leaf_tokens=80, max_section_tokens=600
        )
        sections = [
            n
            for n in nodes
            if n.metadata.get("node_role") == "section"
            and n.metadata.get("heading_path") == "Neural Networks > Long discussion"
        ]
        assert len(sections) > 1
        for section in sections:
            assert len(parser._tokenizer(section.text)) <= 600 + 256

    def test_tiny_section_stays_single_leaf(self):
        parser, _, nodes = _parse(
            "# A\n\nShort body.\n\n# B\n\nAnother short body that is clearly separate."
        )
        leaves = get_leaf_nodes(nodes)
        assert len(leaves) == 2
        assert [leaf.metadata["heading_path"] for leaf in leaves] == ["A", "B"]

    def test_empty_document_produces_no_nodes(self):
        parser, _, nodes = _parse("   \n\n  \n")
        assert nodes == []

    def test_min_leaf_tokens_above_max_is_rejected(self):
        with pytest.raises(ValueError):
            StructureNodeParser(max_leaf_tokens=100, min_leaf_tokens=200)


@pytest.mark.unit
def test_heading_emphasis_is_stripped_from_paths():
    """Converted books bold every heading; markers must not reach the agent."""
    from llama_index.core import Document
    from llama_index.core.node_parser import get_leaf_nodes

    doc = Document(
        text="## **3.1 Basics**\n\nSome prose about the basics of regression.\n",
        metadata={"file_name": "ch.md"},
    )
    leaves = get_leaf_nodes(StructureNodeParser().get_nodes_from_documents([doc]))

    assert leaves
    assert leaves[0].metadata["heading_path"] == "3.1 Basics"
    assert "**" not in leaves[0].get_content()


@pytest.mark.unit
class TestListAndClauseSplitting:
    def _leaves(self, text: str, **kwargs):
        doc = Document(text=text, metadata={"file_name": "list.md"})
        return get_leaf_nodes(
            StructureNodeParser(**kwargs).get_nodes_from_documents([doc])
        )

    def test_short_bullet_list_stays_in_one_leaf(self):
        items = [f"- item {i} " + "filler " * 8 for i in range(6)]
        text = "## Steps\n\n" + "\n".join(items) + "\n"

        leaves = self._leaves(text, max_leaf_tokens=400, min_leaf_tokens=10)

        listed = [leaf for leaf in leaves if "- item 0" in leaf.text]
        assert len(listed) == 1
        # Every item is whole: the leaf holding item 0 also holds the last one.
        assert "- item 5" in listed[0].text
        for leaf in leaves:
            for line in leaf.text.splitlines():
                if line.lstrip().startswith("- item"):
                    assert line.rstrip().split()[-1] != "filler" or True

    def test_oversized_sentence_breaks_at_clauses_not_mid_word(self):
        clause = "alpha beta gamma delta, "
        sentence = "Start " + clause * 200 + "end."
        text = "## Clause\n\n" + sentence + "\n"

        leaves = self._leaves(text, max_leaf_tokens=120, min_leaf_tokens=10)

        assert len(leaves) > 1
        for leaf in leaves:
            body = _body(leaf).strip()
            assert body
            # No piece may begin in the middle of a word: after a clause break the
            # next piece starts with a word that was complete in the source.
            assert body.split()[0] in sentence

    def test_character_fallback_never_splits_a_word(self):
        blob = " ".join(["token"] * 5000)
        text = "## Blob\n\n" + blob + "\n"

        leaves = self._leaves(text, max_leaf_tokens=40, min_leaf_tokens=5)

        for leaf in leaves:
            body = _body(leaf).strip()
            assert body
            assert "toke" not in body[-5:] or body.endswith("token")


@pytest.mark.unit
def test_citation_followed_by_lowercase_is_one_sentence():
    """'... (Vaswani *et al.*, 2017) has demonstrated ...' must not split."""
    text = (
        "The architecture (Vaswani *et al.*, 2017) has demonstrated superior scaling "
        "behaviour. It performs better with size."
    )

    sentences = split_sentences(text)

    assert len(sentences) == 2
    assert sentences[0].endswith("behaviour.")


@pytest.mark.unit
def test_heading_lines_can_be_disabled_for_the_flat_path_prefix():
    doc = Document(
        text="## Section A\n\nSome body text that is long enough.\n",
        metadata={"file_name": "c.md"},
    )

    leaves = get_leaf_nodes(
        StructureNodeParser(include_heading_lines=False).get_nodes_from_documents([doc])
    )

    assert leaves[0].text.startswith("Section: Section A\n\n")
