"""Structure-aware node parsing for TensorTruth indexes.

Why this module exists
----------------------
The previous pipeline parsed documents with ``HierarchicalNodeParser`` over fixed
token windows (2048/512/256). Those windows know nothing about document structure:
they cut sentences, display equations, tables and fenced code in half. Dense
retrieval then returned shards that start mid-clause ("g. the agent must ...") and
contain half an equation, which the answering model cannot quote or reason about.
Chunks also had no section identity, so a retrieved excerpt could not be placed in
the document it came from.

This parser segments the document along its real structure instead:

* markdown headings define sections (``Document -> section -> leaf``),
* fenced code, display math and tables are atomic blocks,
* leaves are packed from whole paragraphs and split at sentence boundaries only
  when a paragraph does not fit,
* every leaf carries ``metadata["heading_path"]`` (``"A > B > C"``) so retrieval
  tools can tell the agent which section a chunk belongs to.

The output is a flat list with PARENT/CHILD relationships between levels and
PREVIOUS/NEXT between adjacent leaves of the same document, exactly like
``HierarchicalNodeParser``, so ``get_leaf_nodes`` and ``AutoMergingRetriever``
keep working unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import (
    BaseNode,
    Document,
    NodeRelationship,
    TextNode,
)
from llama_index.core.utils import get_tokenizer

logger = logging.getLogger(__name__)

# Blocks that must never be cut at an arbitrary character. A code fence that loses
# its closing line, an equation split between "$$", or a table without a header row
# are unusable even when the retrieved text is otherwise correct.
ATOMIC_BLOCK_KINDS = frozenset({"code", "math", "table"})

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([\w.+-]*)[ \t]*$")
_RULE_RE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$")
_LIST_MARKER_RE = re.compile(r"^[ \t]{0,3}(?:[-*+\u2022]|\d{1,3}[.)])[ \t]+")


def _clean_heading(text: str) -> str:
    """Strip markdown emphasis from a heading title.

    Converted books often bold every heading ("**3.1.1 Basics**"). The markers
    carry no meaning for retrieval, and they used to leak into the leaf text
    prefix and into agent-facing section paths.
    """
    cleaned = re.sub(r"[*`]+", "", text)
    cleaned = cleaned.replace("_", " ")
    return " ".join(cleaned.split())


# A sentence ends on terminal punctuation (plus closing quotes/brackets) followed by
# whitespace. Requiring the whitespace is what keeps "3.2" and "e.g.x" in one piece.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\u2026]+[\)\]\"'\u201d\u2019]*")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")

# Abbreviations whose trailing period is not a sentence end. Without this list the
# sentence splitter produces fragments like "as described in Fig." + "3, the ...".
_ABBREVIATIONS = frozenset(
    {
        "e.g.",
        "i.e.",
        "fig.",
        "figs.",
        "eq.",
        "eqs.",
        "vs.",
        "etc.",
        "al.",
        "cf.",
        "sec.",
        "secs.",
        "ch.",
        "chap.",
        "no.",
        "nos.",
        "approx.",
        "resp.",
        "ref.",
        "refs.",
        "dr.",
        "prof.",
        "mr.",
        "mrs.",
        "ms.",
        "st.",
        "ed.",
        "eds.",
        "vol.",
        "pp.",
        "p.",
        "thm.",
        "def.",
        "prop.",
        "ex.",
        "inc.",
        "ltd.",
        "max.",
        "min.",
        "avg.",
        "std.",
        "viz.",
    }
)

_TERMINAL_CHARS = set(".!?:\u2026\"'\u201d\u2019)]}`|*_>")


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences without cutting abbreviations or decimals.

    Returns the original text as a single element when no safe boundary is found.
    """
    if not text:
        return []

    starts = [0]
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end()
        next_pos = end
        while next_pos < len(text) and text[next_pos] in " \t\n":
            next_pos += 1
        if next_pos >= len(text):
            break
        # No whitespace after the punctuation: "3.2", "0.5x", "e.g.x" - not a boundary.
        if next_pos == end:
            continue
        # A sentence cannot start with a lowercase letter or a closing mark.
        # Without this, a citation such as "(Vaswani *et al.*, 2017) has
        # demonstrated..." was cut after the bracket, mid-sentence.
        following = text[next_pos]
        if following.islower() or following in ",;:)":
            continue
        word_match = _WORD_BEFORE_RE.search(text, 0, match.start())
        if word_match is not None:
            word = word_match.group(1)
            # The regex stops before the punctuation, so rebuild "Fig." from "Fig".
            if f"{word.lower()}." in _ABBREVIATIONS:
                continue
            # Single capital initials: "J. Smith".
            if len(word) == 1 and word.isupper():
                continue
            # "2026. 27" style enumerations: a bare number followed by another number.
            if word.isdigit() and text[next_pos].isdigit():
                continue
        starts.append(next_pos)

    if len(starts) == 1:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: List[str] = []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:stop].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


@dataclass
class _Block:
    """A structural unit of the document: one heading, paragraph or atomic block."""

    kind: str  # "heading" | "paragraph" | "code" | "math" | "table"
    text: str
    level: int = 0


@dataclass
class _Section:
    title: str
    level: int
    heading_path: str
    blocks: List[_Block] = field(default_factory=list)
    # Ancestor heading chain, outermost first, as (level, title) pairs. Kept so a
    # leaf can carry the real markdown headings instead of only a path string.
    heading_lines: List[tuple[int, str]] = field(default_factory=list)

    @property
    def heading_line(self) -> str:
        if not self.title:
            return ""
        return f"{'#' * max(1, self.level)} {self.title}"


class StructureNodeParser(NodeParser):
    """Parse markdown/plain text into a Document -> section -> leaf hierarchy.

    Unlike ``HierarchicalNodeParser`` the split points come from the document's own
    structure (headings, blank lines, sentence ends), so a leaf never starts in the
    middle of a clause and code/math/table blocks stay whole. Leaf metadata carries
    ``heading_path``; when ``prepend_heading_path`` is set the same path is prefixed
    to the leaf text so the embedding itself is section-aware.
    """

    max_leaf_tokens: int = Field(
        default=512,
        gt=0,
        description="Hard upper bound (tokens) for the body of a single leaf.",
    )
    min_leaf_tokens: int = Field(
        default=120,
        ge=0,
        description=(
            "Soft lower bound; a short trailing leaf is merged back into its "
            "predecessor when the section allows it."
        ),
    )
    prepend_heading_path: bool = Field(
        default=True,
        description="Prefix leaf text with its section context (see include_heading_lines).",
    )
    include_heading_lines: bool = Field(
        default=True,
        description=(
            "Prefix each leaf with the section's markdown heading chain "
            "('# A', '## B', ...) instead of a flat 'Section: A > B' line, so a "
            "retrieved chunk reads as a self-contained document fragment."
        ),
    )
    max_section_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "When set, a heading whose body exceeds this many tokens is split into "
            "several section nodes sharing one heading_path."
        ),
    )

    _tokenizer: Callable = PrivateAttr()

    def __init__(
        self,
        max_leaf_tokens: int = 512,
        min_leaf_tokens: int = 120,
        tokenizer: Optional[Callable] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        prepend_heading_path: bool = True,
        include_heading_lines: bool = True,
        max_section_tokens: Optional[int] = None,
        callback_manager=None,
    ) -> None:
        if min_leaf_tokens > max_leaf_tokens:
            raise ValueError(
                f"min_leaf_tokens ({min_leaf_tokens}) must not exceed "
                f"max_leaf_tokens ({max_leaf_tokens})."
            )
        if not include_metadata:
            raise ValueError(
                "StructureNodeParser requires include_metadata=True: leaves carry "
                "heading_path in metadata."
            )
        super().__init__(
            max_leaf_tokens=max_leaf_tokens,
            min_leaf_tokens=min_leaf_tokens,
            prepend_heading_path=prepend_heading_path,
            include_heading_lines=include_heading_lines,
            max_section_tokens=max_section_tokens,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            **({"callback_manager": callback_manager} if callback_manager else {}),
        )
        self._tokenizer = tokenizer or get_tokenizer()

    @classmethod
    def class_name(cls) -> str:
        return "StructureNodeParser"

    # ------------------------------------------------------------------ public API

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs,
    ) -> List[BaseNode]:
        """Unused: ``get_nodes_from_documents`` owns the hierarchy construction."""
        return list(nodes)

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs,
    ) -> List[BaseNode]:
        """Return document, section and leaf nodes in document order."""
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: List[BaseNode] = []
            for doc in documents:
                all_nodes.extend(self._parse_document(doc))
            event.on_end(payload={EventPayload.NODES: all_nodes})
        return all_nodes

    # ------------------------------------------------------------- document parsing

    def _parse_document(self, doc: Document) -> List[BaseNode]:
        text = (doc.get_content() or "").replace("\r\n", "\n").replace("\r", "\n")

        sections = self._build_sections(text, self._document_title(doc))
        doc_meta = dict(doc.metadata or {})
        doc_meta["node_role"] = "document"
        doc_node: BaseNode = doc
        doc_node.metadata.update(doc_meta)

        section_groups: List[tuple[BaseNode, List[BaseNode]]] = []
        leaf_nodes: List[BaseNode] = []
        used_ids: dict[str, int] = {}

        for section in sections:
            leaves = self._pack_section(section)
            if not leaves:
                continue
            section_leaves: List[BaseNode] = []
            for leaf_body in leaves:
                leaf_text = self._leaf_text(section, leaf_body)
                meta = self._node_metadata(doc_meta, section, "leaf")
                node = TextNode(
                    id_=self._unique_id(leaf_text, used_ids),
                    text=leaf_text,
                    metadata=meta,
                    # heading_path is already inside the leaf text prefix, so
                    # repeating it in the embedding only costs tokens.
                    excluded_embed_metadata_keys=(
                        ["heading_path", "heading_level", "node_role"]
                        if self.prepend_heading_path
                        else ["node_role"]
                    ),
                )
                node.relationships[NodeRelationship.SOURCE] = (
                    doc_node.as_related_node_info()
                )
                section_leaves.append(node)

            for group in self._group_leaves(section, section_leaves):
                section_text = self._section_text(section, group)
                meta = self._node_metadata(doc_meta, section, "section")
                section_node = TextNode(
                    id_=self._unique_id(section_text, used_ids),
                    text=section_text,
                    metadata=meta,
                    excluded_embed_metadata_keys=["heading_path", "node_role"],
                )
                section_node.relationships[NodeRelationship.SOURCE] = (
                    doc_node.as_related_node_info()
                )
                section_node.relationships[NodeRelationship.PARENT] = (
                    doc_node.as_related_node_info()
                )
                section_node.relationships[NodeRelationship.CHILD] = [
                    leaf.as_related_node_info() for leaf in group
                ]
                for leaf in group:
                    leaf.relationships[NodeRelationship.PARENT] = (
                        section_node.as_related_node_info()
                    )
                section_groups.append((section_node, group))
                leaf_nodes.extend(group)

        if not section_groups:
            logger.debug("Document %s produced no content nodes", doc_node.node_id)
            return []

        doc_node.relationships[NodeRelationship.CHILD] = [
            node.as_related_node_info() for node, _ in section_groups
        ]
        if self.include_prev_next_rel:
            self._link_leaf_neighbours(leaf_nodes)

        # Interleaved, document-ordered output: sections immediately followed by
        # their leaves. Index construction over leaf nodes keeps this order.
        ordered: List[BaseNode] = [doc_node]
        for section_node, group in section_groups:
            ordered.append(section_node)
            ordered.extend(group)
        return ordered

    # ----------------------------------------------------------- block segmentation

    def _build_sections(self, text: str, doc_title: str) -> List[_Section]:
        blocks = self._segment_blocks(text)
        has_heading = any(block.kind == "heading" for block in blocks)

        if not has_heading:
            # Plain text (README dumps, slides exported without headings): one
            # section titled from the file name keeps the hierarchy shape intact.
            body = [b for b in blocks if b.text.strip()]
            if not body:
                return []
            return [
                _Section(
                    title=doc_title,
                    level=1,
                    heading_path=doc_title,
                    blocks=body,
                    heading_lines=[(1, doc_title)],
                )
            ]

        sections: List[_Section] = []
        stack: List[tuple[int, str]] = []
        current: Optional[_Section] = None

        for block in blocks:
            if block.kind == "heading":
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                stack.append((block.level, block.text))
                path = " > ".join(title for _, title in stack)
                current = _Section(
                    title=block.text,
                    level=block.level,
                    heading_path=path,
                    heading_lines=list(stack),
                )
                sections.append(current)
                continue

            if not block.text.strip():
                continue
            if current is None:
                # Content before the first heading still belongs to the document.
                current = _Section(
                    title=doc_title,
                    level=1,
                    heading_path=doc_title,
                    heading_lines=[(1, doc_title)],
                )
                sections.append(current)
            current.blocks.append(block)

        return [section for section in sections if section.blocks]

    def _segment_blocks(self, text: str) -> List[_Block]:
        """Cut the document into headings, paragraphs and atomic blocks."""
        lines = text.split("\n")
        blocks: List[_Block] = []
        paragraph: List[str] = []
        index = 0

        def flush_paragraph() -> None:
            if paragraph:
                body = "\n".join(paragraph).strip()
                if body:
                    blocks.append(_Block(kind="paragraph", text=body))
                paragraph.clear()

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()

            if not stripped:
                flush_paragraph()
                index += 1
                continue

            fence = _FENCE_RE.match(line)
            if fence:
                flush_paragraph()
                block_lines = [line]
                marker = fence.group(1)
                index += 1
                while index < len(lines):
                    block_lines.append(lines[index])
                    closing = _FENCE_RE.match(lines[index])
                    if closing and closing.group(1)[0] == marker[0]:
                        if len(closing.group(1)) >= len(marker):
                            index += 1
                            break
                    index += 1
                blocks.append(_Block(kind="code", text="\n".join(block_lines).rstrip()))
                continue

            math_block = self._consume_display_math(lines, index)
            if math_block is not None:
                flush_paragraph()
                block_text, index = math_block
                blocks.append(_Block(kind="math", text=block_text))
                continue

            if stripped.startswith("|"):
                # Consecutive pipe rows form one atomic table block.
                table = self._consume_table(lines, index)
                flush_paragraph()
                block_text, index = table
                blocks.append(_Block(kind="table", text=block_text))
                continue

            heading = _HEADING_RE.match(line)
            if heading:
                flush_paragraph()
                blocks.append(
                    _Block(
                        kind="heading",
                        text=_clean_heading(heading.group(2)),
                        level=len(heading.group(1)),
                    )
                )
                index += 1
                continue

            if _RULE_RE.match(line):
                # Horizontal rules are separator noise, not content.
                flush_paragraph()
                index += 1
                continue

            if _LIST_MARKER_RE.match(line) and paragraph:
                # Each bullet is its own unit so leaves cannot start mid-item.
                flush_paragraph()

            paragraph.append(line)
            index += 1

        flush_paragraph()
        return blocks

    @staticmethod
    def _consume_display_math(
        lines: List[str], index: int
    ) -> Optional[tuple[str, int]]:
        """Return (block_text, next_index) for a display-math block at ``index``."""
        stripped = lines[index].strip()

        pairs = (("$$", "$$"), ("\\[", "\\]"))
        for opener, closer in pairs:
            if not stripped.startswith(opener):
                continue
            rest = stripped[len(opener) :]
            if rest.rstrip().endswith(closer) and len(rest.strip()) >= len(closer):
                # Single-line form: $$ x = 1 $$
                return lines[index].rstrip(), index + 1
            block = [lines[index]]
            cursor = index + 1
            while cursor < len(lines):
                block.append(lines[cursor])
                if lines[cursor].strip().endswith(closer):
                    cursor += 1
                    break
                cursor += 1
            return "\n".join(block).rstrip(), cursor
        return None

    @staticmethod
    def _consume_table(lines: List[str], index: int) -> tuple[str, int]:
        block: List[str] = []
        cursor = index
        while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
            block.append(lines[cursor].rstrip())
            cursor += 1
        if len(block) < 2:
            # A lone "|" line is far more likely to be prose than a table.
            return lines[index].rstrip(), index + 1
        return "\n".join(block), cursor

    # ----------------------------------------------------------------- leaf packing

    def _leaf_budget(self, section: _Section) -> int:
        prefix_tokens = self._count(self._heading_prefix(section))
        return max(1, self.max_leaf_tokens - prefix_tokens)

    def _pack_section(self, section: _Section) -> List[str]:
        budget = self._leaf_budget(section)
        units = self._section_units(section, budget)
        if not units:
            return []

        leaves: List[str] = []
        current: List[str] = []
        for unit in units:
            # Count the string that would actually be stored: token counts are not
            # additive, so summing per-unit counts drifts past the cap.
            if current and self._count("\n\n".join([*current, unit])) > budget:
                leaves.append("\n\n".join(current))
                current = []
            current.append(unit)
        if current:
            leaves.append("\n\n".join(current))

        leaves = self._merge_short_tail(leaves, budget)
        return self._repair_boundaries(leaves, budget)

    def _section_units(self, section: _Section, budget: int) -> List[str]:
        """Turn blocks into packable units, never splitting what must stay whole."""
        units: List[str] = []
        for block in section.blocks:
            tokens = self._count(block.text)
            if tokens <= budget:
                units.append(block.text)
                continue

            if block.kind in ATOMIC_BLOCK_KINDS:
                # Last resort for a block larger than the whole leaf: cut between
                # lines so the block's internal syntax is never mangled.
                logger.debug(
                    "Atomic %s block of %d tokens exceeds leaf budget %d; "
                    "splitting at line boundaries",
                    block.kind,
                    tokens,
                    budget,
                )
                units.extend(self._split_lines(block.text, budget))
                continue

            units.extend(self._split_paragraph(block.text, budget))
        return self._group_list_items(units, budget)

    def _group_list_items(self, units: List[str], budget: int) -> List[str]:
        """Keep a contiguous list inside one unit when it fits the leaf.

        Packing used to break a leaf in the middle of a bullet list, so a chunk
        started with an orphaned item and lost the list it belonged to. Grouping
        whole lists first means a short list is never split; a list longer than a
        leaf still breaks, but only between items.
        """
        grouped: List[str] = []
        run: List[str] = []

        def flush() -> None:
            nonlocal run
            if run:
                grouped.append("\n\n".join(run))
                run = []

        for unit in units:
            if not _LIST_MARKER_RE.match(unit.lstrip()):
                flush()
                grouped.append(unit)
                continue
            if run and self._count("\n\n".join([*run, unit])) > budget:
                flush()
            run.append(unit)
        flush()
        return grouped

    def _split_paragraph(self, text: str, budget: int) -> List[str]:
        """Split an oversized paragraph at sentence ends, words only as last resort."""
        pieces: List[str] = []
        current: List[str] = []
        for sentence in split_sentences(text):
            if self._count(sentence) > budget:
                if current:
                    pieces.append(" ".join(current))
                    current = []
                pieces.extend(self._split_oversized_sentence(sentence, budget))
                continue
            if current and self._count(" ".join([*current, sentence])) > budget:
                pieces.append(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            pieces.append(" ".join(current))
        return self._fitted(pieces, budget)

    def _split_oversized_sentence(self, sentence: str, budget: int) -> List[str]:
        """Break a sentence too long for one leaf, keeping clauses together.

        Cutting such a sentence at arbitrary word boundaries produced fragments
        that began mid-clause; splitting after , ; : keeps each piece readable and
        leaves word- and character-level cuts as the last resort only.
        """
        clauses = re.split(r"(?<=[,;:])\s+", sentence)
        if len(clauses) < 2:
            return self._hard_split_words(sentence, budget)

        pieces: List[str] = []
        current: List[str] = []
        for clause in clauses:
            if self._count(clause) > budget:
                if current:
                    pieces.append(" ".join(current))
                    current = []
                pieces.extend(self._hard_split_words(clause, budget))
                continue
            if current and self._count(" ".join([*current, clause])) > budget:
                pieces.append(" ".join(current))
                current = []
            current.append(clause)
        if current:
            pieces.append(" ".join(current))
        return self._fitted(pieces, budget)

    def _split_lines(self, text: str, budget: int) -> List[str]:
        pieces: List[str] = []
        current: List[str] = []
        for line in text.split("\n"):
            if self._count(line) > budget:
                if current:
                    pieces.append("\n".join(current))
                    current = []
                pieces.extend(self._hard_split_words(line, budget))
                continue
            if current and self._count("\n".join([*current, line])) > budget:
                pieces.append("\n".join(current))
                current = []
            current.append(line)
        if current:
            pieces.append("\n".join(current))
        return self._fitted(pieces, budget)

    def _fitted(self, pieces: List[str], budget: int) -> List[str]:
        """Guarantee every piece fits: token counts shift across join boundaries."""
        fitted: List[str] = []
        for piece in pieces:
            if not piece.strip():
                continue
            if self._count(piece) <= budget:
                fitted.append(piece)
            else:
                fitted.extend(self._split_chars(piece, budget))
        return fitted

    def _hard_split_words(self, text: str, budget: int) -> List[str]:
        """Split on whitespace so a cut always lands on a word boundary."""
        words = re.split(r"(\s+)", text)
        pieces: List[str] = []
        current = ""
        current_tokens = 0
        for word in words:
            if not word:
                continue
            tokens = self._count(word)
            if tokens > budget:
                # A single unbroken run (long URL, base64 blob): character fallback.
                if current.strip():
                    pieces.append(current.strip())
                current = ""
                current_tokens = 0
                pieces.extend(self._split_chars(word, budget))
                continue
            if current and current_tokens + tokens > budget:
                pieces.append(current.strip())
                current = ""
                current_tokens = 0
            current += word
            current_tokens += tokens
        if current.strip():
            pieces.append(current.strip())
        return self._fitted(pieces, budget)

    def _split_chars(self, text: str, budget: int) -> List[str]:
        pieces: List[str] = []
        step = max(1, budget * 4)
        start = 0
        while start < len(text):
            end = min(len(text), start + step)
            chunk = text[start:end]
            # Shrink until the chunk fits; guarantees forward progress by 1 char.
            while end > start + 1 and self._count(chunk) > budget:
                end = start + max(1, (end - start) // 2)
                chunk = text[start:end]
            # Snap back to a space so a cut never lands inside a word.
            if end < len(text):
                space = chunk.rfind(" ")
                if space > len(chunk) // 2:
                    end = start + space + 1
                    chunk = text[start:end]
            pieces.append(chunk)
            start = end
        return [piece for piece in pieces if piece.strip()]

    def _merge_short_tail(self, leaves: List[str], budget: int) -> List[str]:
        while len(leaves) >= 2:
            tail_tokens = self._count(leaves[-1])
            if tail_tokens >= self.min_leaf_tokens:
                break
            merged = leaves[-2] + "\n\n" + leaves[-1]
            if self._count(merged) > budget:
                break
            leaves = leaves[:-2] + [merged]
        return leaves

    def _repair_boundaries(self, leaves: List[str], budget: int) -> List[str]:
        """Make sure no leaf starts in the middle of a sentence.

        Packing can only break at a unit boundary, but a unit may itself start with
        the tail of a sentence (bullets, tables, hard-split paragraphs). When a leaf
        begins mid-clause and the previous leaf does not end on terminal punctuation,
        the previous leaf's last sentence is moved forward.
        """
        index = 1
        while index < len(leaves):
            head = leaves[index].lstrip()
            previous = leaves[index - 1].rstrip()
            if not head or not self._starts_mid_clause(head):
                index += 1
                continue
            if self._ends_terminal(previous):
                index += 1
                continue

            sentences = split_sentences(previous)
            if len(sentences) > 1:
                last = sentences[-1]
                cut = previous.rfind(last)
                remainder = previous[:cut].rstrip() if cut > 0 else ""
                if remainder:
                    candidate = last + "\n\n" + head
                    if self._count(candidate) <= budget:
                        leaves[index - 1] = remainder
                        leaves[index] = candidate
                        index += 1
                        continue

            # Nothing to move (single unfinished fragment): merge instead of leaving
            # a leaf that begins mid-sentence.
            merged = previous + "\n\n" + head
            if self._count(merged) <= budget:
                leaves[index - 1] = merged
                del leaves[index]
                continue
            index += 1
        return leaves

    @staticmethod
    def _starts_mid_clause(text: str) -> bool:
        first = text[0]
        if first.islower():
            return True
        return first in ")]},.;:"

    @staticmethod
    def _ends_terminal(text: str) -> bool:
        stripped = text.rstrip()
        if not stripped:
            return True
        return stripped[-1] in _TERMINAL_CHARS

    # ------------------------------------------------------------ section grouping

    def _group_leaves(
        self, section: _Section, leaves: List[BaseNode]
    ) -> List[List[BaseNode]]:
        """Group leaves into section nodes, honouring ``max_section_tokens``."""
        if not self.max_section_tokens:
            return [leaves]

        groups: List[List[BaseNode]] = []
        current: List[BaseNode] = []
        current_tokens = 0
        for leaf in leaves:
            tokens = self._count(leaf.get_content())
            if current and current_tokens + tokens > self.max_section_tokens:
                groups.append(current)
                current = []
                current_tokens = 0
            current.append(leaf)
            current_tokens += tokens
        if current:
            groups.append(current)
        return groups

    # ------------------------------------------------------------------- utilities

    def _node_metadata(self, doc_meta: dict, section: _Section, role: str) -> dict:
        metadata = dict(doc_meta)
        metadata["heading_path"] = section.heading_path
        metadata["heading_level"] = section.level
        metadata["node_role"] = role
        return metadata

    def _heading_prefix(self, section: _Section) -> str:
        if not self.prepend_heading_path:
            return ""
        if self.include_heading_lines and section.heading_lines:
            lines = [
                f"{'#' * max(1, level)} {title}"
                for level, title in section.heading_lines
                if title
            ]
            if lines:
                return "\n".join(lines) + "\n\n"
        path = section.heading_path or section.title
        if not path:
            return ""
        return f"Section: {path}\n\n"

    def _leaf_text(self, section: _Section, body: str) -> str:
        return self._heading_prefix(section) + body

    def _section_text(self, section: _Section, leaves: List[BaseNode]) -> str:
        prefix = self._heading_prefix(section)
        bodies = []
        for leaf in leaves:
            text = leaf.text
            # The section is what a human reads after expanding a hit, so strip the
            # per-leaf heading context and let the section's own heading carry it.
            if prefix and text.startswith(prefix):
                text = text[len(prefix) :]
            elif text.startswith("Section: "):
                _, _, text = text.partition("\n\n")
            bodies.append(text)
        body = "\n\n".join(bodies)
        heading = section.heading_line
        return f"{heading}\n\n{body}" if heading else body

    def _unique_id(self, text: str, used: dict) -> str:
        # Content-addressed ids (32 hex chars, same shape as the uuid4 ids the rest
        # of the pipeline produces): rebuilding an unchanged document reuses the same
        # ids, so the docstore and the lexical side-table stay in sync.
        base = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        if base not in used:
            used[base] = 0
            return base
        used[base] += 1
        return f"{base}-{used[base]}"

    @staticmethod
    def _link_leaf_neighbours(leaves: List[BaseNode]) -> None:
        for position, leaf in enumerate(leaves):
            if position > 0:
                leaf.relationships[NodeRelationship.PREVIOUS] = leaves[
                    position - 1
                ].as_related_node_info()
            if position < len(leaves) - 1:
                leaf.relationships[NodeRelationship.NEXT] = leaves[
                    position + 1
                ].as_related_node_info()

    def _count(self, text: str) -> int:
        return len(self._tokenizer(text))

    @staticmethod
    def _document_title(doc: Document) -> str:
        metadata = doc.metadata or {}
        for key in ("display_name", "title", "file_name", "file_path"):
            value = metadata.get(key)
            if value:
                return os.path.splitext(str(value))[0]
        return doc.node_id
