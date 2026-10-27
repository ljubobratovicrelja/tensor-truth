"""Direct-to-context document attachment (no vector index).

Documents can be added to a session's *conversation context* instead of (or
in addition to) the vector index.  Their extracted markdown text is stored in
the session's ``markdown/`` directory and flagged ``in_context`` in the
scope's ``metadata.json``; this module builds the prompt block from those
files and enforces a hard character (token) budget.

Budget rule
-----------
Token accounting matches the orchestrator (``CHARS_PER_TOKEN = 4``).  The
block may consume at most ``CONTEXT_BUDGET_PCT`` (30%) of the model's context
window, and never more than what remains after the base system prompt, the
user prompt, the response buffer (50%) and a minimum history reserve (5%).
Within that budget, space is divided fairly: small documents are included in
full, and whatever they don't use is redistributed among the larger ones.
Every document that does not fit is truncated at the allocation boundary with
an explicit ``[truncated]`` marker and listed in the block's truncation note.
Documents that get zero space are reported as not included — never silently
dropped.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Must match ``orchestrator_service.CHARS_PER_TOKEN``.
CHARS_PER_TOKEN = 4

# Share of the context window the attached-document block may consume.
CONTEXT_BUDGET_PCT = 0.30
# Minimum share of the window reserved for chat history when documents exist.
MIN_HISTORY_PCT = 0.05
# Don't bother emitting a fragment smaller than this (chars).
MIN_DOC_CHARS = 200

_TRUNCATION_MARKER = "\n...[truncated]"

# Rough allowance for the block header, per-document labels and budget notes,
# subtracted from the hard budget before allocating document text.
_OVERHEAD_BASE_CHARS = 800
_OVERHEAD_PER_DOC_CHARS = 200


def _overhead_chars(count: int) -> int:
    return _OVERHEAD_BASE_CHARS + _OVERHEAD_PER_DOC_CHARS * count


def allocate_budget(sizes: List[int], budget_chars: int) -> List[int]:
    """Split ``budget_chars`` fairly across documents of the given sizes.

    Documents that fit inside an equal share are satisfied in full; their
    unused share is redistributed to the remaining (larger) documents.  The
    final pass gives every still-oversized document an equal slice, so the
    total allocation never exceeds ``budget_chars``.

    Returns:
        One allocation per input size (may be 0 when there is no budget).
    """
    n = len(sizes)
    allocation = [0] * n
    if n == 0 or budget_chars <= 0:
        return allocation

    remaining = set(range(n))
    budget_left = budget_chars

    while remaining and budget_left > 0:
        share = budget_left // len(remaining)
        if share <= 0:
            break

        satisfied = [i for i in remaining if sizes[i] <= share]
        if satisfied:
            for i in satisfied:
                allocation[i] = sizes[i]
                budget_left -= sizes[i]
                remaining.discard(i)
            continue

        # Everyone left is larger than an equal share — hand out equal slices.
        for i in sorted(remaining):
            allocation[i] = min(share, budget_left)
            budget_left -= allocation[i]
        break

    return allocation


def truncate_for_budget(text: str, limit: int) -> Tuple[str, bool]:
    """Truncate ``text`` to ``limit`` chars, appending a marker if cut."""
    if limit <= 0:
        return "", True
    if len(text) <= limit:
        return text, False
    keep = max(0, limit - len(_TRUNCATION_MARKER))
    return text[:keep].rstrip() + _TRUNCATION_MARKER, True


def build_context_block(
    documents: List[Dict[str, Any]], budget_chars: int
) -> Tuple[str, List[Dict[str, Any]]]:
    """Render the attached-documents block for the system prompt.

    Args:
        documents: ``[{"doc_id", "filename", "text"}, ...]``.
        budget_chars: Hard character budget for the whole block body.

    Returns:
        ``(block, report)`` where ``report`` has one entry per document with
        ``doc_id``, ``filename``, ``chars``, ``kept_chars``, ``tokens``,
        ``truncated`` and ``included``.
    """
    if not documents:
        return "", []

    sizes = [len(d.get("text") or "") for d in documents]
    body_budget = max(0, budget_chars - _overhead_chars(len(documents)))
    allocation = allocate_budget(sizes, body_budget)

    report: List[Dict[str, Any]] = []
    parts: List[str] = []
    included = 0

    for doc, size, limit in zip(documents, sizes, allocation):
        doc_id = doc.get("doc_id", "")
        filename = doc.get("filename") or doc_id
        text = doc.get("text") or ""

        if limit < MIN_DOC_CHARS and limit < size:
            report.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "chars": size,
                    "kept_chars": 0,
                    "tokens": 0,
                    "truncated": False,
                    "included": False,
                }
            )
            continue

        kept_text, truncated = truncate_for_budget(text, limit)
        kept_chars = len(kept_text)
        # Strip the internal markdown header the upload path prepends.
        kept_text = strip_document_header(kept_text)
        included += 1
        tokens = kept_chars // CHARS_PER_TOKEN

        report.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "chars": size,
                "kept_chars": kept_chars,
                "tokens": tokens,
                "truncated": truncated,
                "included": True,
            }
        )

        parts.append(
            f"--- BEGIN DOCUMENT {included}: {filename} "
            f"(doc_id: {doc_id}, ~{tokens:,} tokens"
            f"{', TRUNCATED' if truncated else ''}) ---\n"
            f"{kept_text}\n"
            f"--- END DOCUMENT {included}: {doc_id} ---"
        )

    omitted = [r for r in report if not r["included"]]
    truncated_docs = [r for r in report if r["truncated"]]
    if not parts:
        return "", report

    header = (
        "==================== ATTACHED DOCUMENTS (FULL TEXT) ====================\n"
        f"The user attached {len(documents)} document(s) directly to this "
        "conversation's context. Their text is included below verbatim — it "
        "was NOT retrieved by search, so no rag_query call is needed to read "
        "it. Treat the text as primary source material and cite documents by "
        'their label (e.g. "Document 1").'
    )
    if truncated_docs:
        header += (
            "\n\nBUDGET NOTE: the attached-document budget was reached and the "
            "following document(s) were truncated: "
            + ", ".join(
                f"{r['filename']} ({r['kept_chars']:,} of {r['chars']:,} chars kept)"
                for r in truncated_docs
            )
            + ". Mention the truncation to the user if it affects the answer."
        )
    if omitted:
        header += (
            "\n\nBUDGET NOTE: these document(s) did not fit in the context "
            "budget and were NOT included: "
            + ", ".join(f"{r['filename']} ({r['chars']:,} chars)" for r in omitted)
            + ". Tell the user they need to index or remove other documents."
        )

    block = header + "\n\n" + "\n\n".join(parts)
    return block, report


def strip_document_header(text: str) -> str:
    """Remove the ``# Document: ...`` header written at upload/conversion time."""
    if not text.startswith("# Document:"):
        return text
    marker = "\n---\n\n"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx + len(marker) :].lstrip("\n")
