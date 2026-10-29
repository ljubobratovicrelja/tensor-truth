"""Prompt construction for selection-driven elaboration.

An elaboration explains one uncertain passage of an earlier assistant answer.
The parent answer is embedded verbatim in the prompt (it is not duplicated in
the chat history), and the selected passage is quoted so the model can bind the
explanation to the exact words the reader is asking about.
"""

from typing import Any, Dict, List, Optional

# Upper bound on a selected passage; longer selections are rejected by the API.
MAX_SELECTED_TEXT_LENGTH = 4000

# Upper bound on a follow-up question inside an elaboration thread.
MAX_FOLLOWUP_PROMPT_LENGTH = 4000

ELABORATION_SYSTEM_DIRECTIVE = (
    "\n\nElaboration mode: the user asked about one unclear passage from your "
    "previous answer. Answer only that passage or the thread question, "
    "concisely; never re-answer the whole question.\n"
    "Ground every claim. This answer goes deeper than the original one, so do "
    "not rely on memory or on the explanation sounding plausible: before "
    "answering, retrieve the material the original answer was based on and, "
    "when a fact is not fully covered by it, use the available tools (RAG "
    "retrieval, web search/fetch) to check it. Answer from the supplied context "
    "only when the passage is trivial or its meaning is entirely internal to "
    "the answer. If the sources do not support a claim, say so instead of "
    "inventing one."
)

# Source excerpts injected into the prompt: enough to anchor the explanation
# without letting a citation-heavy answer crowd out the actual question.
MAX_SOURCE_BLOCK_CHARS = 6000
MAX_SOURCE_EXCERPT_CHARS = 1200


def build_source_block(
    sources: Optional[List[Dict[str, Any]]],
    max_chars: int = MAX_SOURCE_BLOCK_CHARS,
) -> str:
    """Render the excerpts that backed the original answer.

    These are leads, not proof: the elaboration is instructed to verify and to
    re-retrieve when they are incomplete.
    """
    if not sources:
        return ""

    lines: List[str] = [
        "Source excerpts that backed the original answer (a starting point, "
        "not necessarily complete — verify them and retrieve more if needed):"
    ]
    used = 0
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        text = " ".join(str(source.get("text") or "").split())
        if not text:
            continue
        raw_metadata = source.get("metadata")
        metadata: Dict[str, Any] = (
            raw_metadata if isinstance(raw_metadata, dict) else {}
        )
        label = (
            metadata.get("filename")
            or metadata.get("url")
            or metadata.get("source")
            or metadata.get("file_path")
            or f"source {index}"
        )
        excerpt = text[:MAX_SOURCE_EXCERPT_CHARS]
        entry = f"[{index}] ({label}) {excerpt}"
        if used + len(entry) > max_chars:
            break
        lines.append(entry)
        used += len(entry) + 1

    return "\n".join(lines) if len(lines) > 1 else ""


def build_elaboration_prompt(
    parent_content: str,
    selected_text: str,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the prompt that asks for an explanation of one passage.

    Args:
        parent_content: The full assistant answer the passage belongs to.
        selected_text: The passage the reader selected, already trimmed.
        sources: Source excerpts that backed the original answer, if any.

    Returns:
        Prompt string with the parent answer, its sources and the quoted passage.
    """
    source_block = build_source_block(sources)
    source_section = f"{source_block}\n\n" if source_block else ""
    return (
        "Your previous answer contained this passage:\n\n"
        f"{parent_content}\n\n"
        f"{source_section}"
        "The reader has selected this passage from that answer:\n\n"
        f'"{selected_text}"\n\n'
        "They have told you the passage is not clear: they lack the context or "
        "background knowledge needed to understand it, and that gap is what makes "
        "the rest of the answer hard to follow. Explain this passage and only "
        "this passage.\n\n"
        "Rules:\n"
        "- Be brief and direct: 2-4 sentences, at most ~120 words. No preamble.\n"
        "- Say what it means in this answer's own terms, and name the specific "
        "concept or assumption the reader is missing.\n"
        "- Ground it in sources: use the excerpts above and the retrieval/web "
        "tools to confirm anything factual. If the source material does not "
        "settle it, retrieve more before answering.\n"
        "- Do not restate the whole answer, do not add tangents, alternatives, or "
        "caveats that raise new questions.\n"
        "- If a minimal example makes it concrete, use one; otherwise omit it.\n"
        "- Plain prose. Markdown only if the passage itself is code or a formula."
    )


def _render_thread(thread: List[Dict[str, Any]]) -> str:
    """Render thread turns as alternating ``You:`` / ``Reader:`` lines."""
    lines: List[str] = []
    for turn in thread:
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        speaker = "You" if turn.get("role") == "assistant" else "Reader"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def build_elaboration_followup_prompt(
    parent_content: str,
    selected_text: str,
    thread: List[Dict[str, Any]],
    question: str,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the prompt for a follow-up question inside an elaboration thread.

    Args:
        parent_content: The full assistant answer the passage belongs to.
        selected_text: The passage the reader selected, already trimmed.
        thread: Ordered thread turns. Normally starts with the original
            explanation (``role="assistant"``) followed by the follow-up
            ``user``/``assistant`` messages, so the prompt shows the whole
            conversation in order.
        question: The new follow-up question, already trimmed.
        sources: Source excerpts that backed the original answer, if any.

    Returns:
        Prompt string with the parent answer, its sources, the passage, the
        prior thread and the quoted question.
    """
    thread_text = _render_thread(thread)
    source_block = build_source_block(sources)
    source_section = f"{source_block}\n\n" if source_block else ""
    return (
        "Your previous answer contained this passage:\n\n"
        f"{parent_content}\n\n"
        f"{source_section}"
        "The reader asked you to explain this passage:\n\n"
        f'"{selected_text}"\n\n'
        "Your explanation and the thread so far:\n\n"
        f"{thread_text}\n\n"
        "The reader now asks:\n\n"
        f'"{question}"\n\n'
        "Answer this follow-up only, inside the same explanation thread.\n"
        "Rules:\n"
        "- 2-3 sentences, direct, no preamble.\n"
        "- Resolve the specific doubt; do not repeat the earlier explanation or "
        "widen the thread.\n"
        "- If the answer depends on a fact, verify it against the excerpts above "
        "or the retrieval/web tools before answering.\n"
        "- No new tangents, alternatives or caveats that raise further "
        "questions.\n"
        "- Plain prose."
    )


def with_elaboration_directive(system_prompt: Optional[str]) -> str:
    """Append the elaboration directive to an existing system prompt.

    The session (or project-composed) system prompt is preserved; the
    directive only narrows what the model should do for this run.
    """
    return f"{system_prompt or ''}{ELABORATION_SYSTEM_DIRECTIVE}"
