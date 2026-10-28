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
    "concisely; never re-answer the whole question."
)


def build_elaboration_prompt(parent_content: str, selected_text: str) -> str:
    """Build the prompt that asks for an explanation of one passage.

    Args:
        parent_content: The full assistant answer the passage belongs to.
        selected_text: The passage the reader selected, already trimmed.

    Returns:
        Prompt string with the parent answer and the quoted passage.
    """
    return (
        "Your previous answer contained this passage:\n\n"
        f"{parent_content}\n\n"
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

    Returns:
        Prompt string with the parent answer, the passage, the prior thread
        and the quoted question.
    """
    thread_text = _render_thread(thread)
    return (
        "Your previous answer contained this passage:\n\n"
        f"{parent_content}\n\n"
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
