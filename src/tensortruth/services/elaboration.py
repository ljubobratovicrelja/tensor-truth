"""Prompt construction for selection-driven elaboration.

An elaboration explains one uncertain passage of an earlier assistant answer.
The parent answer is embedded verbatim in the prompt (it is not duplicated in
the chat history), and the selected passage is quoted so the model can bind the
explanation to the exact words the reader is asking about.
"""

from typing import Optional

# Upper bound on a selected passage; longer selections are rejected by the API.
MAX_SELECTED_TEXT_LENGTH = 4000

ELABORATION_SYSTEM_DIRECTIVE = (
    "\n\nElaboration mode: the user asked about one unclear passage from your "
    "previous answer. Answer only that passage, concisely; never re-answer the "
    "whole question."
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


def with_elaboration_directive(system_prompt: Optional[str]) -> str:
    """Append the elaboration directive to an existing system prompt.

    The session (or project-composed) system prompt is preserved; the
    directive only narrows what the model should do for this run.
    """
    return f"{system_prompt or ''}{ELABORATION_SYSTEM_DIRECTIVE}"
