"""Tests for the elaboration prompt builder and system directive."""

import pytest

from tensortruth.services.elaboration import (
    ELABORATION_SYSTEM_DIRECTIVE,
    build_elaboration_followup_prompt,
    build_elaboration_prompt,
    with_elaboration_directive,
)

PARENT = "Gradient descent follows the negative gradient of the loss surface."
SELECTED = "the loss surface"


@pytest.mark.unit
class TestBuildElaborationPrompt:
    def test_includes_parent_answer_and_quoted_selection(self):
        prompt = build_elaboration_prompt(PARENT, SELECTED)

        assert PARENT in prompt
        assert f'"{SELECTED}"' in prompt
        # The parent answer is presented before the quoted selection.
        assert prompt.index(PARENT) < prompt.index(f'"{SELECTED}"')

    def test_carries_selection_constraint_rules(self):
        prompt = build_elaboration_prompt(PARENT, SELECTED)

        assert "only this passage" in prompt
        assert "2-4 sentences" in prompt
        assert "No preamble" in prompt
        assert "do not add tangents" in prompt
        assert "Do not restate the whole answer" in prompt

    def test_does_not_leak_markdown_wrapping(self):
        prompt = build_elaboration_prompt(PARENT, SELECTED)
        assert "```" not in prompt


THREAD = [
    {"role": "assistant", "content": "It is the surface the optimizer descends."},
    {"role": "user", "content": "Why does the shape matter?"},
    {"role": "assistant", "content": "Because curvature sets the step size."},
]
QUESTION = "What is curvature here?"


@pytest.mark.unit
class TestBuildElaborationFollowupPrompt:
    def test_includes_parent_selection_thread_and_question(self):
        prompt = build_elaboration_followup_prompt(PARENT, SELECTED, THREAD, QUESTION)

        assert PARENT in prompt
        assert f'"{SELECTED}"' in prompt
        assert prompt.index(PARENT) < prompt.index(f'"{SELECTED}"')
        assert f'"{QUESTION}"' in prompt

    def test_thread_is_rendered_in_order(self):
        prompt = build_elaboration_followup_prompt(PARENT, SELECTED, THREAD, QUESTION)

        positions = [
            prompt.index("You: It is the surface the optimizer descends."),
            prompt.index("Reader: Why does the shape matter?"),
            prompt.index("You: Because curvature sets the step size."),
            prompt.index(f'"{QUESTION}"'),
        ]
        assert positions == sorted(positions)

    def test_carries_followup_brevity_rules(self):
        prompt = build_elaboration_followup_prompt(PARENT, SELECTED, THREAD, QUESTION)

        assert "2-3 sentences" in prompt
        assert "no preamble" in prompt
        assert "Answer this follow-up only" in prompt
        assert "do not repeat the earlier explanation" in prompt
        assert "No new tangents" in prompt

    def test_thread_keeps_each_turn_separate(self):
        prompt = build_elaboration_followup_prompt(PARENT, SELECTED, THREAD, QUESTION)
        thread_block = prompt.split("Your explanation and the thread so far:")[1]
        thread_block = thread_block.split("The reader now asks:")[0]

        assert thread_block.count("You:") == 2
        assert thread_block.count("Reader:") == 1


@pytest.mark.unit
class TestElaborationDirective:
    def test_directive_preserves_existing_system_prompt(self):
        combined = with_elaboration_directive("You are a physics tutor.")

        assert combined.startswith("You are a physics tutor.")
        assert combined.endswith(ELABORATION_SYSTEM_DIRECTIVE)
        assert "Elaboration mode" in combined

    def test_directive_without_system_prompt(self):
        assert with_elaboration_directive(None) == ELABORATION_SYSTEM_DIRECTIVE
        assert with_elaboration_directive("") == ELABORATION_SYSTEM_DIRECTIVE


SOURCES = [
    {
        "text": "A loss surface maps parameter vectors to loss values; curvature is "
        "the second derivative along a direction.",
        "metadata": {"filename": "optimization.pdf"},
    },
    {
        "text": "Step size is bounded by 1/L where L is the Lipschitz constant.",
        "metadata": {"url": "https://example.com/gd"},
    },
]


@pytest.mark.unit
class TestElaborationSources:
    def test_prompt_carries_source_excerpts_and_labels(self):
        prompt = build_elaboration_prompt(PARENT, SELECTED, sources=SOURCES)

        assert "(optimization.pdf)" in prompt
        assert "(https://example.com/gd)" in prompt
        assert "loss surface maps parameter vectors" in prompt
        # Excerpts are presented as leads to verify, not as complete proof.
        assert "not necessarily complete" in prompt

    def test_followup_prompt_carries_source_excerpts(self):
        prompt = build_elaboration_followup_prompt(
            PARENT, SELECTED, THREAD, QUESTION, sources=SOURCES
        )

        assert "(optimization.pdf)" in prompt
        assert "verify it against the excerpts" in prompt

    def test_source_block_is_bounded(self):
        huge = [{"text": "x" * 5000, "metadata": {"filename": "a"}} for _ in range(5)]
        prompt = build_elaboration_prompt(PARENT, SELECTED, sources=huge)

        # 6000-char cap plus the labels/markers; nothing near 25k.
        assert len(prompt) < 9000

    def test_no_sources_leaves_prompt_unchanged(self):
        assert build_elaboration_prompt(PARENT, SELECTED, sources=[]) == (
            build_elaboration_prompt(PARENT, SELECTED)
        )

    def test_directive_forces_source_grounding(self):
        assert "Ground every claim" in ELABORATION_SYSTEM_DIRECTIVE
        assert "retrieve the material" in ELABORATION_SYSTEM_DIRECTIVE
        assert "web search" in ELABORATION_SYSTEM_DIRECTIVE
        # Trivial cases may still be answered from context.
        assert "only when the passage is trivial" in ELABORATION_SYSTEM_DIRECTIVE
        # And it must admit when the sources fall short.
        assert "say so instead of inventing one" in ELABORATION_SYSTEM_DIRECTIVE
