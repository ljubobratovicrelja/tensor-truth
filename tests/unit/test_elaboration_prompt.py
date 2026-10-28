"""Tests for the elaboration prompt builder and system directive."""

import pytest

from tensortruth.services.elaboration import (
    ELABORATION_SYSTEM_DIRECTIVE,
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
