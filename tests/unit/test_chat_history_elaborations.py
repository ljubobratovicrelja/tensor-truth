"""Tests for expand_elaborations (history fed to later model turns)."""

import copy

import pytest

from tensortruth.services.chat_history import (
    ELABORATION_SEPARATOR,
    expand_elaborations,
)


def _elaboration(selected: str, content: str) -> dict:
    return {
        "id": "abc123",
        "selected_text": selected,
        "occurrence_index": 0,
        "content": content,
        "created_at": "2026-10-28T20:00:00+00:00",
    }


@pytest.mark.unit
class TestExpandElaborations:
    def test_identity_when_no_elaborations(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert expand_elaborations(messages) is messages

    def test_empty_input(self):
        assert expand_elaborations([]) == []

    def test_appends_block_after_parent_content(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "The loss surface is high dimensional.",
                "elaborations": [_elaboration("loss surface", "The function space.")],
            },
        ]

        expanded = expand_elaborations(messages)

        content = expanded[1]["content"]
        assert content.startswith("The loss surface is high dimensional.")
        assert ELABORATION_SEPARATOR in content
        assert 'they asked you to explain "loss surface"' in content
        assert "You answered: The function space.]" in content
        assert expanded[0] == messages[0]

    def test_does_not_mutate_input(self):
        messages = [
            {
                "role": "assistant",
                "content": "Answer.",
                "elaborations": [_elaboration("Answer", "Explanation.")],
            }
        ]
        snapshot = copy.deepcopy(messages)

        expanded = expand_elaborations(messages)

        assert messages == snapshot
        assert expanded[0] is not messages[0]
        assert expanded[0]["content"] != messages[0]["content"]

    def test_multiple_elaborations_keep_creation_order(self):
        messages = [
            {
                "role": "assistant",
                "content": "Answer.",
                "elaborations": [
                    _elaboration("first", "explains first"),
                    _elaboration("second", "explains second"),
                ],
            }
        ]

        content = expand_elaborations(messages)[0]["content"]

        assert content.index("first") < content.index("second")
        assert content.index("explains first") < content.index("explains second")

    def test_only_annotated_messages_change(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "plain"},
            {
                "role": "assistant",
                "content": "annotated",
                "elaborations": [_elaboration("annotated", "note")],
            },
        ]

        expanded = expand_elaborations(messages)

        assert expanded[0] is messages[0]
        assert expanded[1] is messages[1]
        assert expanded[2] is not messages[2]


class TestChatContextExpansion:
    """Chat contexts must carry elaboration notes into model history."""

    def test_from_session_expands_assistant_elaborations(self):
        from tensortruth.services.chat_runner import ChatContext

        class _FakePdfService:
            def get_index_path(self):
                return None

        session = {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "elaborations": [_elaboration("answer", "clearer")],
                },
            ],
            "modules": [],
            "params": {},
        }

        context = ChatContext.from_session(
            "session-1", "next question", session, _FakePdfService()
        )

        assert "Reader follow-up" in context.session_messages[1]["content"]
        assert "clearer" in context.session_messages[1]["content"]
        # The session itself is untouched.
        assert "elaborations" in session["messages"][1]
        assert "Reader follow-up" not in session["messages"][1]["content"]
