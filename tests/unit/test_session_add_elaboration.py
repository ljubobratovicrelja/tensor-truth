"""Tests for SessionService.add_elaboration."""

import pytest

from tensortruth.services.models import SessionData
from tensortruth.services.session_service import SessionService


@pytest.fixture
def session_service(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return SessionService(tmp_path / "chat_sessions.json", sessions_dir)


@pytest.fixture
def data():
    return SessionData(
        current_id="s1",
        sessions={
            "s1": {
                "title": "t",
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ],
            }
        },
    )


ELABORATION = {
    "id": "e1",
    "selected_text": "a",
    "occurrence_index": 0,
    "content": "explained",
    "created_at": "2026-10-28T20:00:00+00:00",
}


@pytest.mark.unit
class TestAddElaboration:
    def test_appends_to_target_message(self, session_service, data):
        updated = session_service.add_elaboration("s1", 1, ELABORATION, data)

        messages = updated.sessions["s1"]["messages"]
        assert messages[1]["elaborations"] == [ELABORATION]
        assert "elaborations" not in messages[0]
        assert "elaborations" not in messages[2]

    def test_appends_to_existing_list(self, session_service, data):
        first = session_service.add_elaboration("s1", 1, ELABORATION, data)
        second = session_service.add_elaboration(
            "s1", 1, dict(ELABORATION, id="e2", content="again"), first
        )

        elaborations = second.sessions["s1"]["messages"][1]["elaborations"]
        assert [e["id"] for e in elaborations] == ["e1", "e2"]

    def test_input_is_not_mutated(self, session_service, data):
        before = data.sessions["s1"]["messages"][1]

        session_service.add_elaboration("s1", 1, ELABORATION, data)

        assert "elaborations" not in before
        assert "elaborations" not in data.sessions["s1"]["messages"][1]

    def test_index_out_of_range_is_noop(self, session_service, data):
        assert session_service.add_elaboration("s1", 5, ELABORATION, data) is data
        assert session_service.add_elaboration("s1", -1, ELABORATION, data) is data

    def test_non_assistant_target_is_noop(self, session_service, data):
        assert session_service.add_elaboration("s1", 0, ELABORATION, data) is data

    def test_unknown_session_is_noop(self, session_service, data):
        assert session_service.add_elaboration("nope", 1, ELABORATION, data) is data


USER_TURN = {
    "role": "user",
    "content": "Why does that follow?",
    "created_at": "2026-10-28T20:01:00+00:00",
}
ASSISTANT_TURN = {
    "role": "assistant",
    "content": "Because the norm bounds the gradient.",
    "created_at": "2026-10-28T20:01:05+00:00",
}


@pytest.fixture
def data_with_elaboration(session_service):
    base = SessionData(
        current_id="s1",
        sessions={
            "s1": {
                "title": "t",
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
            }
        },
    )
    return session_service.add_elaboration("s1", 1, ELABORATION, base)


@pytest.mark.unit
class TestAddElaborationMessage:
    def test_appends_turns_to_the_right_elaboration(
        self, session_service, data_with_elaboration
    ):
        first = session_service.add_elaboration_message(
            "s1", 1, "e1", USER_TURN, data_with_elaboration
        )
        second = session_service.add_elaboration_message(
            "s1", 1, "e1", ASSISTANT_TURN, first
        )

        elaboration = second.sessions["s1"]["messages"][1]["elaborations"][0]
        assert elaboration["messages"] == [USER_TURN, ASSISTANT_TURN]
        # The original explanation is untouched.
        assert elaboration["content"] == "explained"

    def test_other_elaborations_are_untouched(
        self, session_service, data_with_elaboration
    ):
        base = session_service.add_elaboration(
            "s1", 1, dict(ELABORATION, id="e2"), data_with_elaboration
        )
        updated = session_service.add_elaboration_message(
            "s1", 1, "e2", USER_TURN, base
        )

        elaborations = updated.sessions["s1"]["messages"][1]["elaborations"]
        assert "messages" not in elaborations[0]
        assert elaborations[1]["messages"] == [USER_TURN]

    def test_unknown_elaboration_id_is_noop(
        self, session_service, data_with_elaboration
    ):
        assert (
            session_service.add_elaboration_message(
                "s1", 1, "nope", USER_TURN, data_with_elaboration
            )
            is data_with_elaboration
        )

    def test_bad_target_is_noop(self, session_service, data_with_elaboration):
        assert (
            session_service.add_elaboration_message(
                "s1", 0, "e1", USER_TURN, data_with_elaboration
            )
            is data_with_elaboration
        )
        assert (
            session_service.add_elaboration_message(
                "s1", 99, "e1", USER_TURN, data_with_elaboration
            )
            is data_with_elaboration
        )
        assert (
            session_service.add_elaboration_message(
                "nope", 1, "e1", USER_TURN, data_with_elaboration
            )
            is data_with_elaboration
        )

    def test_input_is_not_mutated(self, session_service, data_with_elaboration):
        before = data_with_elaboration.sessions["s1"]["messages"][1]
        before_elaboration = before["elaborations"][0]

        session_service.add_elaboration_message(
            "s1", 1, "e1", USER_TURN, data_with_elaboration
        )

        assert "messages" not in before_elaboration
        assert (
            "messages"
            not in data_with_elaboration.sessions["s1"]["messages"][1]["elaborations"][
                0
            ]
        )
