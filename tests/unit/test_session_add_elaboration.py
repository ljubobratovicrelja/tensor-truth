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
