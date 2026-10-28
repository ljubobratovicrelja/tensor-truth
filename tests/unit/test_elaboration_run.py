"""Tests for the elaboration branch of chat_runner.run_turn.

The chat service is stubbed and the orchestrator is disabled, so no LLM or
network call happens.
"""

from contextlib import nullcontext

import pytest

from tensortruth.services import chat_runner
from tensortruth.services.chat_runner import run_turn
from tensortruth.services.models import RAGChunk, SessionData
from tensortruth.services.session_service import SessionService

SESSION_ID = "s1"
PARENT = "Regularization keeps weights small."
SELECTED = "Regularization"


class _FakeSink:
    def __init__(self):
        self.events = []

    async def send_json(self, payload):
        self.events.append(payload)


class _FakePdfService:
    def get_index_path(self):
        return None


class _FakeChatService:
    """Streams a fixed answer and records the query arguments."""

    def __init__(self, tokens=("Small ", "weights.")):
        self.tokens = tokens
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        for token in self.tokens:
            yield RAGChunk(text=token)
        yield RAGChunk(is_complete=True, source_nodes=[], metrics=None)

    def extract_sources(self, source_nodes):
        return []


@pytest.fixture
def session_service(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    service = SessionService(tmp_path / "chat_sessions.json", sessions_dir)
    service.save(
        SessionData(
            current_id=SESSION_ID,
            sessions={
                SESSION_ID: {
                    "title": "Session",
                    "messages": [
                        {"role": "user", "content": "How does regularization work?"},
                        {"role": "assistant", "content": PARENT},
                    ],
                    "modules": [],
                    "module_mode": "manual",
                    "params": {"model": None, "orchestrator_enabled": False},
                }
            },
        )
    )
    return service


@pytest.fixture
def fake_pdf(monkeypatch):
    monkeypatch.setattr(
        chat_runner,
        "get_document_service",
        lambda session_id, scope: nullcontext(_FakePdfService()),
    )


@pytest.mark.asyncio
async def test_elaboration_run_persists_and_reports_id(session_service, fake_pdf):
    sink = _FakeSink()
    chat = _FakeChatService()

    await run_turn(
        sink,
        session_id=SESSION_ID,
        prompt="",
        kind="elaboration",
        metadata={
            "message_index": 1,
            "selected_text": SELECTED,
            "occurrence_index": 0,
        },
        session_service=session_service,
        chat_service=chat,
        project_service=object(),
    )

    done = [e for e in sink.events if e["type"] == "done"][-1]
    assert done["content"] == "Small weights."
    assert done["elaboration_id"]

    # History excludes the parent answer (it is in the prompt instead).
    call = chat.calls[0]
    assert call["session_messages"] == [
        {"role": "user", "content": "How does regularization work?"}
    ]
    assert f'"{SELECTED}"' in call["prompt"]
    assert PARENT in call["prompt"]
    assert "Elaboration mode" in call["params"]["system_prompt"]

    # Persisted onto the parent message; no new chat messages were added.
    stored = session_service.load().sessions[SESSION_ID]
    assert len(stored["messages"]) == 2
    elaborations = stored["messages"][1]["elaborations"]
    assert len(elaborations) == 1
    assert elaborations[0]["id"] == done["elaboration_id"]
    assert elaborations[0]["content"] == "Small weights."
    assert elaborations[0]["selected_text"] == SELECTED
    assert elaborations[0]["created_at"]


@pytest.mark.asyncio
async def test_elaboration_run_rejects_bad_target(session_service, fake_pdf):
    sink = _FakeSink()
    chat = _FakeChatService()

    await run_turn(
        sink,
        session_id=SESSION_ID,
        prompt="",
        kind="elaboration",
        metadata={"message_index": 0, "selected_text": SELECTED},
        session_service=session_service,
        chat_service=chat,
        project_service=object(),
    )

    assert sink.events[-1]["type"] == "error"
    assert chat.calls == []
    assert (
        "elaborations" not in session_service.load().sessions[SESSION_ID]["messages"][0]
    )
