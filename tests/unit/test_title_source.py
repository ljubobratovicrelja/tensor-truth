"""Titles must be derived from the user's message, not the assistant's answer.

Two delivery points are covered:

* ``POST /api/sessions/title-preview`` — the composer's speculative preview.
* ``_apply_provisional_title`` — the immediate label written on the first turn.
"""

import pytest
from fastapi import HTTPException

from tensortruth.api.routes.sessions import preview_title
from tensortruth.api.schemas import TitlePreviewRequest
from tensortruth.services.chat_runner import _apply_provisional_title

USER_PROMPT = "Find the ResNet and BERT papers on arXiv"
NARRATION = "I'll search arXiv for both papers. Found both. Adding them now."


class _FakeSink:
    def __init__(self) -> None:
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class _FakeSessionService:
    def __init__(self) -> None:
        self.titles = []

    def load(self):
        return {"sessions": {}}

    def save(self, data):
        self.data = data

    def update_title(self, session_id, title, data):
        self.titles.append(title)
        return data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_preview_titles_from_the_user_prompt(monkeypatch):
    """The preview endpoint feeds the model the user's words, not an answer."""
    captured = {}

    async def fake_generate(
        text, model_name=None, context_window=16384, provider_id=None
    ):
        captured["text"] = text
        return "ResNet And BERT Papers"

    monkeypatch.setattr(
        "tensortruth.app_utils.title_generation.generate_smart_title_async",
        fake_generate,
    )

    response = await preview_title(
        TitlePreviewRequest(prompt=USER_PROMPT, model="qwen", provider_id="ollama")
    )

    assert response.title == "ResNet And BERT Papers"
    assert captured["text"] == USER_PROMPT


@pytest.mark.unit
@pytest.mark.asyncio
async def test_preview_rejects_an_empty_prompt():
    with pytest.raises(HTTPException) as exc:
        await preview_title(TitlePreviewRequest(prompt="   "))
    assert exc.value.status_code == 400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provisional_title_is_the_user_prompt():
    """First turn labels the session from the prompt before the answer streams."""
    sink = _FakeSink()
    service = _FakeSessionService()

    await _apply_provisional_title(sink, service, "s1", USER_PROMPT)

    assert service.titles, "provisional title was never persisted"
    title = service.titles[0]
    assert title.startswith("Find the ResNet")
    assert "I'll search" not in title
    assert sink.sent == [{"type": "title", "title": title}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provisional_title_skips_an_empty_prompt():
    sink = _FakeSink()
    service = _FakeSessionService()

    await _apply_provisional_title(sink, service, "s1", "   ")

    assert service.titles == []
    assert sink.sent == []
