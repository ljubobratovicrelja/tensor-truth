"""Title generation must survive reasoning models and null content.

Observed failure (2026-10-27, deepseek-v4.1-flash via OpenRouter): the title
call sent ``max_tokens=30``; the model spent all 30 tokens on hidden
``reasoning`` and returned ``content: null`` with ``finish_reason: "length"``.
``_strip_markdown(None)`` then raised ``TypeError: 'NoneType' object is not
subscriptable``, the caller swallowed it, and every session was titled with a
60-char truncation of the assistant's answer.
"""

import pytest

from tensortruth.app_utils import title_generation as tg

PROMPT = "I'm studying LeJEPA paper and fetch the I-JEPA paper from arxiv"


def _payload(choices, status=200):
    return {"choices": choices}


def _choice(content, finish_reason="stop"):
    return {
        "index": 0,
        "finish_reason": finish_reason,
        "message": {"role": "assistant", "content": content, "reasoning": "thinking"},
    }


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    """Stands in for ``aiohttp.ClientSession``; records every payload sent."""

    def __init__(self, payloads, timeout=None, shared=False):
        # A retry builds a second ClientSession, so the scripted responses must
        # live in one queue shared by every session the factory hands out.
        self._payloads = payloads if shared else list(payloads)
        self.posts = []

    def post(self, url, json=None, headers=None):
        self.posts.append(json)
        return _FakeResponse(self._payloads.pop(0))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def fake_aiohttp(monkeypatch):
    """Patch ClientSession with a factory returning one scripted session."""

    def install(payloads):
        sessions = []

        def factory(timeout=None):
            session = _FakeSession(payloads, timeout, shared=True)
            sessions.append(session)
            return session

        monkeypatch.setattr(tg.aiohttp, "ClientSession", factory)
        return sessions

    return install


@pytest.mark.unit
class TestNullContentTolerance:
    def test_strip_markdown_accepts_none(self):
        assert tg._strip_markdown(None) == ""

    def test_clean_title_accepts_none(self):
        assert tg._clean_title(None) is None

    def test_clean_title_drops_trailing_punctuation(self):
        assert tg._clean_title("LeJEPA Papers.") == "LeJEPA Papers"

    def test_choice_text_reads_null_content(self):
        content, finish = tg._choice_text([_choice(None, "length")])
        assert content is None
        assert finish == "length"

    def test_choice_text_survives_none_choice(self):
        assert tg._choice_text([None]) == (None, None)
        assert tg._choice_text(None) == (None, None)


@pytest.mark.unit
@pytest.mark.asyncio
class TestOpenAITitleMode:
    async def test_uses_a_budget_large_enough_for_reasoning(self, fake_aiohttp):
        sessions = fake_aiohttp([_payload([_choice("LeJEPA and IJEPA Study")])])

        title = await tg._generate_title_openai_mode(
            PROMPT, "deepseek/deepseek-v4.1-flash-1017", "http://x", "k", 16384
        )

        assert title == "LeJEPA and IJEPA Study"
        assert sessions[0].posts[0]["max_tokens"] == tg.TITLE_MAX_TOKENS
        assert tg.TITLE_MAX_TOKENS >= 256

    async def test_retries_when_reasoning_ate_the_budget(self, fake_aiohttp):
        sessions = fake_aiohttp(
            [
                _payload([_choice(None, "length")]),
                _payload([_choice("LeJEPA and IJEPA Study")]),
            ]
        )

        title = await tg._generate_title_openai_mode(
            PROMPT, "deepseek/deepseek-v4.1-flash-1017", "http://x", "k", 16384
        )

        assert title == "LeJEPA and IJEPA Study"
        budgets = [s.posts[0]["max_tokens"] for s in sessions]
        assert budgets == [tg.TITLE_MAX_TOKENS, tg.TITLE_RETRY_MAX_TOKENS]

    async def test_no_retry_when_the_model_simply_refused(self, fake_aiohttp):
        sessions = fake_aiohttp([_payload([_choice(None, "stop")])])

        title = await tg._generate_title_openai_mode(
            PROMPT, "m", "http://x", "k", 16384
        )

        assert title is None
        assert len(sessions) == 1

    async def test_gives_up_after_the_retry(self, fake_aiohttp):
        sessions = fake_aiohttp(
            [_payload([_choice(None, "length")]), _payload([_choice(None, "length")])]
        )

        title = await tg._generate_title_openai_mode(
            PROMPT, "m", "http://x", "k", 16384
        )

        assert title is None
        assert len(sessions) == 2

    async def test_unconfigured_provider_falls_back_without_a_request(
        self, monkeypatch
    ):
        """A model that resolves to an unconfigured provider must not wait on it."""
        from tensortruth.core.providers import ModelReference, ProviderRegistry

        calls = []

        def fake_resolve(params, model_name):
            return ModelReference(
                provider_id="ollama",
                model_name=model_name,
                display_name=model_name,
                provider_type="ollama",
                base_url="http://localhost:11434",
            )

        monkeypatch.setattr(
            "tensortruth.core.providers.resolve_model_from_params", fake_resolve
        )

        class _RefusingSession:
            def __init__(self, *a, **k):
                calls.append("session")

            def post(self, *a, **k):
                raise AssertionError("no request should be attempted")

        monkeypatch.setattr(tg.aiohttp, "ClientSession", _RefusingSession)
        monkeypatch.setattr(
            ProviderRegistry,
            "get_instance",
            classmethod(
                lambda cls: type("R", (), {"get_provider": lambda self, p: None})()
            ),
        )

        title = await tg.generate_smart_title_async(PROMPT, "missing/model", 16384)

        assert calls == []
        assert title.startswith("Im studying LeJEPA paper")

    async def test_null_content_never_raises_and_falls_back(self, monkeypatch):
        """The regression: a null completion must yield a fallback, not an error."""

        class _NullSession:
            def __init__(self, *a, **k):
                pass

            def post(self, *a, **k):
                return _FakeResponse(_payload([_choice(None, "length")]))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        monkeypatch.setattr(tg.aiohttp, "ClientSession", _NullSession)

        title = await tg.generate_smart_title_async(
            PROMPT, "deepseek/deepseek-v4.1-flash-1017", 16384, provider_id="openrouter"
        )

        # Falls back to the prompt-derived title instead of raising.
        assert title.startswith("Im studying LeJEPA paper")
