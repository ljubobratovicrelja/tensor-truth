"""Tests for reasoning-effort level probing (core/reasoning_levels)."""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.core import reasoning_levels as rl


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    """Point the cache at a temp file for every test."""
    monkeypatch.setenv("TENSORTRUTH_REASONING_CACHE", str(tmp_path / "levels.json"))
    return tmp_path / "levels.json"


def _ok_response(reasoning: bool = False):
    resp = MagicMock()
    resp.status_code = 200
    payload = {"choices": [{"message": {"content": "OK"}}], "usage": {}}
    if reasoning:
        payload["choices"][0]["message"]["reasoning"] = "thinking..."
        payload["usage"] = {"completion_tokens_details": {"reasoning_tokens": 12}}
    resp.json.return_value = payload
    resp.text = ""
    return resp


def _err(status: int, text: str):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.json.side_effect = ValueError("not json")
    return resp


class TestProbeReasoningLevels:
    def test_all_levels_accepted(self):
        with patch("requests.post", return_value=_ok_response()) as post:
            result = rl.probe_reasoning_levels("http://x/v1", "k", "m")

        assert result["baseline_ok"] is True
        assert result["levels"] == list(rl.CANDIDATE_LEVELS)
        # baseline + one request per candidate
        assert post.call_count == 1 + len(rl.CANDIDATE_LEVELS)

    def test_rejected_level_is_dropped(self):
        def _post(url, headers=None, json=None, timeout=None):
            if json.get("reasoning_effort") == "high":
                return _err(400, '{"error": "unsupported_value: reasoning_effort"}')
            return _ok_response(reasoning=True)

        with patch("requests.post", side_effect=_post):
            result = rl.probe_reasoning_levels("http://x/v1", None, "m")

        assert result["levels"] == ["none", "minimal", "low", "medium"]
        assert result["reasoning"] is True

    def test_unrelated_error_stops_probe(self):
        def _post(url, headers=None, json=None, timeout=None):
            if json.get("reasoning_effort") == "minimal":
                return _err(429, "rate limited")
            return _ok_response()

        with patch("requests.post", side_effect=_post):
            result = rl.probe_reasoning_levels("http://x/v1", "k", "m")

        # Baseline + "none" succeeded, then the probe bailed out.
        assert result["levels"] == ["none"]

    def test_unreachable_model_reports_no_levels(self):
        with patch("requests.post", side_effect=ConnectionError("boom")):
            result = rl.probe_reasoning_levels("http://x/v1", "k", "m")

        assert result["baseline_ok"] is False
        assert result["levels"] == []

    def test_reasoning_flag_from_usage_details(self):
        with patch("requests.post", return_value=_ok_response(reasoning=True)):
            result = rl.probe_reasoning_levels("http://x/v1", "k", "m")
        assert result["reasoning"] is True


class TestCache:
    def test_probe_and_cache_then_reuse(self):
        with patch("requests.post", return_value=_ok_response()) as post:
            first = rl.probe_and_cache("prov", "http://x/v1", "k", "model-a")
            calls = post.call_count
            second = rl.probe_and_cache("prov", "http://x/v1", "k", "model-a")

        assert first["cached"] is False
        assert second["cached"] is True
        assert post.call_count == calls  # no extra requests on cache hit
        assert second["levels"] == first["levels"]

    def test_force_refresh_reprobes(self):
        with patch("requests.post", return_value=_ok_response()) as post:
            rl.probe_and_cache("prov", "http://x/v1", "k", "model-a")
            calls = post.call_count
            rl.probe_and_cache("prov", "http://x/v1", "k", "model-a", force=True)

        assert post.call_count == 2 * calls

    def test_stale_cache_is_ignored(self, monkeypatch):
        with patch("requests.post", return_value=_ok_response()):
            rl.probe_and_cache("prov", "http://x/v1", "k", "model-a")

        monkeypatch.setattr(rl, "CACHE_TTL_SECONDS", -1)
        assert rl.get_cached("prov", "model-a") is None


class TestSelectableLevels:
    def test_none_is_hidden_from_ui(self):
        probe = {"levels": ["none", "low", "high"], "reasoning": True}
        assert rl.selectable_levels(probe) == ["low", "high"]

    def test_empty_when_nothing_accepted(self):
        assert rl.selectable_levels({"levels": [], "reasoning": True}) == []
