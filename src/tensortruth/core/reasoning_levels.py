"""Probe OpenAI-compatible models for the reasoning-effort levels they accept.

The chat UI offers a "Think" dropdown. For openai_compatible providers the
actionable knob is ``reasoning_effort`` (sent as a top-level request field).
Which values a given endpoint/model accepts is not advertised anywhere, so we
ask the endpoint: send a tiny completion for each candidate value and keep the
ones that do not come back as a client error.

Two signals are recorded per model:

- ``levels``: candidate values the endpoint accepted (HTTP 200).
- ``reasoning``: whether the response actually carried reasoning tokens,
  i.e. the model produces reasoning at all.

Results are cached on disk per (provider, model) so the dropdown does not
re-probe on every page load. Probing is deliberately lazy: only the model the
user selected is probed, never the whole catalogue (300+ models on OpenRouter).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Candidate values, ordered from least to most effort. "none" lets the UI's
# "Off" map to an explicit "no reasoning" request on endpoints that support it.
CANDIDATE_LEVELS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high")

# Levels offered to the user (excludes "none", which the UI shows as "Off").
SELECTABLE_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high")

CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
PROBE_TIMEOUT = 30
PROBE_MAX_TOKENS = 24


def _cache_path() -> Path:
    override = os.environ.get("TENSORTRUTH_REASONING_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".tensortruth" / "reasoning_levels.json"


def _load_cache() -> Dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read reasoning-level cache %s: %s", path, e)
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        logger.warning("Could not write reasoning-level cache %s: %s", path, e)


def _cache_key(provider_id: str, model_name: str) -> str:
    return f"{provider_id}/{model_name}"


def get_cached(provider_id: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Return cached probe data for a model, or None if missing/stale."""
    entry = _load_cache().get(_cache_key(provider_id, model_name))
    if not isinstance(entry, dict):
        return None
    checked_at = entry.get("checked_at") or 0
    if time.time() - float(checked_at) > CACHE_TTL_SECONDS:
        return None
    return entry


def _is_unsupported_response(status: int, body: str) -> bool:
    """Best-effort classification of "this endpoint rejected the value"."""
    if status in (400, 404, 422):
        text = (body or "").lower()
        # Only treat it as a value rejection when the param is implicated;
        # otherwise it could be a quota/model error and says nothing about
        # reasoning support.
        return any(
            needle in text
            for needle in (
                "reasoning_effort",
                "reasoning effort",
                "reasoning",
                "unsupported_value",
                "invalid value",
            )
        )
    return False


def probe_reasoning_levels(
    base_url: str,
    api_key: Optional[str],
    model_name: str,
    candidates: tuple[str, ...] = CANDIDATE_LEVELS,
    timeout: int = PROBE_TIMEOUT,
) -> Dict[str, Any]:
    """Ping the endpoint once per candidate level and report what it takes.

    Args:
        base_url: OpenAI-compatible base URL (e.g. ``https://.../api/v1``).
        api_key: Bearer token, or None/empty for keyless local servers.
        model_name: Model id as the endpoint knows it.
        candidates: Values to try.
        timeout: Per-request timeout in seconds.

    Returns:
        ``{"levels": [...], "reasoning": bool, "baseline_ok": bool,
        "checked_at": <epoch>, "model": name}``. ``levels`` only contains
        values the endpoint accepted; ``reasoning`` reflects whether the
        baseline response carried reasoning tokens.
    """
    import requests

    base = (base_url or "").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    accepted: List[str] = []
    reasoning = False

    def _post(level: Optional[str]) -> tuple[bool, int, str, bool]:
        body: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": PROBE_MAX_TOKENS,
            "stream": False,
        }
        if level is not None:
            body["reasoning_effort"] = level
        try:
            resp = requests.post(
                f"{base}/chat/completions", headers=headers, json=body, timeout=timeout
            )
        except Exception as e:  # network/timeout
            logger.info("Reasoning probe failed for %s: %s", model_name, e)
            return False, 0, str(e), False

        if resp.status_code != 200:
            return False, resp.status_code, resp.text[:400], False

        try:
            payload = resp.json()
        except ValueError:
            return True, 200, "", False

        message = (payload.get("choices") or [{}])[0].get("message") or {}
        usage = payload.get("usage") or {}
        has_reasoning = bool(
            message.get("reasoning")
            or message.get("reasoning_content")
            or (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            or usage.get("reasoning_tokens")
        )
        return True, 200, "", has_reasoning

    baseline_ok, _, _, reasoning = _post(None)
    if not baseline_ok:
        # The model itself is unreachable/unusable — report nothing usable.
        return {
            "model": model_name,
            "levels": [],
            "reasoning": False,
            "baseline_ok": False,
            "checked_at": time.time(),
        }

    for level in candidates:
        ok, status, body, _ = _post(level)
        if ok:
            accepted.append(level)
        elif status and not _is_unsupported_response(status, body):
            # A non-parameter error (rate limit, auth, server) — stop guessing.
            logger.info(
                "Reasoning probe for %s stopped at '%s' (HTTP %s)",
                model_name,
                level,
                status,
            )
            break

    return {
        "model": model_name,
        "levels": accepted,
        "reasoning": reasoning,
        "baseline_ok": True,
        "checked_at": time.time(),
    }


def probe_and_cache(
    provider_id: str,
    base_url: str,
    api_key: Optional[str],
    model_name: str,
    force: bool = False,
) -> Dict[str, Any]:
    """Probe (or reuse the cached result) and persist it."""
    if not force:
        cached = get_cached(provider_id, model_name)
        if cached is not None:
            return {**cached, "cached": True}

    result = probe_reasoning_levels(base_url, api_key, model_name)
    cache = _load_cache()
    cache[_cache_key(provider_id, model_name)] = result
    _save_cache(cache)
    return {**result, "cached": False}


def selectable_levels(probe: Dict[str, Any]) -> List[str]:
    """Filter a probe result down to values the UI should offer.

    A level is offered when the endpoint accepted it. ``reasoning`` is reported
    separately: many gateways (OpenRouter, LiteLLM) accept ``reasoning_effort``
    for every model and silently ignore it where it does not apply, and some
    reasoning models expose no reasoning tokens unless asked, so a
    "reasoning tokens seen" gate would hide valid options.
    """
    accepted = set(probe.get("levels") or [])
    return [lvl for lvl in SELECTABLE_LEVELS if lvl in accepted]


def levels_for_model(provider_id: str, model_name: str) -> List[str]:
    """Cached, UI-ready levels for a model (no network). Empty when unknown."""
    cached = get_cached(provider_id, model_name)
    if cached is None:
        return []
    return selectable_levels(cached)
