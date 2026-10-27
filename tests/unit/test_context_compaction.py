"""Tests for conversation compaction (message surgery, not the LLM call)."""

import json

import pytest

from tensortruth.services import context_compaction as comp


def _msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


@pytest.mark.unit
class TestPlan:
    def test_nothing_to_do_when_short(self):
        messages = [_msg("user", "hi"), _msg("assistant", "hello")]
        assert comp.plan_compaction(messages, keep_last=6) is None

    def test_splits_older_and_recent(self):
        messages = [_msg("user", f"m{i}") for i in range(10)]
        plan = comp.plan_compaction(messages, keep_last=4)
        assert len(plan["older"]) == 6
        assert len(plan["recent"]) == 4
        assert plan["recent"][-1]["content"] == "m9"

    def test_blank_older_messages_are_not_compacted(self):
        messages = [_msg("user", "   ") for _ in range(8)] + [_msg("user", "x")]
        assert comp.plan_compaction(messages, keep_last=1) is None

    def test_transcript_is_role_labelled_and_capped(self):
        messages = [_msg("user", "y" * 10_000)]
        plan = comp.plan_compaction(
            messages + [_msg("assistant", "ok")] * 2, keep_last=2
        )
        transcript = comp.plan_compaction(
            messages + [_msg("user", "a")] * 2, keep_last=2
        )["transcript"]
        assert transcript.startswith("USER: yyy")
        assert "[truncated]" in transcript
        assert plan is not None


@pytest.mark.unit
class TestApply:
    def test_summary_prepended_and_tail_kept(self):
        messages = [_msg("user", f"m{i}") for i in range(8)]
        new = comp.apply_compaction(messages, "We decided X.", keep_last=3)
        assert len(new) == 4
        assert new[0]["is_compaction"] is True
        assert new[0]["compacted_messages"] == 5
        assert new[0]["content"].startswith(comp.SUMMARY_HEADER)
        assert "We decided X." in new[0]["content"]
        assert [m["content"] for m in new[1:]] == ["m5", "m6", "m7"]

    def test_no_op_when_nothing_to_do(self):
        messages = [_msg("user", "a")]
        assert comp.apply_compaction(messages, "ignored", keep_last=6) == messages

    def test_second_compaction_folds_previous_summary(self):
        messages = [_msg("user", f"m{i}") for i in range(6)]
        once = comp.apply_compaction(messages, "first summary", keep_last=2)
        more = once + [_msg("user", "later") for _ in range(4)]
        twice = comp.apply_compaction(more, "second summary", keep_last=2)
        # only one summary message survives, and it is the new one
        summaries = [m for m in twice if m.get("is_compaction")]
        assert len(summaries) == 1
        assert "second summary" in summaries[0]["content"]


@pytest.mark.unit
class TestSummarize:
    def test_short_transcript_is_single_call(self):
        calls = []

        def _summarize(text):
            calls.append(text)
            return "summary"

        assert comp.summarize_transcript(_summarize, "short") == "summary"
        assert len(calls) == 1

    def test_long_transcript_is_chunked_and_chained(self):
        calls = []

        def _summarize(text):
            calls.append(text)
            return f"summary{len(calls)}"

        transcript = "\n\n".join(f"USER: {'z' * 900}" for _ in range(200))
        result = comp.summarize_transcript(_summarize, transcript)
        assert len(calls) > 1
        assert "Summary so far" in calls[1]
        assert result == f"summary{len(calls)}"

    def test_empty_summary_raises(self):
        with pytest.raises(ValueError):
            comp.summarize_transcript(lambda _t: "   ", "text")

    def test_split_respects_size(self):
        text = "\n\n".join("a" * 100 for _ in range(10))
        chunks = comp.split_into_chunks(text, size=250)
        assert len(chunks) > 1
        assert all(len(c) <= 300 for c in chunks)


@pytest.mark.unit
class TestBackup:
    def test_writes_recoverable_json(self, tmp_path):
        messages = [_msg("user", f"m{i}") for i in range(8)]
        plan = comp.plan_compaction(messages, keep_last=2)
        path = comp.backup_replaced_messages(tmp_path, "sess-1", plan, "the summary")
        assert path is not None and path.exists()
        payload = json.loads(path.read_text())
        assert payload["session_id"] == "sess-1"
        assert payload["summary"] == "the summary"
        assert len(payload["replaced_messages"]) == 6
