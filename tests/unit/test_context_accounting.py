"""Tests for per-category context accounting."""

from types import SimpleNamespace

import pytest

from tensortruth.services import context_accounting as acc


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    """Isolate module state and persistence from the real ~/.tensortruth."""
    monkeypatch.setattr(acc, "_STATE_PATH", tmp_path / "context_snapshots.json")
    acc.clear_all()
    acc._static_overhead.clear()
    acc._loaded = False
    yield
    acc.clear_all()
    acc._static_overhead.clear()
    acc._loaded = False


@pytest.mark.unit
class TestSnapshots:
    def test_record_and_read_back(self):
        acc.record_snapshot(
            "s1",
            {
                "system": 4000,
                "tools": 8000,
                "modules": 400,
                "documents": 0,
                "history": 2000,
                "prompt": 40,
            },
            context_window=65536,
            model_name="m",
            messages_included=4,
            messages_total=9,
        )
        report = acc.report_for_session("s1")
        assert report is not None
        assert report["measured"] is True
        assert report["total_chars"] == 14440
        assert report["total_tokens"] == 14440 // 4
        assert report["messages_included"] == 4
        assert report["messages_total"] == 9
        assert report["used_percent"] == pytest.approx((14440 // 4) / 65536 * 100)

    def test_categories_are_ordered_and_shares_sum_to_one(self):
        acc.record_snapshot("s1", {"system": 100, "tools": 300, "history": 100})
        report = acc.report_for_session("s1")
        ids = [c["id"] for c in report["categories"]]
        assert ids == acc.CATEGORIES
        assert sum(c["share"] for c in report["categories"]) == pytest.approx(1.0)
        assert sum(c["chars"] for c in report["categories"]) == report["total_chars"]

    def test_unmeasured_session_has_no_report(self):
        assert acc.report_for_session("nope") is None

    def test_negative_counts_clamped(self):
        acc.record_snapshot("s1", {"system": -50, "tools": 10})
        report = acc.report_for_session("s1")
        system = next(c for c in report["categories"] if c["id"] == "system")
        assert system["chars"] == 0

    def test_static_overhead_is_recorded_separately(self):
        assert acc.get_static_overhead() == {}
        acc.record_static_overhead({"system": 500, "tools": 900, "modules": 100})
        assert acc.get_static_overhead() == {
            "system": 500,
            "tools": 900,
            "modules": 100,
        }

    def test_snapshots_do_not_clobber_static_overhead(self):
        """A fast-path turn records tools=0 but must not zero the overhead."""
        acc.record_static_overhead({"system": 500, "tools": 900, "modules": 100})
        acc.record_snapshot("s1", {"system": 500, "tools": 0, "prompt": 40})
        assert acc.get_static_overhead()["tools"] == 900

    def test_clear_snapshot(self):
        acc.record_snapshot("s1", {"system": 1})
        acc.clear_snapshot("s1")
        assert acc.report_for_session("s1") is None

    def test_survives_a_restart(self, tmp_path, monkeypatch):
        """Snapshots and static overhead are persisted, then reloaded."""
        acc.record_static_overhead({"system": 500, "tools": 900, "modules": 0})
        acc.record_snapshot(
            "s1",
            {"system": 500, "tools": 900, "history": 400},
            context_window=100_000,
            messages_included=4,
            messages_total=6,
        )
        assert acc._STATE_PATH.exists()

        # Simulate a fresh process: wipe memory, keep the file.
        acc._snapshots.clear()
        acc._static_overhead.clear()
        acc._loaded = False

        report = acc.report_for_session("s1")
        assert report is not None and report["measured"] is True
        assert report["context_window"] == 100_000
        assert report["messages_included"] == 4
        assert acc.get_static_overhead()["tools"] == 900

    def test_cache_is_bounded(self):
        for i in range(acc._MAX_SNAPSHOTS + 20):
            acc.record_snapshot(f"s{i}", {"system": i})
        assert len(acc._snapshots) == acc._MAX_SNAPSHOTS
        # newest survives
        assert acc.report_for_session(f"s{acc._MAX_SNAPSHOTS + 19}") is not None


@pytest.mark.unit
class TestReport:
    def test_no_window_means_no_percentage(self):
        report = acc.build_report({"system": 40}, context_window=0)
        assert report["used_percent"] == 0.0
        assert report["remaining_tokens"] is None

    def test_remaining_tokens_never_negative(self):
        report = acc.build_report({"system": 40_000}, context_window=1000)
        assert report["remaining_tokens"] == 0


@pytest.mark.unit
class TestToolSchemaChars:
    def _tool(self, name, description, schema=None):
        fn_schema = None
        if schema is not None:

            class _Schema:
                @staticmethod
                def model_json_schema():
                    return schema

            fn_schema = _Schema()
        return SimpleNamespace(
            metadata=SimpleNamespace(
                name=name, description=description, fn_schema=fn_schema
            )
        )

    def test_counts_names_descriptions_and_schemas(self):
        tools = [
            self._tool("rag_query", "Query the index", {"properties": {"q": {}}}),
            self._tool("web_search", "Search the web"),
        ]
        chars = acc.tool_schema_chars(tools)
        assert chars > len("rag_queryQuery the indexweb_searchSearch the web")
        assert chars < 400

    def test_tolerates_objects_without_metadata(self):
        assert acc.tool_schema_chars([object()]) == 0
