"""Tests for the indexed module catalog (services/module_catalog.py)."""

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from tensortruth.services.module_catalog import (
    clear_module_catalog_cache,
    load_module_catalog,
)

MODEL_ID = "bge-m3"


def _make_index(
    root,
    module: str,
    *,
    doc_type: str,
    display_name: str = "",
    group_display_name: str = "",
    book_display_name: str = "",
    library_display_name: str = "",
    doc_ids: int = 3,
) -> None:
    """Create a minimal Chroma-shaped index (sqlite only) for one module."""
    module_dir = root / MODEL_ID / module
    module_dir.mkdir(parents=True)
    with sqlite3.connect(module_dir / "chroma.sqlite3") as conn:
        conn.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT)"
        )
        metadata = {"doc_type": doc_type}
        if display_name:
            metadata["display_name"] = display_name
        if group_display_name:
            metadata["group_display_name"] = group_display_name
        if book_display_name:
            metadata["book_display_name"] = book_display_name
        if library_display_name:
            metadata["library_display_name"] = library_display_name
        for key, value in metadata.items():
            conn.execute(
                "INSERT INTO embedding_metadata VALUES (1, ?, ?)", (key, value)
            )
        for doc_index in range(doc_ids):
            conn.execute("INSERT INTO embeddings (id) VALUES (?)", (doc_index + 1,))
            conn.execute(
                "INSERT INTO embedding_metadata VALUES (?, 'doc_id', ?)",
                (doc_index + 1, f"doc-{doc_index}"),
            )
        conn.commit()


def _config():
    config = MagicMock()
    config.rag.default_embedding_model = "BAAI/bge-m3"
    return config


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_module_catalog_cache()
    yield
    clear_module_catalog_cache()


def test_descriptions_resolved_from_sources_json(tmp_path):
    """papers/ and book_ modules take the description declared in sources.json."""
    indexes = tmp_path / "indexes"
    _make_index(
        indexes,
        "papers_dl_architectures",
        doc_type="paper",
        group_display_name="DL Architectures",
        doc_ids=4,
    )
    _make_index(
        indexes,
        "book_linear_algebra_cherney",
        doc_type="book",
        book_display_name="Linear Algebra",
        doc_ids=5,
    )
    sources = tmp_path / "sources.json"
    sources.write_text(
        json.dumps(
            {
                "papers": {
                    "dl_architectures": {
                        "description": "CNN/Transformer papers and optimization"
                    }
                },
                "books": {
                    "linear_algebra_cherney": {
                        "description": "Free UC Davis linear algebra textbook"
                    }
                },
            }
        )
    )

    catalog = {
        entry.name: entry
        for entry in load_module_catalog(
            _config(), indexes_dir=indexes, sources_path=sources
        )
    }

    assert (
        catalog["papers_dl_architectures"].description
        == "CNN/Transformer papers and optimization"
    )
    assert (
        catalog["book_linear_algebra_cherney"].description
        == "Free UC Davis linear algebra textbook"
    )
    assert catalog["papers_dl_architectures"].doc_count == 4
    assert catalog["book_linear_algebra_cherney"].doc_count == 5
    assert catalog["papers_dl_architectures"].doc_type == "paper"


def test_library_description_is_derived_when_sources_has_none(tmp_path):
    """Libraries carry no description in sources.json, so one is derived."""
    indexes = tmp_path / "indexes"
    _make_index(
        indexes,
        "library_pytorch_2.9",
        doc_type="library_doc",
        library_display_name="Pytorch 2.9",
        doc_ids=7,
    )
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"libraries": {"pytorch_2.9": {"version": "2.9"}}}))

    catalog = load_module_catalog(_config(), indexes_dir=indexes, sources_path=sources)

    assert len(catalog) == 1
    entry = catalog[0]
    assert entry.doc_type == "library_doc"
    assert entry.description == "Pytorch 2.9 library documentation"
    assert entry.doc_count == 7


def test_catalog_is_cached_between_calls(tmp_path, monkeypatch):
    """The second call must not re-open the (expensive) index metadata."""
    import tensortruth.services.module_catalog as catalog_mod

    indexes = tmp_path / "indexes"
    _make_index(indexes, "book_a", doc_type="book", doc_ids=2)

    calls = []
    real_reader = catalog_mod._read_index_metadata

    def counting_reader(path):
        calls.append(path)
        return real_reader(path)

    monkeypatch.setattr(catalog_mod, "_read_index_metadata", counting_reader)

    first = load_module_catalog(_config(), indexes_dir=indexes, sources_path=None)
    second = load_module_catalog(_config(), indexes_dir=indexes, sources_path=None)

    assert [e.name for e in first] == [e.name for e in second]
    assert len(calls) == 1


def test_missing_indexes_dir_returns_empty(tmp_path):
    """No indexes on disk is an empty catalog, never an exception."""
    catalog = load_module_catalog(
        _config(),
        indexes_dir=tmp_path / "missing",
        sources_path=tmp_path / "missing.json",
    )
    assert catalog == []


def test_unreadable_index_degrades_to_bare_entry(tmp_path):
    """A corrupt index still yields an entry with the module name."""
    indexes = tmp_path / "indexes"
    module_dir = indexes / MODEL_ID / "library_broken"
    module_dir.mkdir(parents=True)
    (module_dir / "chroma.sqlite3").write_bytes(b"not a database")

    catalog = load_module_catalog(
        _config(), indexes_dir=indexes, sources_path=tmp_path / "none.json"
    )

    assert len(catalog) == 1
    assert catalog[0].name == "library_broken"
    assert catalog[0].doc_type == "unknown"
    assert catalog[0].doc_count == 0
