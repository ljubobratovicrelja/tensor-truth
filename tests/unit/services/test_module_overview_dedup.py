"""Module overview must not duplicate a module across its name spellings."""

import json

import pytest

from tensortruth.services.index_management_service import IndexManagementService


@pytest.fixture()
def service(tmp_path):
    svc = IndexManagementService()
    svc._indexes_dir = str(tmp_path / "indexes")
    svc._library_docs_dir = str(tmp_path / "library_docs")
    svc._sources_config_path = tmp_path / "sources.json"
    (tmp_path / "indexes" / "bge-m3" / "book_x").mkdir(parents=True)
    (tmp_path / "indexes" / "bge-m3" / "book_x" / "chroma.sqlite3").write_text("")
    (tmp_path / "library_docs" / "book_x").mkdir(parents=True)
    (tmp_path / "library_docs" / "book_x" / "chapter_01.md").write_text("# t\n\nbody\n")
    svc._sources_config_path.write_text(
        json.dumps(
            {
                "books": {"x": {"title": "X", "authors": ["A"]}},
                "papers": {},
                "libraries": {},
            }
        )
    )
    return svc


@pytest.mark.unit
def test_prefixed_index_and_bare_source_are_one_module(service):
    modules = service.get_modules_overview("bge-m3")

    x_modules = [m for m in modules if m["name"].endswith("x") or m["name"] == "book_x"]
    assert len(x_modules) == 1, [m["name"] for m in x_modules]
    only = x_modules[0]
    assert only["name"] == "book_x"
    assert only["has_index"] is True
