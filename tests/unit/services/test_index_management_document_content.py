"""Document-content reconstruction for the index-management viewer."""

import json
from pathlib import Path

import pytest

from tensortruth.services.index_management_service import IndexManagementService

DOC_ID = "chapter_01_Preface"
ROOT_TEXT = "Preface\n\nBody one. Body two.\n\nMore body."


def _node(text: str, **relationships: list[str]) -> dict:
    return {
        "__data__": {
            "text": text,
            "metadata": {},
            "relationships": {
                key: [{"node_id": nid} for nid in ids]
                for key, ids in relationships.items()
            },
        }
    }


def _write_store(tmp_path: Path, docstore: dict) -> Path:
    index_dir = tmp_path / "bge-m3" / "book_x"
    index_dir.mkdir(parents=True)
    (index_dir / "docstore.json").write_text(json.dumps(docstore))
    return index_dir


@pytest.fixture()
def service(tmp_path):
    svc = IndexManagementService()
    svc._indexes_dir = str(tmp_path)
    return svc


@pytest.mark.unit
def test_structure_index_renders_root_text_without_char_offsets(service, tmp_path):
    """Structure-parsed nodes carry no start_char_idx; the viewer used to 500."""
    _write_store(
        tmp_path,
        {
            "docstore/ref_doc_info": {
                DOC_ID: {
                    "node_ids": ["sec1", "leaf1", "leaf2"],
                    "metadata": {"file_name": "chapter_01_Preface.md"},
                }
            },
            "docstore/data": {
                DOC_ID: _node(ROOT_TEXT, **{"5": ["sec1"]}),
                "sec1": _node(
                    "Section: Preface\n\nBody one. Body two.\n\nMore body.",
                    **{"4": [DOC_ID], "5": ["leaf1", "leaf2"]},
                ),
                "leaf1": _node(
                    "Section: Preface\n\nBody one. Body two.", **{"4": ["sec1"]}
                ),
                "leaf2": _node("Section: Preface\n\nMore body.", **{"4": ["sec1"]}),
            },
        },
    )

    content = service.get_document_content("bge-m3", "book_x", DOC_ID)

    assert content["text"] == ROOT_TEXT
    assert content["metadata"]["file_name"] == "chapter_01_Preface.md"
    spans = content["chunks"]
    assert [c["chunk_id"] for c in spans] == ["sec1", "leaf1", "leaf2"]
    assert [c["level"] for c in spans] == [0, 1, 1]
    assert all(0 <= c["start"] <= c["end"] <= len(ROOT_TEXT) for c in spans)
    assert spans == sorted(spans, key=lambda c: c["start"])


@pytest.mark.unit
def test_legacy_index_with_offsets_merges_overlapping_chunks(service, tmp_path):
    _write_store(
        tmp_path,
        {
            "docstore/ref_doc_info": {"-1": {"node_ids": ["n1", "n2"], "metadata": {}}},
            "docstore/data": {
                "n1": {
                    "__data__": {
                        "text": "alpha beta gamma",
                        "start_char_idx": 0,
                        "end_char_idx": 16,
                        "metadata": {},
                        "relationships": {},
                    }
                },
                "n2": {
                    "__data__": {
                        "text": "beta gamma delta",
                        "start_char_idx": 6,
                        "end_char_idx": 22,
                        "metadata": {},
                        "relationships": {},
                    }
                },
            },
        },
    )

    content = service.get_document_content("bge-m3", "book_x", "-1")

    assert content["text"] == "alpha beta gamma delta"
    assert [c["chunk_id"] for c in content["chunks"]] == ["n1", "n2"]
