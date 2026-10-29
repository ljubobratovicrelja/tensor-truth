"""Chunk spans must not drift when node text and root text differ in newlines."""

import json
import pytest

from tensortruth.services.index_management_service import IndexManagementService

DOC_ID = "doc-crlf"


@pytest.fixture()
def service(tmp_path):
    svc = IndexManagementService()
    svc._indexes_dir = str(tmp_path)
    return svc


@pytest.mark.unit
def test_span_ends_on_a_word_boundary_with_crlf_root(service, tmp_path):
    """Root text keeps CRLF, node text is LF; arithmetic ends cut mid-word."""
    para = (
        "For any computing technology to reach its full impact, it must be well "
        "understood. Our goal in this book is to present a unified resource to "
        "bring would-be practitioners up to speed."
    )
    root = f"# Preface\r\n\r\n{para}\r\n\r\nNext paragraph follows here.\r\n"
    index_dir = tmp_path / "bge-m3" / "book_x"
    index_dir.mkdir(parents=True)
    (index_dir / "docstore.json").write_text(
        json.dumps(
            {
                "docstore/ref_doc_info": {
                    DOC_ID: {"node_ids": ["sec1", "leaf1"], "metadata": {}}
                },
                "docstore/data": {
                    DOC_ID: {
                        "__data__": {
                            "text": root,
                            "metadata": {},
                            "relationships": {"5": [{"node_id": "sec1"}]},
                        }
                    },
                    "sec1": {
                        "__data__": {
                            "text": f"Section: Preface\n\n{para}",
                            "metadata": {},
                            "relationships": {
                                "4": [{"node_id": DOC_ID}],
                                "5": [{"node_id": "leaf1"}],
                            },
                        }
                    },
                    "leaf1": {
                        "__data__": {
                            "text": f"Section: Preface\n\n{para}",
                            "metadata": {},
                            "relationships": {"4": [{"node_id": "sec1"}]},
                        }
                    },
                },
            }
        )
    )

    content = service.get_document_content("bge-m3", "book_x", DOC_ID)
    leaf = content["chunks"][1]
    sliced = content["text"][leaf["start"] : leaf["end"]]

    assert sliced.endswith("up to speed.")
    assert sliced.startswith("For any computing technology")
