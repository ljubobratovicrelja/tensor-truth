import json

from tensortruth.services.index_management_service import IndexManagementService

DOC = "doc-head"


def test_span_includes_heading_chain(tmp_path):
    root = "# A\n## B\n\nFirst paragraph of B.\n\nSecond paragraph of B.\n"
    leaf_text = "# A\n## B\n\nFirst paragraph of B."
    index_dir = tmp_path / "bge-m3" / "book_x"
    index_dir.mkdir(parents=True)
    (index_dir / "docstore.json").write_text(
        json.dumps(
            {
                "docstore/ref_doc_info": {DOC: {"node_ids": ["leaf1"], "metadata": {}}},
                "docstore/data": {
                    DOC: {
                        "__data__": {
                            "text": root,
                            "metadata": {},
                            "relationships": {"5": [{"node_id": "leaf1"}]},
                        }
                    },
                    "leaf1": {
                        "__data__": {
                            "text": leaf_text,
                            "metadata": {},
                            "relationships": {"4": [{"node_id": DOC}]},
                        }
                    },
                },
            }
        )
    )
    svc = IndexManagementService()
    svc._indexes_dir = str(tmp_path)
    content = svc.get_document_content("bge-m3", "book_x", DOC)
    span = content["chunks"][0]
    sliced = content["text"][span["start"] : span["end"]]
    assert sliced.startswith("# A\n## B\n\nFirst paragraph of B.")
