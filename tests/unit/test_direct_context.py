"""Tests for direct-to-context document attachments (no vector index)."""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.services.context_documents import (
    CHARS_PER_TOKEN,
    allocate_budget,
    build_context_block,
    strip_document_header,
)
from tensortruth.services.document_service import DocumentService
from tensortruth.services.orchestrator_service import (
    CONTEXT_BUDGET_PCT,
    OrchestratorService,
)

UPLOAD_HEADER = "# Document: paper.pdf\n# Source: Session Upload\n\n---\n\n"


# ---------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------


class TestAllocateBudget:
    def test_budget_is_never_exceeded(self):
        sizes = [10_000, 10_000, 10_000]
        allocation = allocate_budget(sizes, 9000)
        assert sum(allocation) <= 9000
        assert all(a > 0 for a in allocation)

    def test_small_documents_get_full_text_and_leftover_goes_to_large(self):
        sizes = [100, 100, 100_000]
        allocation = allocate_budget(sizes, 10_000)
        assert allocation[0] == 100
        assert allocation[1] == 100
        # Remaining ~9,800 chars all go to the large document
        assert allocation[2] > 9_000
        assert sum(allocation) <= 10_000

    def test_zero_budget_allocates_nothing(self):
        assert allocate_budget([10, 20], 0) == [0, 0]

    def test_exact_fit(self):
        assert allocate_budget([500, 500], 1000) == [500, 500]


# ---------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------


class TestBuildContextBlock:
    def test_renders_labels_and_full_text(self):
        docs = [
            {"doc_id": "doc_aaa", "filename": "paper.pdf", "text": "Alpha body"},
            {"doc_id": "doc_bbb", "filename": "notes.md", "text": "Beta body"},
        ]
        block, report = build_context_block(docs, 100_000)

        assert "ATTACHED DOCUMENTS" in block
        assert "--- BEGIN DOCUMENT 1: paper.pdf" in block
        assert "--- BEGIN DOCUMENT 2: notes.md" in block
        assert "doc_id: doc_aaa" in block
        assert "Alpha body" in block
        assert "Beta body" in block
        assert "rag_query call is needed" in block
        assert all(not r["truncated"] for r in report)
        assert all(r["included"] for r in report)

    def test_truncates_per_document_with_marker_and_reports(self):
        docs = [
            {
                "doc_id": f"doc_{i}",
                "filename": f"f{i}.md",
                "text": "x" * 20_000,
            }
            for i in range(3)
        ]
        block, report = build_context_block(docs, 12_000)

        assert len(block) <= 12_200  # hard budget + labelling slack
        for entry in report:
            assert entry["truncated"] is True
            assert 0 < entry["kept_chars"] < entry["chars"]
        assert block.count("[truncated]") >= 3
        assert "BUDGET NOTE" in block
        assert "were truncated" in block

    def test_omits_documents_that_cannot_fit_and_reports_them(self):
        docs = [
            {"doc_id": "doc_big", "filename": "big.md", "text": "y" * 50_000},
        ]
        block, report = build_context_block(docs, 500)
        assert block == ""
        assert report[0]["included"] is False

    def test_strips_upload_header(self):
        text = UPLOAD_HEADER + "Real content"
        block, _ = build_context_block(
            [{"doc_id": "doc_a", "filename": "a.md", "text": text}], 10_000
        )
        assert "Real content" in block
        assert "# Source: Session Upload" not in block

    def test_strip_document_header_keeps_plain_text(self):
        assert strip_document_header("plain") == "plain"


# ---------------------------------------------------------------
# DocumentService attachment state
# ---------------------------------------------------------------


@pytest.fixture
def doc_service(tmp_path):
    scope = tmp_path / "sess_1"
    scope.mkdir()
    return DocumentService(scope_id="sess_1", scope_dir=scope, scope_type="session")


class TestDocumentServiceContext:
    def test_attach_text_document(self, doc_service):
        meta = doc_service.upload_text(b"Some body text", "notes.txt")

        chars = doc_service.set_in_context(meta.pdf_id)

        assert chars is not None and chars > len("Some body text")
        assert doc_service.is_in_context(meta.pdf_id)
        assert doc_service.metadata_cache[meta.pdf_id]["context_chars"] == chars

    def test_detach_clears_flags(self, doc_service):
        meta = doc_service.upload_text(b"Body", "notes.txt")
        doc_service.set_in_context(meta.pdf_id)

        doc_service.set_in_context(meta.pdf_id, False)

        assert not doc_service.is_in_context(meta.pdf_id)
        assert "context_chars" not in doc_service.metadata_cache[meta.pdf_id]

    def test_attach_unknown_document_raises(self, doc_service):
        with pytest.raises(ValueError):
            doc_service.set_in_context("doc_missing")

    def test_get_context_documents_only_returns_flagged(self, doc_service):
        attached = doc_service.upload_text(b"Attached body", "a.txt")
        doc_service.upload_text(b"Indexed only", "b.txt")
        doc_service.set_in_context(attached.pdf_id)

        docs = doc_service.get_context_documents()

        assert len(docs) == 1
        assert docs[0]["doc_id"] == attached.pdf_id
        assert "Attached body" in docs[0]["text"]
        assert docs[0]["filename"] == "a.txt"

    def test_attach_pdf_converts_with_fast_converter(self, doc_service, monkeypatch):
        meta = doc_service.upload(b"%PDF-fake", "paper.pdf")
        md_path = doc_service.scope_dir / "markdown" / f"{meta.pdf_id}.md"

        calls = []

        def fake_convert(pdf_path, use_marker=True):
            calls.append(use_marker)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(UPLOAD_HEADER + "converted body", encoding="utf-8")
            return md_path

        monkeypatch.setattr(doc_service, "convert_to_markdown", fake_convert)

        chars = doc_service.set_in_context(meta.pdf_id)

        assert chars and chars > 0
        assert doc_service.is_in_context(meta.pdf_id)
        assert calls == [False]  # never the slow marker pipeline


# ---------------------------------------------------------------
# Prompt injection in the orchestrator
# ---------------------------------------------------------------


def _orchestrator(context_window=32768, session_id="sess_1"):
    return OrchestratorService(
        tool_service=MagicMock(),
        rag_service=MagicMock(),
        model="test-model",
        base_url="http://localhost:11434",
        context_window=context_window,
        session_id=session_id,
    )


class TestOrchestratorContextInjection:
    def test_prompt_includes_attached_documents(self):
        service = _orchestrator()
        docs = [
            {
                "doc_id": "doc_1",
                "filename": "paper.pdf",
                "text": UPLOAD_HEADER + "Unique finding about tensors.",
            }
        ]

        block = service._build_in_context_block(docs, "base", "question")
        prompt = service._build_system_prompt([], in_context_block=block)

        assert "ATTACHED DOCUMENTS" in prompt
        assert "Unique finding about tensors." in prompt
        assert "paper.pdf" in prompt

    def test_no_documents_means_no_block(self):
        service = _orchestrator()
        assert service._build_in_context_block([], "base", "q") == ""
        assert "ATTACHED DOCUMENTS" not in service._build_system_prompt([])

    def test_budget_cap_scales_with_context_window(self):
        service = _orchestrator(context_window=100_000)
        budget = service._context_documents_budget("base", "q")
        cap = int(100_000 * CHARS_PER_TOKEN * CONTEXT_BUDGET_PCT)
        assert budget == cap

    def test_budget_never_exceeds_remaining_room(self):
        # Tiny window plus a big base prompt leaves almost no room.
        service = _orchestrator(context_window=4096)
        budget = service._context_documents_budget("z" * 14_000, "q")
        assert budget == 0

    def test_oversized_document_is_truncated_not_dropped(self):
        service = _orchestrator(context_window=8192)
        docs = [{"doc_id": "doc_1", "filename": "big.md", "text": "w" * 200_000}]

        block = service._build_in_context_block(docs, "base", "q")
        budget = service._context_documents_budget("base", "q")

        assert "big.md" in block
        assert "[truncated]" in block
        assert "BUDGET NOTE" in block
        assert len(block) <= budget

    def test_loads_documents_for_session(self):
        service = _orchestrator()
        docs = [{"doc_id": "d", "filename": "f", "text": "t", "chars": 1}]
        service_mock = MagicMock()
        service_mock.__enter__.return_value = service_mock
        service_mock.get_context_documents.return_value = docs

        with patch(
            "tensortruth.api.deps.get_document_service", return_value=service_mock
        ):
            assert service._load_in_context_documents() == docs

    def test_no_session_means_no_documents(self):
        service = _orchestrator(session_id=None)
        assert service._load_in_context_documents() == []
