"""Tests for the build_index orchestrator tool and its shared build helper.

The tool must:
- refuse project scope when the session has no project,
- turn "nothing to do" REST 400s into friendly strings,
- ask for confirmation before doing any work,
- block on the TaskRunner while emitting machine-readable progress,
- report completion, failure and timeout without raising.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from tensortruth.api.routes.documents import _build_index
from tensortruth.api.schemas import ReindexTaskResponse
from tensortruth.services.models import ToolProgress
from tensortruth.services.orchestrator_tool_wrappers import (
    _format_build_stage,
    create_all_tool_wrappers,
    create_build_index_tool,
)
from tensortruth.services.task_runner import TaskInfo, TaskStatus


def _doc_service(pdf_count=2, markdown=(), unindexed=("pdf_ab12cd34", "doc_ef34gh56")):
    service = MagicMock()
    service.get_pdf_count.return_value = pdf_count
    service.get_all_markdown_files.return_value = list(markdown)
    service.get_unindexed_doc_ids.return_value = list(unindexed)
    service.__enter__.return_value = service
    service.__exit__.return_value = False
    return service


def _confirmation_service(outcome="approved"):
    service = MagicMock()
    service.create_confirmation.return_value = SimpleNamespace(confirmation_id="c-1")
    service.await_resolution = AsyncMock(return_value=outcome)
    return service


class _FakeTaskRunner:
    """Returns a scripted list of TaskInfo objects from get_task()."""

    def __init__(self, infos):
        self._infos = list(infos)

    def get_task(self, task_id):
        if not self._infos:
            return self._infos_last
        info = self._infos.pop(0)
        self._infos_last = info
        return info


def _task(status, progress=0, stage="", result=None, error=None):
    return TaskInfo(
        task_id="t-1",
        task_type="build-index",
        status=status,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        progress=progress,
        stage=stage,
        result=result,
        error=error,
    )


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(
        "tensortruth.services.orchestrator_tool_wrappers._BUILD_INDEX_POLL_INTERVAL",
        0.0,
    )


class TestFormatBuildStage:
    def test_conversion_stage_gets_position_and_title(self):
        message, phase = _format_build_stage(
            "Converting pdf_ab12cd34_Attention_Is_All_You_Need.pdf (1/2)"
        )
        assert message == "Converting 1/2: Attention_Is_All_You_Need.pdf"
        assert phase == "converting"

    def test_plain_stage_is_passed_through(self):
        message, phase = _format_build_stage(
            "Loading conversion models (one-time setup)"
        )
        assert message == "Loading conversion models (one-time setup)"
        assert phase == "loading"

    def test_empty_stage(self):
        message, phase = _format_build_stage("")
        assert phase == "preparing"
        assert "Preparing" in message


class TestRegistration:
    def test_registered_alongside_other_agent_tools(self):
        tools = create_all_tool_wrappers(
            MagicMock(),
            AsyncMock(),
            confirmation_service=_confirmation_service(),
            session_id="s-1",
        )

        names = [tool.metadata.name for tool in tools]
        assert "build_index" in names
        assert "add_arxiv_paper" in names

    def test_not_registered_without_confirmation_service(self):
        tools = create_all_tool_wrappers(MagicMock(), AsyncMock(), session_id="s-1")

        assert "build_index" not in [tool.metadata.name for tool in tools]


class TestBuildIndexToolInputs:
    @pytest.mark.asyncio
    async def test_project_scope_without_project_returns_error(self):
        tool = create_build_index_tool(
            AsyncMock(), _confirmation_service(), session_id="s-1", project_id=None
        )

        result = await tool.acall(scope="project")

        assert "not part of a project" in str(result)
        assert "scope='session'" in str(result)

    @pytest.mark.asyncio
    async def test_invalid_scope_returns_error(self):
        tool = create_build_index_tool(
            AsyncMock(), _confirmation_service(), session_id="s-1"
        )

        result = await tool.acall(scope="everything")

        assert "scope must be 'session' or 'project'" in str(result)

    @pytest.mark.asyncio
    async def test_no_documents_returns_friendly_string(self):
        confirmation_service = _confirmation_service()
        tool = create_build_index_tool(
            AsyncMock(), confirmation_service, session_id="s-1"
        )

        with patch(
            "tensortruth.api.deps.get_document_service",
            return_value=_doc_service(pdf_count=0, unindexed=()),
        ):
            result = await tool.acall(scope="session")

        assert "no documents" in str(result).lower()
        confirmation_service.create_confirmation.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_documents_indexed_returns_friendly_string(self):
        confirmation_service = _confirmation_service()
        tool = create_build_index_tool(
            AsyncMock(), confirmation_service, session_id="s-1"
        )

        with patch(
            "tensortruth.api.deps.get_document_service",
            return_value=_doc_service(pdf_count=3, unindexed=()),
        ):
            result = await tool.acall(scope="session")

        assert "already indexed" in str(result)
        confirmation_service.create_confirmation.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirmation_rejected_skips_build(self):
        confirmation_service = _confirmation_service(outcome="rejected")
        emitter = AsyncMock()
        tool = create_build_index_tool(emitter, confirmation_service, session_id="s-1")

        with (
            patch(
                "tensortruth.api.deps.get_document_service",
                return_value=_doc_service(),
            ),
            patch(
                "tensortruth.api.routes.documents._build_index", new=AsyncMock()
            ) as build,
        ):
            result = await tool.acall(scope="session")

        assert "rejected" in str(result)
        build.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_confirmation_title_mentions_document_count(self):
        confirmation_service = _confirmation_service(outcome="rejected")
        tool = create_build_index_tool(
            AsyncMock(), confirmation_service, session_id="s-1"
        )

        with patch(
            "tensortruth.api.deps.get_document_service",
            return_value=_doc_service(unindexed=("pdf_ab12cd34", "doc_ef34gh56")),
        ):
            await tool.acall(scope="session")

        kwargs = confirmation_service.create_confirmation.call_args.kwargs
        assert kwargs["action_type"] == "build_index"
        assert kwargs["title"] == "Build index for 2 session documents"
        assert kwargs["details"]["document_count"] == 2


class TestBuildIndexToolRun:
    @pytest.mark.asyncio
    async def test_blocks_until_complete_and_emits_progress_metadata(self):
        emitter = AsyncMock()
        tool = create_build_index_tool(
            emitter, _confirmation_service(), session_id="s-1"
        )
        runner = _FakeTaskRunner(
            [
                _task(
                    TaskStatus.RUNNING,
                    progress=30,
                    stage="Converting pdf_ab12cd34_Paper.pdf (1/2)",
                ),
                _task(
                    TaskStatus.COMPLETED,
                    progress=100,
                    stage="Complete",
                    result={"documents_indexed": 2, "chunks": 42},
                ),
            ]
        )

        with (
            patch(
                "tensortruth.api.deps.get_document_service",
                return_value=_doc_service(),
            ),
            patch(
                "tensortruth.api.routes.documents._build_index",
                new=AsyncMock(
                    return_value=ReindexTaskResponse(task_id="t-1", pdf_count=2)
                ),
            ),
            patch("tensortruth.api.deps.get_task_runner", return_value=runner),
        ):
            result = await tool.acall(scope="session")

        text = str(result)
        assert "Index build complete for the session" in text
        assert "Documents indexed: 2" in text
        assert "Chunks in index: 42" in text
        assert "Elapsed:" in text

        progresses = [
            call.args[0]
            for call in emitter.await_args_list
            if isinstance(call.args[0], ToolProgress)
        ]
        converting = [p for p in progresses if p.metadata.get("percent") == 30]
        assert converting, progresses
        assert converting[0].metadata["message"] == "Converting 1/2: Paper.pdf"
        assert (
            converting[0].metadata["stage"] == "Converting pdf_ab12cd34_Paper.pdf (1/2)"
        )
        assert converting[0].phase == "converting"
        assert converting[0].tool_id == "build_index"
        # Final frame hits 100% so the bar can't be left mid-way.
        assert progresses[-1].metadata["percent"] == 100

    @pytest.mark.asyncio
    async def test_task_error_is_surfaced_as_string(self):
        emitter = AsyncMock()
        tool = create_build_index_tool(
            emitter, _confirmation_service(), session_id="s-1"
        )
        runner = _FakeTaskRunner(
            [_task(TaskStatus.ERROR, progress=10, stage="embedding", error="boom")]
        )

        with (
            patch(
                "tensortruth.api.deps.get_document_service",
                return_value=_doc_service(),
            ),
            patch(
                "tensortruth.api.routes.documents._build_index",
                new=AsyncMock(
                    return_value=ReindexTaskResponse(task_id="t-1", pdf_count=2)
                ),
            ),
            patch("tensortruth.api.deps.get_task_runner", return_value=runner),
        ):
            result = await tool.acall(scope="session")

        assert "index build failed" in str(result)
        assert "boom" in str(result)

    @pytest.mark.asyncio
    async def test_rest_400_is_surfaced_as_friendly_string(self):
        tool = create_build_index_tool(
            AsyncMock(), _confirmation_service(), session_id="s-1"
        )

        with (
            patch(
                "tensortruth.api.deps.get_document_service",
                return_value=_doc_service(),
            ),
            patch(
                "tensortruth.api.routes.documents._build_index",
                new=AsyncMock(
                    side_effect=HTTPException(
                        status_code=400, detail="All documents already indexed"
                    )
                ),
            ),
        ):
            result = await tool.acall(scope="session")

        assert "Could not start the index build" in str(result)
        assert "already indexed" in str(result)


class TestSharedBuildHelper:
    @pytest.mark.asyncio
    async def test_no_documents_raises_400(self):
        runner = MagicMock()
        runner.submit = AsyncMock(return_value="t-1")

        with patch(
            "tensortruth.api.routes.documents.get_document_service",
            return_value=_doc_service(pdf_count=0, unindexed=()),
        ):
            with pytest.raises(HTTPException) as exc:
                await _build_index("s-1", "session", runner)

        assert exc.value.status_code == 400
        assert exc.value.detail == "No documents to index"
        runner.submit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_indexed_raises_400(self):
        runner = MagicMock()
        runner.submit = AsyncMock(return_value="t-1")

        with patch(
            "tensortruth.api.routes.documents.get_document_service",
            return_value=_doc_service(pdf_count=2, unindexed=()),
        ):
            with pytest.raises(HTTPException) as exc:
                await _build_index("s-1", "session", runner)

        assert exc.value.detail == "All documents already indexed"

    @pytest.mark.asyncio
    async def test_submits_with_on_complete_and_reports_counts(self):
        runner = MagicMock()
        runner.submit = AsyncMock(return_value="t-1")
        service = _doc_service()
        builder = service._get_index_builder.return_value
        builder.index_exists.return_value = True
        collection = (
            builder._get_chroma_client.return_value.get_or_create_collection.return_value
        )
        collection.count.return_value = 17
        on_complete = MagicMock()

        with patch(
            "tensortruth.api.routes.documents.get_document_service",
            return_value=service,
        ):
            response = await _build_index(
                "s-1", "session", runner, None, on_complete=on_complete
            )

            assert response.task_id == "t-1"
            assert response.pdf_count == 2
            kwargs = runner.submit.call_args.kwargs
            assert kwargs["on_complete"] is on_complete
            assert kwargs["metadata"] == {
                "scope_id": "s-1",
                "scope_type": "session",
            }

            build_fn = runner.submit.call_args.args[1]
            result = build_fn(lambda *args: None)
        assert result == {
            "scope_id": "s-1",
            "scope_type": "session",
            "documents_indexed": 2,
            "chunks": 17,
        }
