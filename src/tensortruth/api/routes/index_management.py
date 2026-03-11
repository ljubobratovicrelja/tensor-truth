"""API routes for index management.

Provides endpoints for managing documentation sources, fetching docs,
building indexes, and viewing index metadata.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from tensortruth.api.deps import TaskRunnerDep, get_index_management_service
from tensortruth.api.schemas.index_management import (
    BatchBuildRequest,
    BuildRequest,
    FetchRequest,
    ModulesOverviewResponse,
    SourceCreateRequest,
    SourcesResponse,
    SourceUpdateRequest,
    ValidateResponse,
)
from tensortruth.api.schemas.task import TaskSubmitResponse
from tensortruth.services.index_management_service import IndexManagementService

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_service() -> IndexManagementService:
    return get_index_management_service()


def _make_on_complete():
    """Create an on_complete callback that clears the RAG engine cache.

    After build/delete, clear the engine so it picks up new indexes on next use.
    """

    def _clear_engine(task_info):  # type: ignore[no-untyped-def]
        try:
            from tensortruth.api.deps import get_rag_service

            rag = get_rag_service()
            rag.clear()
            logger.info("RAG engine cleared after index operation")
        except Exception as e:
            logger.warning(f"Failed to clear RAG engine after index op: {e}")

    return _clear_engine


# ── Sources CRUD ──────────────────────────────────────────────────────


@router.get("/sources", response_model=SourcesResponse)
async def get_sources():
    """List all sources from sources.json."""
    service = _get_service()
    data = service.get_sources()
    return SourcesResponse(
        libraries=data.get("libraries", {}),
        papers=data.get("papers", {}),
        books=data.get("books", {}),
    )


@router.post("/sources", status_code=201)
async def add_source(request: SourceCreateRequest):
    """Add a new source to sources.json."""
    service = _get_service()
    try:
        service.add_source(request.source_type, request.name, request.config)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {
        "status": "created",
        "source_type": request.source_type,
        "name": request.name,
    }


@router.put("/sources/{source_type}/{name}")
async def update_source(source_type: str, name: str, request: SourceUpdateRequest):
    """Update an existing source in sources.json."""
    service = _get_service()
    try:
        service.update_source(source_type, name, request.config)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "updated", "source_type": source_type, "name": name}


@router.delete("/sources/{source_type}/{name}", status_code=204)
async def delete_source(source_type: str, name: str):
    """Delete a source from sources.json."""
    service = _get_service()
    try:
        service.delete_source(source_type, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Modules overview ──────────────────────────────────────────────────


@router.get("/modules", response_model=ModulesOverviewResponse)
async def get_modules_overview(
    embedding_model_id: Optional[str] = Query(None),
):
    """Get merged module overview: source config + fetch status + index status."""
    service = _get_service()
    modules_data = service.get_modules_overview(embedding_model_id)
    from tensortruth.api.schemas.index_management import ModuleOverview

    modules = [ModuleOverview(**m) for m in modules_data]
    return ModulesOverviewResponse(modules=modules)


# ── Fetch & Build ─────────────────────────────────────────────────────


@router.post("/fetch", response_model=TaskSubmitResponse)
async def fetch_source(request: FetchRequest, task_runner: TaskRunnerDep):
    """Start fetching documentation for a source. Returns task_id for progress tracking."""
    service = _get_service()
    try:
        fn = service.create_fetch_callable(
            source_type=request.source_type,
            name=request.name,
            workers=request.workers,
            converter=request.converter,
            output_format=request.output_format,
            enable_cleanup=request.enable_cleanup,
            min_size=request.min_size,
            pages_per_chunk=request.pages_per_chunk,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    task_id = await task_runner.submit(
        task_type="fetch_source",
        fn=fn,
        metadata={"source_type": request.source_type, "name": request.name},
    )
    return TaskSubmitResponse(task_id=task_id)


@router.post("/build", response_model=TaskSubmitResponse)
async def build_index(request: BuildRequest, task_runner: TaskRunnerDep):
    """Start building a vector index for a module. Returns task_id."""
    service = _get_service()
    try:
        fn = service.create_build_callable(
            module_name=request.module_name,
            chunking_strategy=request.chunking_strategy,
            chunk_sizes=request.chunk_sizes,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model,
            semantic_buffer_size=request.semantic_buffer_size,
            semantic_breakpoint_threshold=request.semantic_breakpoint_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task_id = await task_runner.submit(
        task_type="build_index",
        fn=fn,
        on_complete=_make_on_complete(),
        metadata={"module_name": request.module_name},
    )
    return TaskSubmitResponse(task_id=task_id)


@router.post("/build/batch")
async def batch_build(request: BatchBuildRequest, task_runner: TaskRunnerDep):
    """Start building indexes for multiple modules. Returns list of task_ids."""
    service = _get_service()
    task_ids: list[dict[str, Any]] = []

    for module_name in request.module_names:
        try:
            fn = service.create_build_callable(
                module_name=module_name,
                chunking_strategy=request.chunking_strategy,
                chunk_sizes=request.chunk_sizes,
                chunk_overlap=request.chunk_overlap,
                embedding_model=request.embedding_model,
                semantic_buffer_size=request.semantic_buffer_size,
                semantic_breakpoint_threshold=request.semantic_breakpoint_threshold,
            )
            task_id = await task_runner.submit(
                task_type="build_index",
                fn=fn,
                on_complete=_make_on_complete(),
                metadata={"module_name": module_name},
            )
            task_ids.append({"module_name": module_name, "task_id": task_id})
        except ValueError as e:
            task_ids.append(
                {"module_name": module_name, "task_id": None, "error": str(e)}
            )

    return {"tasks": task_ids}


# ── Index operations ──────────────────────────────────────────────────


@router.delete("/indexes/{embedding_model_id}/{module_name}", status_code=204)
async def delete_index(embedding_model_id: str, module_name: str):
    """Delete a built index from disk."""
    service = _get_service()
    try:
        service.delete_index(embedding_model_id, module_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Reinitialize RAG engine after deletion
    _make_on_complete()(None)


# ── Validation ────────────────────────────────────────────────────────


@router.post("/validate", response_model=ValidateResponse)
async def validate_sources():
    """Validate sources configuration against filesystem."""
    service = _get_service()
    result = service.validate_sources()
    return ValidateResponse(**result)
