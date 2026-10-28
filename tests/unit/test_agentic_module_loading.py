"""Tests for agent-driven module loading (load_modules, prompt, cache identity)."""

from unittest.mock import MagicMock

import pytest

from tensortruth.services.module_catalog import ModuleDescription
from tensortruth.services.orchestrator_service import (
    OrchestratorService,
    _compute_cache_identity,
)
from tensortruth.services.orchestrator_tool_wrappers import (
    create_all_tool_wrappers,
    create_load_modules_tool,
    create_rag_tool,
)

CATALOG = [
    ModuleDescription(
        name="papers_dl",
        display_name="DL Papers",
        doc_type="paper",
        description="CNN and transformer papers",
        doc_count=20,
    ),
    ModuleDescription(
        name="library_pytorch_2.9",
        display_name="PyTorch 2.9",
        doc_type="library_doc",
        description="PyTorch library documentation",
        doc_count=2299,
    ),
]


def _rag_service(loaded: bool = False):
    service = MagicMock()
    service.is_loaded.return_value = loaded
    return service


def _service(tool_service, rag_service, **kwargs):
    return OrchestratorService(
        tool_service=tool_service,
        rag_service=rag_service,
        model="test-model",
        base_url="http://localhost:11434",
        context_window=4096,
        **kwargs,
    )


# --- load_modules tool ----------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_module_rejected_with_valid_names():
    rag = _rag_service()
    tool = create_load_modules_tool(rag, lambda _p: None, module_catalog=CATALOG)

    result = str(await tool.acall(modules=["pytorch"]))

    assert "unknown module" in result
    assert "pytorch" in result
    assert "papers_dl" in result and "library_pytorch_2.9" in result
    rag.load_engine.assert_not_called()


@pytest.mark.asyncio
async def test_more_than_six_modules_rejected():
    catalog = [
        ModuleDescription(f"book_{i}", f"Book {i}", "book", "d", 1) for i in range(8)
    ]
    rag = _rag_service()
    tool = create_load_modules_tool(rag, lambda _p: None, module_catalog=catalog)

    result = str(await tool.acall(modules=[entry.name for entry in catalog[:7]]))

    assert "at most 6 modules" in result
    rag.load_engine.assert_not_called()


@pytest.mark.asyncio
async def test_load_modules_reloads_with_merged_set_and_preserves_context():
    rag = _rag_service()
    loaded = ["library_pytorch_2.9"]
    phases = []
    session_service = MagicMock()
    session_service.load.return_value = "session-data"
    session_service.update_modules.return_value = "updated-data"

    tool = create_load_modules_tool(
        rag,
        phases.append,
        module_catalog=CATALOG,
        loaded_modules=loaded,
        session_params={"model": "qwen", "temperature": 0.2},
        additional_index_paths=["/tmp/session-index"],
        session_id="sess-1",
        session_service=session_service,
    )

    result = str(await tool.acall(modules=["papers_dl"]))

    # One reload, merged modules, params and index paths preserved.
    rag.load_engine.assert_called_once_with(
        ["library_pytorch_2.9", "papers_dl"],
        {"model": "qwen", "temperature": 0.2},
        ["/tmp/session-index"],
        owner="sess-1",
    )
    # Live set shared with the orchestrator.
    assert loaded == ["library_pytorch_2.9", "papers_dl"]
    # Session persisted so the next turn's needs_reload and the UI agree.
    session_service.update_modules.assert_called_once_with(
        "sess-1", ["library_pytorch_2.9", "papers_dl"], "session-data"
    )
    session_service.save.assert_called_once_with("updated-data")

    assert "Loaded: papers_dl (20 docs)" in result
    assert "Currently loaded: library_pytorch_2.9, papers_dl" in result

    assert phases, "expected a loading progress phase"
    assert phases[0].tool_id == "modules"
    assert phases[0].phase == "loading_module"
    assert phases[0].metadata["module"] == "papers_dl"


@pytest.mark.asyncio
async def test_loading_an_already_loaded_module_does_not_reload():
    rag = _rag_service()
    tool = create_load_modules_tool(
        rag,
        lambda _p: None,
        module_catalog=CATALOG,
        loaded_modules=["papers_dl"],
        session_id=None,
    )

    result = str(await tool.acall(modules=["papers_dl"]))

    rag.load_engine.assert_not_called()
    assert "Already loaded: papers_dl" in result


@pytest.mark.asyncio
async def test_rag_query_without_engine_returns_guidance():
    rag = _rag_service(loaded=False)
    tool = create_rag_tool(
        rag,
        lambda _p: None,
        catalog_names=[entry.name for entry in CATALOG],
    )

    result = str(await tool.acall(query="what is a transformer?"))

    assert "No knowledge modules are loaded" in result
    assert "load_modules" in result
    assert "papers_dl" in result


# --- registration ---------------------------------------------------------


def test_registration_includes_rag_tools_when_engine_not_loaded():
    tool_service = MagicMock()
    tool_service.tools = []
    rag = _rag_service(loaded=False)

    tools = create_all_tool_wrappers(tool_service, lambda _p: None, rag_service=rag)

    names = {tool.metadata.name for tool in tools}
    assert {"rag_query", "load_modules"} <= names


def test_manual_mode_omits_load_modules_but_keeps_rag_query():
    tool_service = MagicMock()
    tool_service.tools = []

    tools = create_all_tool_wrappers(
        tool_service,
        lambda _p: None,
        rag_service=_rag_service(loaded=False),
        module_mode="manual",
    )

    names = {tool.metadata.name for tool in tools}
    assert "rag_query" in names
    assert "load_modules" not in names


@pytest.mark.asyncio
async def test_rag_query_guidance_differs_by_mode_without_engine():
    catalog_names = [entry.name for entry in CATALOG]

    auto_tool = create_rag_tool(
        _rag_service(loaded=False), lambda _p: None, catalog_names=catalog_names
    )
    manual_tool = create_rag_tool(
        _rag_service(loaded=False),
        lambda _p: None,
        catalog_names=catalog_names,
        module_mode="manual",
    )

    auto_result = str(await auto_tool.acall(query="what is a transformer?"))
    manual_result = str(await manual_tool.acall(query="what is a transformer?"))

    assert "load_modules" in auto_result
    assert "load_modules" not in manual_result
    assert "user has not selected any knowledge modules" in manual_result


# --- system prompt --------------------------------------------------------


def test_prompt_lists_whole_catalog_with_loaded_markers():
    tool_service = MagicMock()
    tool_service.tools = []
    service = _service(
        tool_service,
        _rag_service(loaded=True),
        module_descriptions=[CATALOG[0]],
        module_catalog=CATALOG,
    )

    prompt = service._build_system_prompt(service._build_tools())

    # Unloaded modules are listed too, so the agent knows they exist.
    assert "papers_dl" in prompt
    assert "library_pytorch_2.9" in prompt
    assert "CNN and transformer papers" in prompt
    assert "2299 docs" in prompt
    # Only the session's active module is marked loaded.
    papers_line = next(line for line in prompt.splitlines() if "papers_dl" in line)
    library_line = next(
        line for line in prompt.splitlines() if "library_pytorch_2.9" in line
    )
    assert "[loaded]" in papers_line
    assert "[loaded]" not in library_line
    assert "call load_modules before answering" in prompt
    assert "Never claim to have read content from an unloaded module" in prompt


def test_manual_prompt_lists_only_selected_modules_without_catalog():
    tool_service = MagicMock()
    tool_service.tools = []
    service = _service(
        tool_service,
        _rag_service(loaded=True),
        module_descriptions=[CATALOG[0]],
        module_catalog=CATALOG,
        module_mode="manual",
    )

    tools = service._build_tools()
    prompt = service._build_system_prompt(tools)

    assert "papers_dl" in prompt
    # No catalog entry the user did not select, no [loaded] markers.
    assert "library_pytorch_2.9" not in prompt
    assert "[loaded]" not in prompt
    # The selection is fixed and load_modules is not even registered.
    assert "fixed for this session" in prompt
    assert "load_modules" not in prompt
    assert "load_modules" not in {tool.metadata.name for tool in tools}


def test_auto_prompt_is_the_default_and_lists_catalog():
    tool_service = MagicMock()
    tool_service.tools = []
    service = _service(
        tool_service,
        _rag_service(loaded=True),
        module_descriptions=[CATALOG[0]],
        module_catalog=CATALOG,
    )

    tools = service._build_tools()
    prompt = service._build_system_prompt(tools)

    assert service._module_mode == "auto"
    assert "library_pytorch_2.9" in prompt
    assert "call load_modules before answering" in prompt
    assert "load_modules" in {tool.metadata.name for tool in tools}


def test_prompt_truncates_long_descriptions_and_compacts_huge_catalogs():
    tool_service = MagicMock()
    tool_service.tools = []
    long_desc = "x" * 300
    catalog = [
        ModuleDescription(f"book_{i}", f"Book {i}", "book", long_desc, i)
        for i in range(3)
    ]
    service = _service(
        tool_service,
        _rag_service(loaded=True),
        module_descriptions=[],
        module_catalog=catalog,
    )
    prompt = service._build_system_prompt(service._build_tools())
    assert "x" * 88 not in prompt
    assert "x" * 87 + "..." in prompt

    # Above the cap, descriptions and doc counts are dropped entirely.
    big = [
        ModuleDescription(f"book_{i}", f"Book {i}", "book", long_desc, i)
        for i in range(81)
    ]
    service._module_catalog = big
    compact_prompt = service._build_system_prompt(service._build_tools())
    assert "x" * 20 not in compact_prompt
    assert "book_80" in compact_prompt


# --- cache identity -------------------------------------------------------


def _identity(rag_service, module_descriptions, module_mode="auto"):
    tool_service = MagicMock()
    tool_service.tools = []
    return _compute_cache_identity(
        session_id="sess-1",
        tool_service=tool_service,
        rag_service=rag_service,
        model="m",
        base_url="http://x",
        context_window=4096,
        session_params={"model": "m"},
        module_descriptions=module_descriptions,
        custom_instructions=None,
        project_metadata=None,
        max_iterations=10,
        mcp_server_service=None,
        module_mode=module_mode,
    )


def test_cache_identity_changes_when_loaded_set_changes():
    rag = _rag_service(loaded=True)
    before = _identity(rag, [])
    after = _identity(rag, [CATALOG[0]])
    assert before != after


def test_cache_identity_includes_rag_tools_even_when_not_loaded():
    rag = _rag_service(loaded=False)
    identity = _identity(rag, [])
    wrapped = identity[4]
    assert "rag_query" in wrapped
    assert "load_modules" in wrapped


def test_cache_identity_changes_when_module_mode_flips():
    rag = _rag_service(loaded=True)
    auto = _identity(rag, [CATALOG[0]], module_mode="auto")
    manual = _identity(rag, [CATALOG[0]], module_mode="manual")
    assert auto != manual
    # Manual mode also drops load_modules from the tool-name set.
    assert "load_modules" in auto[4]
    assert "load_modules" not in manual[4]
