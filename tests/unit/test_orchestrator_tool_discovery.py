"""Tests for orchestrator tool auto-discovery (Story 10).

Verifies the tool assembly mechanism in OrchestratorService._build_tools():
- Tool discovery with only web tools (no RAG)
- Tool discovery with RAG + web tools
- Tool discovery with MCP tools included
- Built-in tool name filtering works correctly
- Tool descriptions are non-empty and distinct
- get_tool_names() returns correct names
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.tools import FunctionTool

from tensortruth.services.orchestrator_service import (
    _WRAPPED_BUILTIN_TOOL_NAMES,
    OrchestratorService,
)

# --- Fixtures ---


@pytest.fixture
def tool_service():
    """ToolService with no MCP tools (empty .tools list)."""
    service = MagicMock()
    service.tools = []
    service.execute_tool = AsyncMock()
    return service


@pytest.fixture
def rag_service_loaded():
    """RAGService that reports is_loaded() = True."""
    service = MagicMock()
    service.is_loaded.return_value = True
    service.retrieve = MagicMock(
        return_value=MagicMock(
            source_nodes=[],
            confidence_level="none",
            metrics=None,
            condensed_query="test",
            num_sources=0,
        )
    )
    return service


@pytest.fixture
def rag_service_not_loaded():
    """RAGService that reports is_loaded() = False."""
    service = MagicMock()
    service.is_loaded.return_value = False
    return service


@pytest.fixture
def progress_emitter():
    """No-op progress emitter for tool building."""
    return MagicMock()


def _make_mcp_tool(name: str, description: str = "An MCP tool") -> MagicMock:
    """Create a mock FunctionTool representing an MCP tool."""
    tool = MagicMock(spec=FunctionTool)
    tool.metadata = MagicMock()
    tool.metadata.name = name
    tool.metadata.description = description
    return tool


def _create_service(tool_service, rag_service, **kwargs):
    """Helper to create an OrchestratorService with default params."""
    return OrchestratorService(
        tool_service=tool_service,
        rag_service=rag_service,
        model="test-model",
        base_url="http://localhost:11434",
        context_window=4096,
        **kwargs,
    )


# --- Tests: Tool discovery with only web tools (no RAG) ---


class TestToolDiscoveryWebOnly:
    """When RAG is unavailable, only web tools should be registered."""

    def test_web_tools_only_when_rag_not_loaded(
        self, tool_service, rag_service_not_loaded, progress_emitter
    ):
        """Should include only web tools when RAG is not loaded."""
        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(progress_emitter)

        names = {t.metadata.name for t in tools}
        assert names == {"web_search", "fetch_page", "fetch_pages_batch"}
        assert "rag_query" not in names

    def test_web_tools_only_when_rag_is_none(self, tool_service, progress_emitter):
        """Should include only web tools when rag_service is None."""
        svc = _create_service(
            tool_service, rag_service=MagicMock(is_loaded=MagicMock(return_value=False))
        )
        tools = svc._build_tools(progress_emitter)

        names = {t.metadata.name for t in tools}
        assert "rag_query" not in names
        assert len(tools) == 3


# --- Tests: Tool discovery with RAG + web tools ---


class TestToolDiscoveryWithRAG:
    """When RAG is loaded, rag_query should be included alongside web tools."""

    def test_includes_rag_when_loaded(
        self, tool_service, rag_service_loaded, progress_emitter
    ):
        """Should include rag_query plus the three web tools."""
        svc = _create_service(tool_service, rag_service_loaded)
        tools = svc._build_tools(progress_emitter)

        names = {t.metadata.name for t in tools}
        assert names == {"rag_query", "web_search", "fetch_page", "fetch_pages_batch"}

    def test_rag_tool_is_first(
        self, tool_service, rag_service_loaded, progress_emitter
    ):
        """rag_query should appear first in the tool list (wrapped tools order)."""
        svc = _create_service(tool_service, rag_service_loaded)
        tools = svc._build_tools(progress_emitter)

        assert tools[0].metadata.name == "rag_query"


# --- Tests: Tool discovery with MCP tools ---


class TestToolDiscoveryWithMCPTools:
    """MCP tools from ToolService should be included alongside wrapped tools."""

    def test_includes_mcp_tools(
        self, tool_service, rag_service_not_loaded, progress_emitter
    ):
        """Should include MCP tools that don't conflict with wrapped names."""
        mcp_tool = _make_mcp_tool("custom_mcp_tool", "A custom MCP tool")
        tool_service.tools = [mcp_tool]

        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(progress_emitter)

        names = {t.metadata.name for t in tools}
        assert "custom_mcp_tool" in names
        # 3 web tools + 1 MCP tool
        assert len(tools) == 4

    def test_includes_multiple_mcp_tools(
        self, tool_service, rag_service_loaded, progress_emitter
    ):
        """Should include all non-conflicting MCP tools."""
        mcp1 = _make_mcp_tool("mcp_analyzer", "Analyze data")
        mcp2 = _make_mcp_tool("mcp_summarizer", "Summarize text")
        tool_service.tools = [mcp1, mcp2]

        svc = _create_service(tool_service, rag_service_loaded)
        tools = svc._build_tools(progress_emitter)

        names = {t.metadata.name for t in tools}
        assert "mcp_analyzer" in names
        assert "mcp_summarizer" in names
        # 4 wrapped (rag + 3 web) + 2 MCP
        assert len(tools) == 6

    def test_no_mcp_tools_is_fine(
        self, tool_service, rag_service_not_loaded, progress_emitter
    ):
        """Should work gracefully when no MCP tools are available."""
        tool_service.tools = []

        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(progress_emitter)

        # Should still have the 3 web tools
        assert len(tools) == 3


# --- Tests: Built-in tool name filtering ---


class TestBuiltinToolNameFiltering:
    """MCP tools with names matching wrapped built-in tools must be filtered."""

    def test_filters_search_web_from_mcp(
        self, tool_service, rag_service_not_loaded, progress_emitter
    ):
        """Should filter out an MCP tool named 'search_web' to avoid duplicates."""
        conflicting = _make_mcp_tool("search_web", "Conflicting search_web")
        tool_service.tools = [conflicting]

        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(progress_emitter)

        # The conflicting MCP tool should be excluded
        names = [t.metadata.name for t in tools]
        # web_search should appear (wrapped), not search_web from MCP
        assert names.count("search_web") == 0
        # web_search IS the wrapped version of search_web
        assert "web_search" in names

    def test_filters_all_wrapped_builtin_names(
        self, tool_service, rag_service_not_loaded, progress_emitter
    ):
        """Should filter all names in _WRAPPED_BUILTIN_TOOL_NAMES."""
        conflicting_tools = [
            _make_mcp_tool(name, f"Conflicting {name}")
            for name in _WRAPPED_BUILTIN_TOOL_NAMES
        ]
        tool_service.tools = conflicting_tools

        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(progress_emitter)

        # None of the conflicting MCP tools should be included
        for builtin_name in _WRAPPED_BUILTIN_TOOL_NAMES:
            # The raw builtin name should not appear (it's filtered)
            mcp_count = sum(
                1
                for t in tools
                if t.metadata.name == builtin_name and isinstance(t, MagicMock)
            )
            assert mcp_count == 0, f"MCP tool '{builtin_name}' should be filtered"

    def test_keeps_non_conflicting_mcp_tools(
        self, tool_service, rag_service_not_loaded, progress_emitter
    ):
        """Non-conflicting MCP tools should be kept even when conflicting ones are filtered."""
        conflicting = _make_mcp_tool("search_web", "Conflicting")
        good_tool = _make_mcp_tool("my_custom_tool", "A custom tool")
        tool_service.tools = [conflicting, good_tool]

        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(progress_emitter)

        names = {t.metadata.name for t in tools}
        assert "my_custom_tool" in names

    def test_wrapped_builtin_names_constant_is_complete(self):
        """The _WRAPPED_BUILTIN_TOOL_NAMES set should contain all expected names."""
        expected = {"search_web", "fetch_page", "fetch_pages_batch", "search_focused"}
        assert _WRAPPED_BUILTIN_TOOL_NAMES == expected


# --- Tests: Tool descriptions ---


class TestToolDescriptions:
    """All tools must have non-empty, distinct descriptions for LLM routing."""

    def test_all_wrapped_tools_have_descriptions(
        self, tool_service, rag_service_loaded, progress_emitter
    ):
        """Every wrapped tool should have a non-empty description."""
        svc = _create_service(tool_service, rag_service_loaded)
        tools = svc._build_tools(progress_emitter)

        for tool in tools:
            assert (
                tool.metadata.description
            ), f"Tool '{tool.metadata.name}' has no description"
            assert (
                len(tool.metadata.description) > 20
            ), f"Tool '{tool.metadata.name}' description is too short"

    def test_tool_descriptions_are_distinct(
        self, tool_service, rag_service_loaded, progress_emitter
    ):
        """No two tools should share the exact same description."""
        svc = _create_service(tool_service, rag_service_loaded)
        tools = svc._build_tools(progress_emitter)

        descriptions = [t.metadata.description for t in tools]
        assert len(descriptions) == len(
            set(descriptions)
        ), "Duplicate tool descriptions found"

    def test_rag_tool_description_mentions_knowledge_base(
        self, tool_service, rag_service_loaded, progress_emitter
    ):
        """rag_query description should mention knowledge base for clear routing."""
        svc = _create_service(tool_service, rag_service_loaded)
        tools = svc._build_tools(progress_emitter)

        rag_tool = next(t for t in tools if t.metadata.name == "rag_query")
        desc = rag_tool.metadata.description.lower()
        assert "knowledge base" in desc

    def test_web_search_description_mentions_current_info(
        self, tool_service, rag_service_not_loaded, progress_emitter
    ):
        """web_search description should guide toward current/live information."""
        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(progress_emitter)

        ws_tool = next(t for t in tools if t.metadata.name == "web_search")
        desc = ws_tool.metadata.description.lower()
        assert "current" in desc or "recent" in desc or "web" in desc

    def test_fetch_page_description_mentions_single_page(
        self, tool_service, rag_service_not_loaded, progress_emitter
    ):
        """fetch_page description should mention it fetches a single page."""
        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(progress_emitter)

        fp_tool = next(t for t in tools if t.metadata.name == "fetch_page")
        desc = fp_tool.metadata.description.lower()
        assert "single" in desc or "page" in desc

    def test_fetch_pages_batch_description_mentions_parallel(
        self, tool_service, rag_service_not_loaded, progress_emitter
    ):
        """fetch_pages_batch description should mention parallel fetching."""
        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(progress_emitter)

        fpb_tool = next(t for t in tools if t.metadata.name == "fetch_pages_batch")
        desc = fpb_tool.metadata.description.lower()
        assert "parallel" in desc


# --- Tests: get_tool_names() ---


class TestGetToolNames:
    """Tests for OrchestratorService.get_tool_names()."""

    def test_returns_empty_before_execute(self, tool_service, rag_service_not_loaded):
        """Should return empty list before any execution."""
        svc = _create_service(tool_service, rag_service_not_loaded)
        assert svc.get_tool_names() == []

    def test_returns_sorted_names_after_build(
        self, tool_service, rag_service_loaded, progress_emitter
    ):
        """Should return sorted tool names after tools are built."""
        svc = _create_service(tool_service, rag_service_loaded)
        tools = svc._build_tools(progress_emitter)
        svc._tools = tools

        names = svc.get_tool_names()
        assert names == sorted(names)
        assert set(names) == {
            "fetch_page",
            "fetch_pages_batch",
            "rag_query",
            "web_search",
        }

    def test_includes_mcp_tool_names(
        self, tool_service, rag_service_not_loaded, progress_emitter
    ):
        """Should include MCP tool names in the list."""
        mcp_tool = _make_mcp_tool("alpha_tool", "Alpha")
        tool_service.tools = [mcp_tool]

        svc = _create_service(tool_service, rag_service_not_loaded)
        tools = svc._build_tools(progress_emitter)
        svc._tools = tools

        names = svc.get_tool_names()
        assert "alpha_tool" in names
        # Sorted: alpha_tool, fetch_page, fetch_pages_batch, web_search
        assert names[0] == "alpha_tool"

    def test_returns_empty_when_tools_is_none(
        self, tool_service, rag_service_not_loaded
    ):
        """Should return empty list when _tools is None (pre-init state)."""
        svc = _create_service(tool_service, rag_service_not_loaded)
        svc._tools = None
        assert svc.get_tool_names() == []


# --- Tests: Logging ---


class TestToolBuildLogging:
    """Verify that _build_tools() produces appropriate log messages."""

    def test_logs_tool_count_at_info_level(
        self, tool_service, rag_service_loaded, progress_emitter, caplog
    ):
        """Should log the total tool count at INFO level."""
        import logging

        with caplog.at_level(
            logging.INFO, logger="tensortruth.services.orchestrator_service"
        ):
            svc = _create_service(tool_service, rag_service_loaded)
            svc._build_tools(progress_emitter)

        # Find the info-level log message
        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            "4 tools" in msg and "4 wrapped" in msg for msg in info_msgs
        ), f"Expected INFO log about 4 tools, got: {info_msgs}"

    def test_logs_tool_descriptions_at_debug_level(
        self, tool_service, rag_service_not_loaded, progress_emitter, caplog
    ):
        """Should log individual tool descriptions at DEBUG level."""
        import logging

        with caplog.at_level(
            logging.DEBUG, logger="tensortruth.services.orchestrator_service"
        ):
            svc = _create_service(tool_service, rag_service_not_loaded)
            svc._build_tools(progress_emitter)

        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        # Each of the 3 tools should have a debug log for its description
        tool_desc_msgs = [m for m in debug_msgs if "Tool '" in m]
        assert len(tool_desc_msgs) == 3
