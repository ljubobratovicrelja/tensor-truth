"""Tests for orchestrator tool wrappers.

Each wrapper is a factory that returns a FunctionTool capturing a ToolService
and a progress emitter via closure. Tests verify:
- Tools call the correct ToolService method with correct parameters
- Progress is emitted at the right phases
- Tool results are extracted and returned as strings
- Error cases are handled gracefully
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from tensortruth.services.models import RAGRetrievalResult, ToolProgress
from tensortruth.services.orchestrator_tool_wrappers import (
    create_all_tool_wrappers,
    create_fetch_page_tool,
    create_fetch_pages_batch_tool,
    create_rag_tool,
    create_web_search_tool,
)


@pytest.fixture
def tool_service():
    """Mock ToolService with execute_tool as AsyncMock."""
    service = MagicMock()
    service.execute_tool = AsyncMock()
    return service


@pytest.fixture
def progress_emitter():
    """Mock progress emitter that records all emitted ToolProgress objects."""
    emitter = AsyncMock()
    return emitter


class TestWebSearchTool:
    """Tests for create_web_search_tool."""

    @pytest.mark.asyncio
    async def test_calls_search_web_via_tool_service(
        self, tool_service, progress_emitter
    ):
        """Should delegate to ToolService.execute_tool with 'search_web'."""
        tool_service.execute_tool.return_value = {
            "success": True,
            "data": '[{"url": "https://example.com", "title": "Example"}]',
        }
        tool = create_web_search_tool(tool_service, progress_emitter)

        result = await tool.acall(query="test query")

        tool_service.execute_tool.assert_awaited_once_with(
            "search_web", {"queries": "test query"}
        )
        assert "example.com" in str(result)

    @pytest.mark.asyncio
    async def test_emits_searching_progress(self, tool_service, progress_emitter):
        """Should emit a 'searching' phase ToolProgress."""
        tool_service.execute_tool.return_value = {"success": True, "data": "[]"}
        tool = create_web_search_tool(tool_service, progress_emitter)

        await tool.acall(query="AI news")

        progress_emitter.assert_awaited_once()
        emitted: ToolProgress = progress_emitter.call_args[0][0]
        assert emitted.tool_id == "web_search"
        assert emitted.phase == "searching"
        assert "Searching" in emitted.message
        assert emitted.metadata["query"] == "AI news"

    @pytest.mark.asyncio
    async def test_returns_error_on_failure(self, tool_service, progress_emitter):
        """Should return an error string when ToolService reports failure."""
        tool_service.execute_tool.return_value = {
            "success": False,
            "error": "Network timeout",
        }
        tool = create_web_search_tool(tool_service, progress_emitter)

        result = await tool.acall(query="failing query")

        assert "Error:" in str(result)
        assert "Network timeout" in str(result)

    def test_tool_metadata(self, tool_service, progress_emitter):
        """Should have correct name and description for LLM consumption."""
        tool = create_web_search_tool(tool_service, progress_emitter)

        assert tool.metadata.name == "web_search"
        assert "web" in tool.metadata.description.lower()
        assert "search" in tool.metadata.description.lower()


class TestFetchPageTool:
    """Tests for create_fetch_page_tool."""

    @pytest.mark.asyncio
    async def test_calls_fetch_page_via_tool_service(
        self, tool_service, progress_emitter
    ):
        """Should delegate to ToolService.execute_tool with 'fetch_page'."""
        tool_service.execute_tool.return_value = {
            "success": True,
            "data": "# Page Title\nSome content here.",
        }
        tool = create_fetch_page_tool(tool_service, progress_emitter)

        result = await tool.acall(url="https://example.com")

        tool_service.execute_tool.assert_awaited_once_with(
            "fetch_page", {"url": "https://example.com"}
        )
        # Summary includes title (extracted from heading) and URL
        assert "Page Title" in str(result)
        assert "chars" in str(result)

    @pytest.mark.asyncio
    async def test_emits_fetching_progress(self, tool_service, progress_emitter):
        """Should emit a 'fetching' phase ToolProgress with URL metadata."""
        tool_service.execute_tool.return_value = {"success": True, "data": "content"}
        tool = create_fetch_page_tool(tool_service, progress_emitter)

        await tool.acall(url="https://example.com/page")

        progress_emitter.assert_awaited_once()
        emitted: ToolProgress = progress_emitter.call_args[0][0]
        assert emitted.tool_id == "fetch_page"
        assert emitted.phase == "fetching"
        assert "Fetching" in emitted.message
        assert emitted.metadata["url"] == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_returns_error_on_failure(self, tool_service, progress_emitter):
        """Should return an error string when fetch fails."""
        tool_service.execute_tool.return_value = {
            "success": False,
            "error": "404 Not Found",
        }
        tool = create_fetch_page_tool(tool_service, progress_emitter)

        result = await tool.acall(url="https://example.com/missing")

        assert "Error:" in str(result)
        assert "404 Not Found" in str(result)

    def test_tool_metadata(self, tool_service, progress_emitter):
        """Should have correct name and description for LLM consumption."""
        tool = create_fetch_page_tool(tool_service, progress_emitter)

        assert tool.metadata.name == "fetch_page"
        assert "fetch" in tool.metadata.description.lower()
        assert "page" in tool.metadata.description.lower()


class TestFetchPagesBatchTool:
    """Tests for create_fetch_pages_batch_tool.

    The tool uses SourceFetchPipeline (core/source_pipeline.py) for fetching,
    content-reranking, and context fitting. Tests mock the pipeline.
    """

    @pytest.mark.asyncio
    async def test_fetches_pages_via_source_pipeline(
        self, tool_service, progress_emitter
    ):
        """Should use SourceFetchPipeline to fetch, rerank, and fit pages."""
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(
            return_value=(
                [
                    ("https://a.com", "Page A", "Content A"),
                    ("https://b.com", "Page B", "Content B"),
                ],
                [],  # source_nodes
                {},  # allocations
            )
        )

        with patch(
            "tensortruth.core.source_pipeline.SourceFetchPipeline",
            return_value=mock_pipeline,
        ):
            tool = create_fetch_pages_batch_tool(tool_service, progress_emitter)
            result = await tool.acall(urls=["https://a.com", "https://b.com"])

        # Summary includes titles and URLs but NOT page body content
        assert "Page A" in str(result)
        assert "a.com" in str(result)
        assert "Content A" not in str(result)

    @pytest.mark.asyncio
    async def test_emits_fetching_and_fetched_progress(
        self, tool_service, progress_emitter
    ):
        """Should emit 'fetching' progress initially and 'fetched' after completion."""
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(
            return_value=(
                [("https://a.com", "Page A", "Content A")],
                [],
                {},
            )
        )

        with patch(
            "tensortruth.core.source_pipeline.SourceFetchPipeline",
            return_value=mock_pipeline,
        ):
            tool = create_fetch_pages_batch_tool(tool_service, progress_emitter)
            await tool.acall(urls=["https://a.com", "https://b.com"])

        assert progress_emitter.await_count == 2

        # First call: fetching phase
        first: ToolProgress = progress_emitter.call_args_list[0][0][0]
        assert first.tool_id == "fetch_pages_batch"
        assert first.phase == "fetching"
        assert "2 pages" in first.message

        # Second call: fetched phase with summary
        second: ToolProgress = progress_emitter.call_args_list[1][0][0]
        assert second.tool_id == "fetch_pages_batch"
        assert second.phase == "fetched"
        assert second.metadata["fitted"] == 1
        assert second.metadata["total"] == 2

    @pytest.mark.asyncio
    async def test_returns_no_pages_message_on_empty_result(
        self, tool_service, progress_emitter
    ):
        """Should return a message when no pages could be fetched."""
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=([], [], {}))

        with patch(
            "tensortruth.core.source_pipeline.SourceFetchPipeline",
            return_value=mock_pipeline,
        ):
            tool = create_fetch_pages_batch_tool(tool_service, progress_emitter)
            result = await tool.acall(urls=["https://example.com"])

        assert "No pages could be fetched" in str(result)

    @pytest.mark.asyncio
    async def test_emits_source_nodes_via_callback(
        self, tool_service, progress_emitter
    ):
        """Should pass SourceNodes to the web_result_callback."""
        from tensortruth.core.source import SourceNode, SourceStatus, SourceType

        source_node = SourceNode(
            id="s1",
            title="Page A",
            source_type=SourceType.WEB,
            url="https://a.com",
            content="Content A",
            score=0.85,
            status=SourceStatus.SUCCESS,
            content_chars=100,
        )
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(
            return_value=(
                [("https://a.com", "Page A", "Content A")],
                [source_node],
                {},
            )
        )

        captured_nodes = []

        def callback(nodes):
            captured_nodes.extend(nodes)

        with patch(
            "tensortruth.core.source_pipeline.SourceFetchPipeline",
            return_value=mock_pipeline,
        ):
            tool = create_fetch_pages_batch_tool(
                tool_service,
                progress_emitter,
                web_result_callback=callback,
            )
            await tool.acall(urls=["https://a.com"])

        assert len(captured_nodes) == 1
        assert captured_nodes[0].url == "https://a.com"

    def test_tool_metadata(self, tool_service, progress_emitter):
        """Should have correct name and description for LLM consumption."""
        tool = create_fetch_pages_batch_tool(tool_service, progress_emitter)

        assert tool.metadata.name == "fetch_pages_batch"
        assert "parallel" in tool.metadata.description.lower()


class TestRAGQueryTool:
    """Tests for create_rag_tool."""

    @pytest.fixture
    def rag_service(self):
        """Mock RAGService with a retrieve method."""
        service = MagicMock()
        service.is_loaded.return_value = True
        return service

    @pytest.mark.asyncio
    async def test_calls_rag_retrieve(self, rag_service, progress_emitter):
        """Should call RAGService.retrieve() with the query."""
        nodes = [
            NodeWithScore(node=TextNode(text="Content about transformers"), score=0.85),
        ]
        rag_service.retrieve.return_value = RAGRetrievalResult(
            source_nodes=nodes,
            confidence_level="normal",
            metrics={
                "score_distribution": {"mean": 0.85, "max": 0.85},
                "quality": {"high_confidence_ratio": 1.0},
                "coverage": {"total_chunks": 1},
            },
            condensed_query="transformers",
            num_sources=1,
        )

        tool = create_rag_tool(rag_service, progress_emitter)
        result = await tool.acall(query="What are transformers?")

        # Verify retrieve was called
        assert rag_service.retrieve.called
        call_args = rag_service.retrieve.call_args
        assert call_args[1]["query"] == "What are transformers?"

        # Verify summary contains source info but NOT chunk text
        result_str = str(result)
        assert "1 source" in result_str
        assert "confidence: normal" in result_str
        assert "Content about transformers" not in result_str

    @pytest.mark.asyncio
    async def test_emits_retrieving_progress(self, rag_service, progress_emitter):
        """Should emit a 'retrieving' phase ToolProgress."""
        rag_service.retrieve.return_value = RAGRetrievalResult(
            condensed_query="test",
        )

        tool = create_rag_tool(rag_service, progress_emitter)
        await tool.acall(query="test query")

        # Should emit at least one progress event
        assert progress_emitter.await_count >= 1
        emitted: ToolProgress = progress_emitter.call_args_list[0][0][0]
        assert emitted.tool_id == "rag"
        assert emitted.phase == "retrieving"
        assert "knowledge base" in emitted.message.lower()
        assert emitted.metadata["query"] == "test query"

    @pytest.mark.asyncio
    async def test_handles_no_sources(self, rag_service, progress_emitter):
        """Should return a clear message when no sources are found."""
        rag_service.retrieve.return_value = RAGRetrievalResult(
            confidence_level="none",
            condensed_query="obscure topic",
        )

        tool = create_rag_tool(rag_service, progress_emitter)
        result = await tool.acall(query="obscure topic")

        result_str = str(result)
        assert "No relevant sources" in result_str
        assert "none" in result_str.lower()

    @pytest.mark.asyncio
    async def test_forwards_session_context(self, rag_service, progress_emitter):
        """Should pass session params and messages to retrieve()."""
        rag_service.retrieve.return_value = RAGRetrievalResult(
            condensed_query="test",
        )

        session_params = {"model": "test-model", "confidence_cutoff": 0.5}
        session_messages = [
            {"role": "user", "content": "Previous question"},
        ]

        tool = create_rag_tool(
            rag_service,
            progress_emitter,
            session_params=session_params,
            session_messages=session_messages,
        )
        await tool.acall(query="Follow-up")

        call_args = rag_service.retrieve.call_args
        assert call_args[1]["params"] == session_params
        assert call_args[1]["session_messages"] == session_messages

    def test_tool_metadata(self, rag_service, progress_emitter):
        """Should have correct name and description for LLM consumption."""
        tool = create_rag_tool(rag_service, progress_emitter)

        assert tool.metadata.name == "rag_query"
        desc = tool.metadata.description.lower()
        assert "knowledge base" in desc
        assert "document" in desc or "paper" in desc

    @pytest.mark.asyncio
    async def test_summary_is_compact(self, rag_service, progress_emitter):
        """Summary should be compact even when source content is large."""
        long_text = "A" * 3000
        nodes = [
            NodeWithScore(node=TextNode(text=long_text), score=0.9),
        ]
        rag_service.retrieve.return_value = RAGRetrievalResult(
            source_nodes=nodes,
            confidence_level="normal",
            metrics={
                "score_distribution": {"mean": 0.9, "max": 0.9},
                "quality": {"high_confidence_ratio": 1.0},
                "coverage": {"total_chunks": 1},
            },
            condensed_query="test",
            num_sources=1,
        )

        tool = create_rag_tool(rag_service, progress_emitter)
        result = await tool.acall(query="test")

        result_str = str(result)
        # Summary should be compact — no chunk text
        assert len(result_str) < 500
        assert "1 source" in result_str


class TestCreateAllToolWrappers:
    """Tests for the convenience create_all_tool_wrappers factory."""

    def test_returns_three_web_tools_without_rag(self, tool_service, progress_emitter):
        """Should return exactly three web tool wrappers when no RAG service."""
        tools = create_all_tool_wrappers(tool_service, progress_emitter)

        assert len(tools) == 3
        names = {t.metadata.name for t in tools}
        assert names == {"web_search", "fetch_page", "fetch_pages_batch"}

    def test_returns_four_tools_with_loaded_rag(self, tool_service, progress_emitter):
        """Should include rag_query when RAG service is loaded."""
        rag_service = MagicMock()
        rag_service.is_loaded.return_value = True

        tools = create_all_tool_wrappers(
            tool_service, progress_emitter, rag_service=rag_service
        )

        assert len(tools) == 4
        names = {t.metadata.name for t in tools}
        assert names == {"rag_query", "web_search", "fetch_page", "fetch_pages_batch"}

    def test_excludes_rag_when_not_loaded(self, tool_service, progress_emitter):
        """Should NOT include rag_query when RAG service is not loaded."""
        rag_service = MagicMock()
        rag_service.is_loaded.return_value = False

        tools = create_all_tool_wrappers(
            tool_service, progress_emitter, rag_service=rag_service
        )

        assert len(tools) == 3
        names = {t.metadata.name for t in tools}
        assert "rag_query" not in names

    def test_all_tools_have_descriptions(self, tool_service, progress_emitter):
        """All tools should have non-empty descriptions for the LLM."""
        tools = create_all_tool_wrappers(tool_service, progress_emitter)

        for tool in tools:
            assert tool.metadata.description
            assert len(tool.metadata.description) > 20


class TestProgressEmitterVariants:
    """Test that both sync and async progress emitters work."""

    @pytest.mark.asyncio
    async def test_sync_emitter_works(self, tool_service):
        """Should work with a synchronous progress emitter."""
        emitted = []

        def sync_emitter(progress: ToolProgress) -> None:
            emitted.append(progress)

        tool_service.execute_tool.return_value = {"success": True, "data": "[]"}
        tool = create_web_search_tool(tool_service, sync_emitter)

        await tool.acall(query="test")

        assert len(emitted) == 1
        assert emitted[0].tool_id == "web_search"

    @pytest.mark.asyncio
    async def test_async_emitter_works(self, tool_service):
        """Should work with an async progress emitter."""
        emitted = []

        async def async_emitter(progress: ToolProgress) -> None:
            emitted.append(progress)

        tool_service.execute_tool.return_value = {"success": True, "data": "content"}
        tool = create_fetch_page_tool(tool_service, async_emitter)

        await tool.acall(url="https://example.com")

        assert len(emitted) == 1
        assert emitted[0].tool_id == "fetch_page"


class TestFullOutputCallback:
    """Tests for the full_output_callback side-channel."""

    @pytest.mark.asyncio
    async def test_rag_query_returns_summary(self, progress_emitter):
        """rag_query should return a compact summary without chunk text."""
        rag_service = MagicMock()
        rag_service.is_loaded.return_value = True
        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="Detailed content about neural networks " * 50,
                    metadata={"title": "NN Paper"},
                ),
                score=0.92,
            ),
        ]
        rag_service.retrieve.return_value = RAGRetrievalResult(
            source_nodes=nodes,
            confidence_level="high",
            condensed_query="neural networks",
            num_sources=1,
        )

        tool = create_rag_tool(rag_service, progress_emitter)
        result = await tool.acall(query="neural networks")

        result_str = str(result)
        assert "1 source" in result_str
        assert "confidence: high" in result_str
        assert "NN Paper" in result_str
        assert "0.9200" in result_str
        # Should NOT contain chunk text
        assert "Detailed content about neural" not in result_str

    @pytest.mark.asyncio
    async def test_rag_query_full_output_callback(self, progress_emitter):
        """Callback should receive full formatted result with chunk text."""
        rag_service = MagicMock()
        rag_service.is_loaded.return_value = True
        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="Detailed content about neural networks",
                    metadata={"title": "NN Paper"},
                ),
                score=0.92,
            ),
        ]
        rag_service.retrieve.return_value = RAGRetrievalResult(
            source_nodes=nodes,
            confidence_level="high",
            condensed_query="neural networks",
            num_sources=1,
        )

        captured = []

        def callback(tool_name, full_output):
            captured.append((tool_name, full_output))

        tool = create_rag_tool(
            rag_service, progress_emitter, full_output_callback=callback
        )
        await tool.acall(query="neural networks")

        assert len(captured) == 1
        assert captured[0][0] == "rag_query"
        assert "Detailed content about neural networks" in captured[0][1]

    @pytest.mark.asyncio
    async def test_fetch_page_returns_summary(self, tool_service, progress_emitter):
        """fetch_page should return a compact summary with title and char count."""
        tool_service.execute_tool.return_value = {
            "success": True,
            "data": "# My Page Title\nLots of page content here " * 100,
        }
        tool = create_fetch_page_tool(tool_service, progress_emitter)

        result = await tool.acall(url="https://example.com/article")

        result_str = str(result)
        assert "My Page Title" in result_str
        assert "chars" in result_str
        assert "example.com" in result_str
        # Should NOT contain full page body
        assert "Lots of page content" not in result_str

    @pytest.mark.asyncio
    async def test_fetch_page_full_output_callback(
        self, tool_service, progress_emitter
    ):
        """Callback should receive full page content."""
        tool_service.execute_tool.return_value = {
            "success": True,
            "data": "# Title\nFull body content here",
        }

        captured = []

        def callback(tool_name, full_output):
            captured.append((tool_name, full_output))

        tool = create_fetch_page_tool(
            tool_service, progress_emitter, full_output_callback=callback
        )
        await tool.acall(url="https://example.com")

        assert len(captured) == 1
        assert captured[0][0] == "fetch_page"
        assert "Full body content here" in captured[0][1]

    @pytest.mark.asyncio
    async def test_fetch_pages_batch_returns_summary(
        self, tool_service, progress_emitter
    ):
        """fetch_pages_batch should return summary with titles, not page bodies."""
        from tensortruth.core.source import SourceNode, SourceStatus, SourceType

        source_node = SourceNode(
            id="s1",
            title="Page A",
            source_type=SourceType.WEB,
            url="https://a.com",
            content="Full page body content " * 100,
            score=0.85,
            status=SourceStatus.SUCCESS,
            content_chars=2000,
        )
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(
            return_value=(
                [
                    ("https://a.com", "Page A", "Full page body content " * 100),
                ],
                [source_node],
                {},
            )
        )

        with patch(
            "tensortruth.core.source_pipeline.SourceFetchPipeline",
            return_value=mock_pipeline,
        ):
            tool = create_fetch_pages_batch_tool(tool_service, progress_emitter)
            result = await tool.acall(urls=["https://a.com"])

        result_str = str(result)
        assert "Page A" in result_str
        assert "a.com" in result_str
        assert "chars" in result_str
        # Should NOT contain full page body
        assert "Full page body content" not in result_str

    @pytest.mark.asyncio
    async def test_fetch_pages_batch_full_output_callback(
        self, tool_service, progress_emitter
    ):
        """Callback should receive complete content including page bodies."""
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(
            return_value=(
                [
                    ("https://a.com", "Page A", "Complete page body text"),
                ],
                [],
                {},
            )
        )

        captured = []

        def callback(tool_name, full_output):
            captured.append((tool_name, full_output))

        with patch(
            "tensortruth.core.source_pipeline.SourceFetchPipeline",
            return_value=mock_pipeline,
        ):
            tool = create_fetch_pages_batch_tool(
                tool_service, progress_emitter, full_output_callback=callback
            )
            await tool.acall(urls=["https://a.com"])

        assert len(captured) == 1
        assert captured[0][0] == "fetch_pages_batch"
        assert "Complete page body text" in captured[0][1]
