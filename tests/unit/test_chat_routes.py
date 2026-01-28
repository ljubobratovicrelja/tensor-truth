"""Unit tests for tensortruth.api.routes.chat module."""

from unittest.mock import MagicMock

import pytest

from tensortruth.api.routes.chat import ChatContext, _extract_sources
from tensortruth.api.schemas.chat import SourceNode


@pytest.mark.unit
class TestExtractSources:
    """Tests for _extract_sources helper function."""

    def test_extract_from_node_with_score(self):
        """Test extraction from LlamaIndex NodeWithScore objects."""
        # Mock NodeWithScore with inner node
        inner_node = MagicMock()
        inner_node.get_content.return_value = (
            "This is the merged content from retriever"
        )

        node_with_score = MagicMock()
        node_with_score.node = inner_node
        node_with_score.score = 0.95
        node_with_score.metadata = {"source": "test.pdf", "page": 5}

        result = _extract_sources([node_with_score])

        assert len(result) == 1
        assert isinstance(result[0], SourceNode)
        assert result[0].text == "This is the merged content from retriever"
        assert result[0].score == 0.95
        assert result[0].metadata["source"] == "test.pdf"
        assert result[0].metadata["page"] == 5

    def test_empty_source_nodes_list(self):
        """Test that empty list returns empty list."""
        result = _extract_sources([])

        assert result == []

    def test_metadata_extraction_all_fields(self):
        """Test metadata extraction with various field types."""
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Content"

        node = MagicMock()
        node.node = inner_node
        node.score = 0.8
        node.metadata = {
            "doc_type": "paper",
            "source": "arxiv",
            "title": "Neural Networks Paper",
            "file_name": "paper.pdf",
            "page_label": "3",
            "authors": ["Author A", "Author B"],
        }

        result = _extract_sources([node])

        assert result[0].metadata["doc_type"] == "paper"
        assert result[0].metadata["source"] == "arxiv"
        assert result[0].metadata["title"] == "Neural Networks Paper"
        assert result[0].metadata["authors"] == ["Author A", "Author B"]

    def test_score_field_mapping(self):
        """Test score is correctly mapped from node."""
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Text"

        node = MagicMock()
        node.node = inner_node
        node.score = 0.7654
        node.metadata = {}

        result = _extract_sources([node])

        assert result[0].score == 0.7654

    def test_missing_score_attribute(self):
        """Test handling of nodes without score attribute.

        When a node has no score, effective_score returns a status-based default:
        - SUCCESS (default status): 1.0
        - Other statuses: 0.0
        """
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Text"

        node = MagicMock(spec=["node", "metadata"])  # No 'score' in spec
        node.node = inner_node
        node.metadata = {}
        # hasattr(node, 'score') will return False

        result = _extract_sources([node])

        # No explicit score + SUCCESS status = effective_score of 1.0
        assert result[0].score == 1.0

    def test_missing_metadata_attribute(self):
        """Test handling of nodes without metadata attribute.

        With SourceConverter, unified source fields are added to metadata
        even when the original node has no metadata.
        """
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Text"

        node = MagicMock(spec=["node", "score"])  # No 'metadata' in spec
        node.node = inner_node
        node.score = 0.5

        result = _extract_sources([node])

        # SourceConverter adds unified source fields to metadata
        assert "doc_type" in result[0].metadata
        assert result[0].metadata["fetch_status"] == "success"

    def test_fallback_to_text_attribute(self):
        """Test fallback to .text when get_content() not available."""
        # Node without get_content method
        inner_node = MagicMock(spec=["text"])
        inner_node.text = "Text from .text attribute"

        node = MagicMock()
        node.node = inner_node
        node.text = "Fallback text"
        node.score = 0.6
        node.metadata = {}

        result = _extract_sources([node])

        # Should use node.text as fallback
        assert result[0].text == "Fallback text"

    def test_fallback_to_str_conversion(self):
        """Test fallback to str(node) when no text methods available."""
        # Node without get_content or text
        inner_node = MagicMock(spec=[])

        node = MagicMock(spec=["node", "score", "metadata"])
        node.node = inner_node
        node.score = 0.4
        node.metadata = {}

        result = _extract_sources([node])

        # Should convert to string
        assert isinstance(result[0].text, str)

    def test_raw_node_without_inner_node(self):
        """Test handling of raw nodes without .node wrapper."""
        # Direct node without NodeWithScore wrapper
        node = MagicMock()
        # No .node attribute - simulate raw node
        del node.node
        node.get_content = MagicMock(return_value="Direct content")
        node.score = 0.9
        node.metadata = {"source": "direct"}

        result = _extract_sources([node])

        assert len(result) == 1
        assert result[0].text == "Direct content"
        assert result[0].score == 0.9

    def test_multiple_source_nodes(self):
        """Test extraction from multiple source nodes."""
        nodes = []
        for i in range(3):
            inner_node = MagicMock()
            inner_node.get_content.return_value = f"Content {i}"

            node = MagicMock()
            node.node = inner_node
            node.score = 0.9 - i * 0.1
            node.metadata = {"index": i}
            nodes.append(node)

        result = _extract_sources(nodes)

        assert len(result) == 3
        assert result[0].text == "Content 0"
        assert result[0].score == 0.9
        assert result[1].text == "Content 1"
        assert result[1].score == 0.8
        assert result[2].text == "Content 2"
        assert result[2].score == 0.7

    def test_exception_in_node_propagates(self):
        """Test that exceptions in node processing propagate (not silently swallowed).

        source_nodes always come from LlamaIndex retriever, so exceptions
        indicate real bugs that should be surfaced, not hidden.
        """
        # Bad node that raises exception
        bad_inner = MagicMock()
        bad_inner.get_content.side_effect = Exception("Extraction error")
        bad_node = MagicMock()
        bad_node.node = bad_inner
        bad_node.score = 0.5
        bad_node.metadata = {}

        with pytest.raises(Exception, match="Extraction error"):
            _extract_sources([bad_node])

    def test_none_metadata_handled_gracefully(self):
        """Test that None metadata is handled gracefully by SourceConverter.

        With SourceConverter, None metadata is treated as empty dict
        and unified source fields are added.
        """
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Content"

        node = MagicMock()
        node.node = inner_node
        node.score = 0.7
        node.metadata = None  # Explicitly None - now handled gracefully

        result = _extract_sources([node])

        # Node should be extracted with unified metadata fields
        assert len(result) == 1
        assert result[0].text == "Content"
        assert result[0].score == 0.7
        assert "doc_type" in result[0].metadata
        assert result[0].metadata["fetch_status"] == "success"

    def test_get_content_preferred_over_text(self):
        """Test that get_content() is preferred when both available."""
        inner_node = MagicMock()
        inner_node.get_content.return_value = "Merged parent content"
        inner_node.text = "Leaf node text"

        node = MagicMock()
        node.node = inner_node
        node.score = 0.8
        node.metadata = {}

        result = _extract_sources([node])

        # get_content() should be preferred (returns merged content)
        assert result[0].text == "Merged parent content"


@pytest.mark.unit
class TestChatContext:
    """Tests for ChatContext dataclass."""

    def test_is_llm_only_mode_no_modules_no_pdfs(self):
        """LLM-only mode when modules=[] and no PDF index."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=[],
            params={},
            session_messages=[],
            session_index_path=None,
        )
        assert context.is_llm_only_mode is True

    def test_is_rag_mode_when_modules_present(self):
        """RAG mode when modules=['pytorch']."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=["pytorch"],
            params={},
            session_messages=[],
            session_index_path=None,
        )
        assert context.is_llm_only_mode is False

    def test_is_rag_mode_when_pdf_index_present(self):
        """RAG mode when PDF index exists (even without modules)."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=[],
            params={},
            session_messages=[],
            session_index_path="/path/to/index",
        )
        assert context.is_llm_only_mode is False

    def test_is_rag_mode_when_both_present(self):
        """RAG mode when both modules and PDF index exist."""
        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=["pytorch"],
            params={},
            session_messages=[],
            session_index_path="/path/to/index",
        )
        assert context.is_llm_only_mode is False

    def test_from_session_creates_context(self):
        """Test from_session factory method creates valid context."""
        session = {
            "modules": ["pytorch"],
            "params": {"temperature": 0.7},
            "messages": [{"role": "user", "content": "previous"}],
        }

        # Mock PDF service that returns an index path
        mock_pdf_service = MagicMock()
        mock_pdf_service.get_index_path.return_value = "/path/to/index"

        context = ChatContext.from_session(
            session_id="test-123",
            prompt="current question",
            session=session,
            pdf_service=mock_pdf_service,
        )

        assert context.session_id == "test-123"
        assert context.prompt == "current question"
        assert context.modules == ["pytorch"]
        assert context.params == {"temperature": 0.7}
        assert context.session_messages == [{"role": "user", "content": "previous"}]
        assert context.session_index_path == "/path/to/index"

    def test_from_session_handles_no_index(self):
        """Test from_session handles None index path."""
        session = {
            "modules": [],
            "params": {},
            "messages": [],
        }

        mock_pdf_service = MagicMock()
        mock_pdf_service.get_index_path.return_value = None

        context = ChatContext.from_session(
            session_id="test-123",
            prompt="question",
            session=session,
            pdf_service=mock_pdf_service,
        )

        assert context.session_index_path is None
        assert context.is_llm_only_mode is True

    def test_from_session_handles_missing_modules(self):
        """Test from_session handles session without modules key."""
        session = {
            "params": {},
            "messages": [],
        }

        mock_pdf_service = MagicMock()
        mock_pdf_service.get_index_path.return_value = None

        context = ChatContext.from_session(
            session_id="test-123",
            prompt="question",
            session=session,
            pdf_service=mock_pdf_service,
        )

        assert context.modules == []
        assert context.is_llm_only_mode is True


@pytest.mark.unit
class TestChatEngineReload:
    """Tests for RAG engine reload logic via _execute_chat_query."""

    def test_calls_needs_reload_with_correct_args(self):
        """Verify needs_reload() receives modules, params, session_index_path."""
        from tensortruth.api.routes.chat import _execute_chat_query
        from tensortruth.services.models import RAGChunk

        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [RAGChunk(text="response", is_complete=True, source_nodes=[])]
        )

        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=["pytorch", "numpy"],
            params={"temperature": 0.5},
            session_messages=[],
            session_index_path="/path/to/pdf/index",
        )

        _execute_chat_query(mock_rag_service, context)

        mock_rag_service.needs_reload.assert_called_once_with(
            ["pytorch", "numpy"],
            {"temperature": 0.5},
            "/path/to/pdf/index",
        )

    def test_loads_engine_when_needs_reload_true(self):
        """Verify load_engine() called when needs_reload() returns True."""
        from tensortruth.api.routes.chat import _execute_chat_query
        from tensortruth.services.models import RAGChunk

        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = True
        mock_rag_service.query.return_value = iter(
            [RAGChunk(text="response", is_complete=True, source_nodes=[])]
        )

        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=["pytorch"],
            params={"temperature": 0.7},
            session_messages=[],
            session_index_path="/path/to/index",
        )

        _execute_chat_query(mock_rag_service, context)

        mock_rag_service.load_engine.assert_called_once_with(
            modules=["pytorch"],
            params={"temperature": 0.7},
            session_index_path="/path/to/index",
        )

    def test_skips_load_when_needs_reload_false(self):
        """Verify load_engine() NOT called when needs_reload() returns False."""
        from tensortruth.api.routes.chat import _execute_chat_query
        from tensortruth.services.models import RAGChunk

        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [RAGChunk(text="response", is_complete=True, source_nodes=[])]
        )

        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=["pytorch"],
            params={},
            session_messages=[],
            session_index_path=None,
        )

        _execute_chat_query(mock_rag_service, context)

        mock_rag_service.load_engine.assert_not_called()

    def test_uses_query_llm_only_when_llm_only_mode(self):
        """Verify query_llm_only() is used in LLM-only mode."""
        from tensortruth.api.routes.chat import _execute_chat_query
        from tensortruth.services.models import RAGChunk

        mock_rag_service = MagicMock()
        # Text and is_complete are sent in separate chunks (matching real behavior)
        mock_rag_service.query_llm_only.return_value = iter(
            [
                RAGChunk(text="LLM response"),
                RAGChunk(is_complete=True, source_nodes=[]),
            ]
        )

        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=[],
            params={"temperature": 0.8},
            session_messages=[{"role": "user", "content": "previous"}],
            session_index_path=None,
        )

        response, sources, metrics = _execute_chat_query(mock_rag_service, context)

        mock_rag_service.query_llm_only.assert_called_once_with(
            "Hello",
            {"temperature": 0.8},
            session_messages=[{"role": "user", "content": "previous"}],
        )
        mock_rag_service.query.assert_not_called()
        mock_rag_service.needs_reload.assert_not_called()
        assert response == "LLM response"

    def test_uses_query_when_rag_mode(self):
        """Verify query() is used in RAG mode."""
        from tensortruth.api.routes.chat import _execute_chat_query
        from tensortruth.services.models import RAGChunk

        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        # Text and is_complete are sent in separate chunks (matching real behavior)
        mock_rag_service.query.return_value = iter(
            [
                RAGChunk(text="RAG response"),
                RAGChunk(is_complete=True, source_nodes=[]),
            ]
        )

        context = ChatContext(
            session_id="test-session",
            prompt="Hello",
            modules=["pytorch"],
            params={},
            session_messages=[{"role": "user", "content": "previous"}],
            session_index_path=None,
        )

        response, sources, metrics = _execute_chat_query(mock_rag_service, context)

        mock_rag_service.query.assert_called_once_with(
            "Hello",
            session_messages=[{"role": "user", "content": "previous"}],
        )
        mock_rag_service.query_llm_only.assert_not_called()
        assert response == "RAG response"

    def test_accumulates_response_from_multiple_chunks(self):
        """Test that response is accumulated from multiple text chunks."""
        from tensortruth.api.routes.chat import _execute_chat_query
        from tensortruth.services.models import RAGChunk

        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [
                RAGChunk(text="Hello "),
                RAGChunk(text="world"),
                RAGChunk(text="!", is_complete=True, source_nodes=[]),
            ]
        )

        context = ChatContext(
            session_id="test-session",
            prompt="Greet me",
            modules=["pytorch"],
            params={},
            session_messages=[],
            session_index_path=None,
        )

        response, sources, metrics = _execute_chat_query(mock_rag_service, context)

        assert response == "Hello world"

    def test_extracts_sources_and_metrics_from_final_chunk(self):
        """Test that sources and metrics are extracted from is_complete chunk."""
        from tensortruth.api.routes.chat import _execute_chat_query
        from tensortruth.services.models import RAGChunk

        mock_source_node = MagicMock()
        mock_source_node.node = MagicMock()
        mock_source_node.node.get_content.return_value = "Source content"
        mock_source_node.score = 0.88
        mock_source_node.metadata = {"source": "test.pdf"}

        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [
                RAGChunk(text="Response"),
                RAGChunk(
                    is_complete=True,
                    source_nodes=[mock_source_node],
                    metrics={"mean_score": 0.88},
                ),
            ]
        )

        context = ChatContext(
            session_id="test-session",
            prompt="Question",
            modules=["pytorch"],
            params={},
            session_messages=[],
            session_index_path=None,
        )

        response, sources, metrics = _execute_chat_query(mock_rag_service, context)

        assert response == "Response"
        assert len(sources) == 1
        assert sources[0].text == "Source content"
        assert sources[0].score == 0.88
        assert metrics == {"mean_score": 0.88}
