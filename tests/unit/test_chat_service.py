"""Unit tests for ChatService routing logic."""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.services.chat_service import ChatResult, ChatService
from tensortruth.services.models import RAGChunk


@pytest.mark.unit
class TestChatService:
    """Tests for ChatService unified query routing."""

    def test_zero_modules_calls_unified_query(self):
        """Verify modules=[] and no index still calls needs_reload() and query()."""
        mock_rag_service = MagicMock()
        # needs_reload returns False for zero modules (nothing to load)
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [RAGChunk(text="LLM response", is_complete=True, source_nodes=[])]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        list(
            chat_service.query(
                prompt="Hello",
                modules=[],
                params={"temperature": 0.7},
                session_messages=[],
                session_index_path=None,
            )
        )

        mock_rag_service.needs_reload.assert_called_once_with(
            [], {"temperature": 0.7}, None
        )
        mock_rag_service.query.assert_called_once_with(
            "Hello",
            {"temperature": 0.7},
            session_messages=[],
        )

    def test_routes_to_rag_when_modules_present(self):
        """Verify query called when modules provided."""
        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [RAGChunk(text="RAG response", is_complete=True, source_nodes=[])]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        list(
            chat_service.query(
                prompt="Hello",
                modules=["pytorch"],
                params={},
                session_messages=[{"role": "user", "content": "previous"}],
                session_index_path=None,
            )
        )

        mock_rag_service.query.assert_called_once_with(
            "Hello",
            {},
            session_messages=[{"role": "user", "content": "previous"}],
        )

    def test_routes_to_rag_when_pdf_index_present(self):
        """Verify query called when session_index_path provided."""
        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [RAGChunk(text="RAG response", is_complete=True, source_nodes=[])]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        list(
            chat_service.query(
                prompt="Hello",
                modules=[],
                params={},
                session_messages=[],
                session_index_path="/path/to/pdf/index",
            )
        )

        mock_rag_service.query.assert_called_once_with(
            "Hello",
            {},
            session_messages=[],
        )

    def test_calls_load_engine_when_needs_reload(self):
        """Verify load_engine called when needs_reload returns True."""
        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = True
        mock_rag_service.query.return_value = iter(
            [RAGChunk(text="RAG response", is_complete=True, source_nodes=[])]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        chunks = list(
            chat_service.query(
                prompt="Hello",
                modules=["pytorch"],
                params={"temperature": 0.7},
                session_messages=[],
                session_index_path="/path/to/index",
            )
        )

        # First chunk should be loading_models status
        assert chunks[0].status == "loading_models"

        mock_rag_service.needs_reload.assert_called_once_with(
            ["pytorch"],
            {"temperature": 0.7},
            "/path/to/index",
        )
        mock_rag_service.load_engine.assert_called_once_with(
            modules=["pytorch"],
            params={"temperature": 0.7},
            session_index_path="/path/to/index",
        )

    def test_skips_load_engine_when_no_reload_needed(self):
        """Verify load_engine NOT called when needs_reload returns False."""
        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [RAGChunk(text="RAG response", is_complete=True, source_nodes=[])]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        list(
            chat_service.query(
                prompt="Hello",
                modules=["pytorch"],
                params={},
                session_messages=[],
                session_index_path=None,
            )
        )

        mock_rag_service.needs_reload.assert_called_once()
        mock_rag_service.load_engine.assert_not_called()

    def test_passes_session_messages_to_query(self):
        """Verify session_messages are passed to query."""
        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [RAGChunk(text="response", is_complete=True, source_nodes=[])]
        )

        chat_service = ChatService(rag_service=mock_rag_service)
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
        ]

        list(
            chat_service.query(
                prompt="second",
                modules=["pytorch"],
                params={},
                session_messages=messages,
                session_index_path=None,
            )
        )

        mock_rag_service.query.assert_called_once_with(
            "second",
            {},
            session_messages=messages,
        )

    def test_returns_generator_from_rag_service(self):
        """Verify ChatService returns the generator from RAGService."""
        mock_rag_service = MagicMock()
        expected_chunks = [
            RAGChunk(text="Hello "),
            RAGChunk(text="world"),
            RAGChunk(text="!", is_complete=True, source_nodes=[]),
        ]
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(expected_chunks)

        chat_service = ChatService(rag_service=mock_rag_service)

        result = list(
            chat_service.query(
                prompt="Hello",
                modules=["pytorch"],
                params={},
                session_messages=[],
                session_index_path=None,
            )
        )

        assert len(result) == 3
        assert result[0].text == "Hello "
        assert result[1].text == "world"
        assert result[2].text == "!"
        assert result[2].is_complete is True

    def test_yields_loading_models_status_when_reload_needed(self):
        """Verify loading_models status is yielded before engine load."""
        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = True
        mock_rag_service.query.return_value = iter(
            [
                RAGChunk(status="retrieving"),
                RAGChunk(text="response"),
                RAGChunk(is_complete=True, source_nodes=[]),
            ]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        chunks = list(
            chat_service.query(
                prompt="Hello",
                modules=["pytorch"],
                params={},
                session_messages=[],
                session_index_path=None,
            )
        )

        # First chunk should be loading_models, then RAG chunks follow
        assert chunks[0].status == "loading_models"
        assert chunks[1].status == "retrieving"
        assert chunks[2].text == "response"
        assert chunks[3].is_complete is True

    def test_no_loading_models_when_no_reload_needed(self):
        """Verify no loading_models status when engine doesn't need reload."""
        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [
                RAGChunk(status="retrieving"),
                RAGChunk(text="response"),
                RAGChunk(is_complete=True, source_nodes=[]),
            ]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        chunks = list(
            chat_service.query(
                prompt="Hello",
                modules=["pytorch"],
                params={},
                session_messages=[],
                session_index_path=None,
            )
        )

        # No loading_models status, starts directly with retrieving
        assert chunks[0].status == "retrieving"
        assert len(chunks) == 3


@pytest.mark.unit
class TestChatServiceExecute:
    """Tests for ChatService.execute() non-streaming method."""

    def test_execute_returns_chat_result(self):
        """Verify execute() returns a ChatResult with clean types."""
        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [
                RAGChunk(text="Hello "),
                RAGChunk(text="world"),
                RAGChunk(
                    is_complete=True,
                    source_nodes=[],
                    metrics={"mean_score": 0.88},
                ),
            ]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        result = chat_service.execute(
            prompt="Hello",
            modules=["pytorch"],
            params={},
            session_messages=[],
            session_index_path=None,
        )

        assert isinstance(result, ChatResult)
        assert result.response == "Hello world"
        assert result.metrics == {"mean_score": 0.88}

    def test_execute_zero_modules(self):
        """Verify execute() works with zero modules (unified query path)."""
        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [
                RAGChunk(text="LLM response"),
                RAGChunk(is_complete=True, source_nodes=[]),
            ]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        result = chat_service.execute(
            prompt="Hello",
            modules=[],
            params={"model": "llama3"},
            session_messages=[],
            session_index_path=None,
        )

        assert result.response == "LLM response"
        mock_rag_service.query.assert_called_once_with(
            "Hello",
            {"model": "llama3"},
            session_messages=[],
        )

    @patch("tensortruth.services.chat_service.SourceConverter")
    def test_execute_extracts_sources_to_api_format(self, mock_converter):
        """Verify execute() converts source nodes to API dicts."""
        # Setup mock source node
        mock_source_node = MagicMock()
        mock_source_node.score = 0.95

        # Setup SourceConverter mocks
        mock_unified = MagicMock()
        mock_converter.from_rag_node.return_value = mock_unified
        mock_converter.to_api_schema.return_value = {
            "text": "Source content",
            "score": 0.95,
            "metadata": {"source": "test.pdf"},
        }

        mock_rag_service = MagicMock()
        mock_rag_service.needs_reload.return_value = False
        mock_rag_service.query.return_value = iter(
            [
                RAGChunk(text="Response"),
                RAGChunk(
                    is_complete=True,
                    source_nodes=[mock_source_node],
                    metrics={"mean_score": 0.95},
                ),
            ]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        result = chat_service.execute(
            prompt="Question",
            modules=["pytorch"],
            params={},
            session_messages=[],
            session_index_path=None,
        )

        # Verify source was converted
        assert len(result.sources) == 1
        assert result.sources[0]["text"] == "Source content"
        assert result.sources[0]["score"] == 0.95
        mock_converter.from_rag_node.assert_called_once()
        mock_converter.to_api_schema.assert_called_once()


@pytest.mark.unit
class TestChatServiceExtractSources:
    """Tests for ChatService.extract_sources() method."""

    @patch("tensortruth.services.chat_service.SourceConverter")
    def test_extract_sources_converts_nodes(self, mock_converter):
        """Verify extract_sources() uses SourceConverter correctly."""
        mock_node1 = MagicMock()
        mock_node2 = MagicMock()

        mock_unified1 = MagicMock()
        mock_unified2 = MagicMock()

        mock_converter.from_rag_node.side_effect = [mock_unified1, mock_unified2]
        mock_converter.to_api_schema.side_effect = [
            {"text": "Source 1", "score": 0.9, "metadata": {}},
            {"text": "Source 2", "score": 0.8, "metadata": {}},
        ]

        mock_rag_service = MagicMock()
        chat_service = ChatService(rag_service=mock_rag_service)

        result = chat_service.extract_sources([mock_node1, mock_node2])

        assert len(result) == 2
        assert result[0]["text"] == "Source 1"
        assert result[1]["text"] == "Source 2"

    def test_extract_sources_empty_list(self):
        """Verify extract_sources() handles empty list."""
        mock_rag_service = MagicMock()
        chat_service = ChatService(rag_service=mock_rag_service)

        result = chat_service.extract_sources([])

        assert result == []
