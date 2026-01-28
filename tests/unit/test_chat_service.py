"""Unit tests for ChatService routing logic."""

from unittest.mock import MagicMock

import pytest

from tensortruth.services.chat_service import ChatService
from tensortruth.services.models import RAGChunk


@pytest.mark.unit
class TestChatService:
    """Tests for ChatService routing logic."""

    def test_routes_to_llm_only_when_no_modules_no_pdfs(self):
        """Verify query_llm_only called when modules=[] and no index."""
        mock_rag_service = MagicMock()
        mock_rag_service.query_llm_only.return_value = iter(
            [RAGChunk(text="LLM response", is_complete=True, source_nodes=[])]
        )

        chat_service = ChatService(rag_service=mock_rag_service)

        # Consume generator to trigger the call
        list(
            chat_service.query(
                prompt="Hello",
                modules=[],
                params={"temperature": 0.7},
                session_messages=[],
                session_index_path=None,
            )
        )

        mock_rag_service.query_llm_only.assert_called_once_with(
            "Hello",
            {"temperature": 0.7},
            session_messages=[],
        )
        mock_rag_service.query.assert_not_called()
        mock_rag_service.needs_reload.assert_not_called()

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
            session_messages=[{"role": "user", "content": "previous"}],
        )
        mock_rag_service.query_llm_only.assert_not_called()

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
            session_messages=[],
        )
        mock_rag_service.query_llm_only.assert_not_called()

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

    def test_is_llm_only_mode_helper(self):
        """Test the is_llm_only_mode convenience method."""
        mock_rag_service = MagicMock()
        chat_service = ChatService(rag_service=mock_rag_service)

        # No modules, no PDF index -> LLM only
        assert chat_service.is_llm_only_mode([], None) is True

        # Modules present -> RAG mode
        assert chat_service.is_llm_only_mode(["pytorch"], None) is False

        # PDF index present -> RAG mode
        assert chat_service.is_llm_only_mode([], "/path/to/index") is False

        # Both present -> RAG mode
        assert chat_service.is_llm_only_mode(["pytorch"], "/path/to/index") is False

    def test_passes_session_messages_to_llm_only(self):
        """Verify session_messages are passed to query_llm_only."""
        mock_rag_service = MagicMock()
        mock_rag_service.query_llm_only.return_value = iter(
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
                modules=[],
                params={"model": "llama3"},
                session_messages=messages,
                session_index_path=None,
            )
        )

        mock_rag_service.query_llm_only.assert_called_once_with(
            "second",
            {"model": "llama3"},
            session_messages=messages,
        )

    def test_passes_session_messages_to_rag_query(self):
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
