"""Unit tests for service layer models."""

import pytest

from tensortruth.services.models import (
    IntentResult,
    PDFMetadata,
    RAGChunk,
    RAGResponse,
    SessionData,
    SessionInfo,
)


class TestSessionData:
    """Tests for SessionData dataclass."""

    def test_creation(self):
        """SessionData can be created with expected fields."""
        data = SessionData(
            current_id="test-id",
            sessions={"test-id": {"title": "Test"}},
        )

        assert data.current_id == "test-id"
        assert "test-id" in data.sessions

    def test_default_sessions(self):
        """SessionData defaults to empty sessions dict."""
        data = SessionData(current_id=None)

        assert data.sessions == {}

    def test_to_dict(self):
        """SessionData converts to dict correctly."""
        data = SessionData(
            current_id="test-id",
            sessions={"test-id": {"title": "Test"}},
        )

        result = data.to_dict()

        assert result == {
            "current_id": "test-id",
            "sessions": {"test-id": {"title": "Test"}},
        }

    def test_from_dict(self):
        """SessionData can be created from dict."""
        raw_data = {
            "current_id": "test-id",
            "sessions": {"test-id": {"title": "Test"}},
        }

        data = SessionData.from_dict(raw_data)

        assert data.current_id == "test-id"
        assert data.sessions["test-id"]["title"] == "Test"

    def test_from_dict_handles_missing_fields(self):
        """SessionData from_dict handles missing fields."""
        data = SessionData.from_dict({})

        assert data.current_id is None
        assert data.sessions == {}


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_creation(self):
        """SessionInfo can be created with expected fields."""
        info = SessionInfo(
            session_id="test-id",
            title="Test Session",
            created_at="2024-01-01",
            messages=[{"role": "user", "content": "hi"}],
            modules=["mod1"],
            params={"temp": 0.5},
        )

        assert info.session_id == "test-id"
        assert info.title == "Test Session"
        assert len(info.messages) == 1

    def test_to_dict(self):
        """SessionInfo converts to dict correctly."""
        info = SessionInfo(
            session_id="test-id",
            title="Test",
            created_at="2024-01-01",
        )

        result = info.to_dict()

        assert result["title"] == "Test"
        assert "session_id" not in result  # ID not in dict

    def test_from_dict(self):
        """SessionInfo can be created from dict."""
        raw_data = {
            "title": "From Dict",
            "created_at": "2024-01-01",
            "messages": [],
        }

        info = SessionInfo.from_dict("test-id", raw_data)

        assert info.session_id == "test-id"
        assert info.title == "From Dict"


class TestIntentResult:
    """Tests for IntentResult dataclass."""

    def test_creation_chat(self):
        """IntentResult can be created for chat intent."""
        result = IntentResult(
            intent="chat",
            query=None,
            reason="no_triggers",
        )

        assert result.intent == "chat"
        assert result.query is None

    def test_creation_browse(self):
        """IntentResult can be created for browse intent."""
        result = IntentResult(
            intent="browse",
            query="AI news",
            reason="explicit_browse",
        )

        assert result.intent == "browse"
        assert result.query == "AI news"

    def test_creation_search(self):
        """IntentResult can be created for search intent."""
        result = IntentResult(
            intent="search",
            query="python features",
            reason="explicit_search",
        )

        assert result.intent == "search"
        assert result.query == "python features"


class TestPDFMetadata:
    """Tests for PDFMetadata dataclass."""

    def test_creation(self):
        """PDFMetadata can be created with expected fields."""
        meta = PDFMetadata(
            pdf_id="pdf_abc123",
            filename="document.pdf",
            path="/path/to/pdf",
            file_size=1024,
            page_count=10,
        )

        assert meta.pdf_id == "pdf_abc123"
        assert meta.filename == "document.pdf"
        assert meta.page_count == 10

    def test_to_dict(self):
        """PDFMetadata converts to dict correctly."""
        meta = PDFMetadata(
            pdf_id="pdf_abc123",
            filename="document.pdf",
            path="/path/to/pdf",
            file_size=1024,
            page_count=10,
        )

        result = meta.to_dict()

        assert result["id"] == "pdf_abc123"
        assert result["filename"] == "document.pdf"


class TestRAGModels:
    """Tests for RAG-related models."""

    def test_rag_chunk_creation(self):
        """RAGChunk can be created."""
        chunk = RAGChunk(
            text="Some response text",
            is_complete=False,
        )

        assert chunk.text == "Some response text"
        assert chunk.is_complete is False
        assert chunk.source_nodes == []

    def test_rag_chunk_with_sources(self):
        """RAGChunk can include source nodes."""
        chunk = RAGChunk(
            text="",
            source_nodes=["node1", "node2"],
            is_complete=True,
        )

        assert len(chunk.source_nodes) == 2
        assert chunk.is_complete is True

    def test_rag_chunk_with_thinking(self):
        """RAGChunk can include thinking content."""
        chunk = RAGChunk(
            thinking="Let me analyze this step by step...",
        )

        assert chunk.thinking == "Let me analyze this step by step..."
        assert chunk.text == ""
        assert chunk.status is None

    def test_rag_chunk_with_status(self):
        """RAGChunk can include pipeline status."""
        for status in ["retrieving", "thinking", "generating"]:
            chunk = RAGChunk(status=status)

            assert chunk.status == status
            assert chunk.text == ""
            assert chunk.thinking is None

    def test_rag_chunk_defaults(self):
        """RAGChunk has sensible defaults."""
        chunk = RAGChunk()

        assert chunk.text == ""
        assert chunk.thinking is None
        assert chunk.source_nodes == []
        assert chunk.is_complete is False
        assert chunk.status is None

    def test_rag_response_creation(self):
        """RAGResponse can be created."""
        response = RAGResponse(
            text="Full response",
            source_nodes=["node1"],
            confidence_level="normal",
        )

        assert response.text == "Full response"
        assert response.confidence_level == "normal"

    def test_rag_response_defaults(self):
        """RAGResponse has sensible defaults."""
        response = RAGResponse(text="Response")

        assert response.source_nodes == []
        assert response.confidence_level == "normal"
