"""Unit tests for chat-related API schemas."""

import pytest

from tensortruth.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    IntentRequest,
    IntentResponse,
    SourceNode,
    StreamDone,
    StreamSources,
    StreamStatus,
    StreamThinking,
    StreamToken,
)


@pytest.mark.unit
class TestChatRequest:
    """Tests for ChatRequest schema."""

    def test_valid_prompt(self):
        """ChatRequest accepts a valid prompt."""
        request = ChatRequest(prompt="What is machine learning?")
        assert request.prompt == "What is machine learning?"

    def test_empty_prompt_rejected(self):
        """ChatRequest rejects empty prompt."""
        with pytest.raises(ValueError):
            ChatRequest(prompt="")


@pytest.mark.unit
class TestSourceNode:
    """Tests for SourceNode schema."""

    def test_creation(self):
        """SourceNode can be created with expected fields."""
        node = SourceNode(
            text="Some context text",
            score=0.95,
            metadata={"source": "test.pdf"},
        )
        assert node.text == "Some context text"
        assert node.score == 0.95
        assert node.metadata["source"] == "test.pdf"

    def test_defaults(self):
        """SourceNode has sensible defaults."""
        node = SourceNode(text="Context")
        assert node.score is None
        assert node.metadata == {}


@pytest.mark.unit
class TestChatResponse:
    """Tests for ChatResponse schema."""

    def test_creation(self):
        """ChatResponse can be created."""
        response = ChatResponse(
            content="Here is the answer...",
            sources=[SourceNode(text="source 1")],
            confidence_level="high",
        )
        assert response.content == "Here is the answer..."
        assert len(response.sources) == 1
        assert response.confidence_level == "high"

    def test_defaults(self):
        """ChatResponse has sensible defaults."""
        response = ChatResponse(content="Answer")
        assert response.sources == []
        assert response.confidence_level == "normal"


@pytest.mark.unit
class TestStreamToken:
    """Tests for StreamToken WebSocket message schema."""

    def test_type_is_token(self):
        """StreamToken always has type 'token'."""
        msg = StreamToken(content="Hello")
        assert msg.type == "token"
        assert msg.content == "Hello"


@pytest.mark.unit
class TestStreamSources:
    """Tests for StreamSources WebSocket message schema."""

    def test_type_is_sources(self):
        """StreamSources always has type 'sources'."""
        msg = StreamSources(data=[SourceNode(text="src")])
        assert msg.type == "sources"
        assert len(msg.data) == 1


@pytest.mark.unit
class TestStreamDone:
    """Tests for StreamDone WebSocket message schema."""

    def test_type_is_done(self):
        """StreamDone always has type 'done'."""
        msg = StreamDone(content="Full response", confidence_level="normal")
        assert msg.type == "done"
        assert msg.content == "Full response"

    def test_defaults(self):
        """StreamDone has default confidence level."""
        msg = StreamDone(content="Response")
        assert msg.confidence_level == "normal"


@pytest.mark.unit
class TestStreamThinking:
    """Tests for StreamThinking WebSocket message schema."""

    def test_type_is_thinking(self):
        """StreamThinking always has type 'thinking'."""
        msg = StreamThinking(content="Let me analyze...")
        assert msg.type == "thinking"
        assert msg.content == "Let me analyze..."


@pytest.mark.unit
class TestStreamStatus:
    """Tests for StreamStatus WebSocket message schema."""

    def test_type_is_status(self):
        """StreamStatus always has type 'status'."""
        msg = StreamStatus(status="retrieving")
        assert msg.type == "status"
        assert msg.status == "retrieving"

    def test_valid_statuses(self):
        """StreamStatus accepts all valid status values."""
        for status in ["retrieving", "thinking", "generating"]:
            msg = StreamStatus(status=status)
            assert msg.status == status


@pytest.mark.unit
class TestIntentRequest:
    """Tests for IntentRequest schema."""

    def test_creation(self):
        """IntentRequest can be created."""
        request = IntentRequest(
            message="search for AI news",
            recent_messages=[{"role": "user", "content": "hi"}],
        )
        assert request.message == "search for AI news"
        assert len(request.recent_messages) == 1

    def test_defaults(self):
        """IntentRequest has sensible defaults."""
        request = IntentRequest(message="hello")
        assert request.recent_messages == []


@pytest.mark.unit
class TestIntentResponse:
    """Tests for IntentResponse schema."""

    def test_creation_chat(self):
        """IntentResponse for chat intent."""
        response = IntentResponse(intent="chat", query=None, reason="no_triggers")
        assert response.intent == "chat"
        assert response.query is None

    def test_creation_browse(self):
        """IntentResponse for browse intent."""
        response = IntentResponse(
            intent="browse", query="AI news", reason="explicit_browse"
        )
        assert response.intent == "browse"
        assert response.query == "AI news"

    def test_creation_search(self):
        """IntentResponse for search intent."""
        response = IntentResponse(
            intent="search", query="python features", reason="explicit_search"
        )
        assert response.intent == "search"
        assert response.query == "python features"
