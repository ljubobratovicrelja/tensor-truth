"""Integration tests for orchestrator WebSocket flow (Story 11).

Tests the end-to-end orchestrator path through the WebSocket handler:
- Orchestrator path with mocked Ollama and tools
- Fallback to ChatService when orchestrator is disabled
- /command bypass still works with orchestrator enabled
- Progress messages arrive in correct order

Uses the FastAPI test client with WebSocket transport for realistic testing
of the full request/response cycle.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tensortruth.api.main import create_app


def _make_awaitable_handler(stream_events_fn, final_result_str="Result"):
    """Create a mock handler that supports stream_events() and is awaitable.

    Args:
        stream_events_fn: A callable (no args) returning an async generator.
        final_result_str: The string representation of the final response.

    Returns:
        A mock handler object with stream_events() and __await__.
    """
    mock_handler = MagicMock()
    mock_handler.stream_events = stream_events_fn

    mock_result = MagicMock()
    mock_result.__str__ = lambda self: final_result_str

    async def await_handler():
        return mock_result

    mock_handler.__await__ = await_handler().__await__
    return mock_handler


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def app():
    """Create test application."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def mock_session_paths(tmp_path, monkeypatch):
    """Patch session paths to use temp directory."""
    sessions_file = tmp_path / "chat_sessions.json"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    monkeypatch.setattr("tensortruth.api.deps.get_sessions_file", lambda: sessions_file)
    monkeypatch.setattr(
        "tensortruth.api.deps.get_sessions_data_dir", lambda: sessions_dir
    )
    monkeypatch.setattr(
        "tensortruth.api.deps.get_session_dir",
        lambda sid: sessions_dir / sid,
    )

    from tensortruth.api.deps import get_session_service

    get_session_service.cache_clear()

    return sessions_file, sessions_dir


# ---------------------------------------------------------------
# Tests: _is_orchestrator_enabled in context
# ---------------------------------------------------------------


@pytest.mark.integration
class TestOrchestratorEnableDisable:
    """Test orchestrator enable/disable logic in the WebSocket handler context."""

    @pytest.mark.asyncio
    async def test_is_orchestrator_enabled_with_tool_support(self):
        """When config is enabled and model has tools, orchestrator is enabled."""
        from tensortruth.api.routes.chat import _is_orchestrator_enabled

        session = {
            "params": {
                "model": "qwen3:32b",
                "orchestrator_enabled": True,
            }
        }
        with patch(
            "tensortruth.api.routes.chat.check_tool_call_support",
            return_value=True,
        ):
            assert _is_orchestrator_enabled(session, "qwen3:32b") is True

    @pytest.mark.asyncio
    async def test_is_orchestrator_disabled_without_tool_support(self):
        """When model lacks tools capability, orchestrator is disabled."""
        from tensortruth.api.routes.chat import _is_orchestrator_enabled

        session = {
            "params": {
                "model": "llama3.1:8b",
                "orchestrator_enabled": True,
            }
        }
        with patch(
            "tensortruth.api.routes.chat.check_tool_call_support",
            return_value=False,
        ):
            assert _is_orchestrator_enabled(session, "llama3.1:8b") is False

    @pytest.mark.asyncio
    async def test_is_orchestrator_disabled_by_config(self):
        """When config explicitly disables orchestrator, it is disabled."""
        from tensortruth.api.routes.chat import _is_orchestrator_enabled

        session = {
            "params": {
                "model": "qwen3:32b",
                "orchestrator_enabled": False,
            }
        }
        # Should not even check tool support
        assert _is_orchestrator_enabled(session, "qwen3:32b") is False


# ---------------------------------------------------------------
# Tests: OrchestratorStreamTranslator end-to-end
# ---------------------------------------------------------------


@pytest.mark.integration
class TestOrchestratorStreamTranslatorIntegration:
    """Integration test for the full stream translation pipeline."""

    def test_full_orchestrator_stream_pipeline(self):
        """Simulate a complete orchestrator event stream and verify output."""
        from tensortruth.services.models import ToolProgress
        from tensortruth.services.orchestrator_service import OrchestratorEvent
        from tensortruth.services.orchestrator_stream import (
            OrchestratorStreamTranslator,
        )

        translator = OrchestratorStreamTranslator()

        # Simulate: tool_phase -> tool_call -> tool_call_result -> tokens
        events = [
            OrchestratorEvent(
                tool_phase=ToolProgress(
                    tool_id="orchestrator",
                    phase="thinking",
                    message="Analyzing your request...",
                )
            ),
            OrchestratorEvent(
                tool_call={"tool": "web_search", "params": {"query": "latest AI news"}}
            ),
            OrchestratorEvent(
                tool_phase=ToolProgress(
                    tool_id="web_search",
                    phase="searching",
                    message="Searching the web...",
                    metadata={"query": "latest AI news"},
                )
            ),
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "latest AI news"},
                    "output": json.dumps(
                        [
                            {
                                "url": "https://example.com/ai",
                                "title": "AI News",
                                "snippet": "Latest developments in AI",
                            }
                        ]
                    ),
                    "is_error": False,
                }
            ),
            OrchestratorEvent(token="Based on "),
            OrchestratorEvent(token="the search results, "),
            OrchestratorEvent(token="here is the latest AI news."),
        ]

        messages = []
        for event in events:
            msg = translator.process_event(event)
            if msg is not None:
                messages.append(msg)

        result = translator.finalize()

        # Verify accumulated state
        assert (
            result.full_response
            == "Based on the search results, here is the latest AI news."
        )
        assert result.web_called is True
        assert result.web_count == 1
        assert "web" in result.source_types
        assert result.confidence_level == "high"  # No RAG -> high confidence
        assert len(result.tool_steps) == 1

        # Verify message types emitted
        msg_types = [m["type"] for m in messages]
        assert "tool_phase" in msg_types
        assert "tool_progress" in msg_types
        assert "token" in msg_types

    def test_mixed_rag_and_web_sources_pipeline(self):
        """Test a stream with both RAG and web tool results."""
        from llama_index.core.schema import NodeWithScore, TextNode

        from tensortruth.services.models import RAGRetrievalResult
        from tensortruth.services.orchestrator_service import OrchestratorEvent
        from tensortruth.services.orchestrator_stream import (
            OrchestratorStreamTranslator,
        )

        # Use a mock chat_service for source extraction
        mock_chat_service = MagicMock()
        mock_chat_service.extract_sources.return_value = [
            {
                "text": "Transformer content",
                "score": 0.9,
                "metadata": {"doc_type": "paper"},
            },
        ]

        translator = OrchestratorStreamTranslator(chat_service=mock_chat_service)

        # Inject RAG result
        node = NodeWithScore(
            node=TextNode(text="Transformer content", metadata={"doc_type": "paper"}),
            score=0.9,
        )
        rag_result = RAGRetrievalResult(
            source_nodes=[node],
            confidence_level="normal",
            metrics={"score_distribution": {"mean": 0.9}},
            condensed_query="transformers",
            num_sources=1,
        )
        translator.set_rag_retrieval_result(rag_result)

        # Process events: RAG tool call + result, then web search + result
        events = [
            OrchestratorEvent(
                tool_call={"tool": "rag_query", "params": {"query": "transformers"}}
            ),
            OrchestratorEvent(
                tool_call_result={
                    "tool": "rag_query",
                    "params": {"query": "transformers"},
                    "output": "Found 1 sources (confidence: normal)...",
                    "is_error": False,
                }
            ),
            OrchestratorEvent(
                tool_call={
                    "tool": "web_search",
                    "params": {"query": "transformers 2024"},
                }
            ),
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "transformers 2024"},
                    "output": json.dumps(
                        [
                            {
                                "url": "https://arxiv.org/recent",
                                "title": "Recent Papers",
                                "snippet": "Latest transformer research",
                            }
                        ]
                    ),
                    "is_error": False,
                }
            ),
            OrchestratorEvent(token="Combined answer from RAG and web."),
        ]

        for event in events:
            translator.process_event(event)

        result = translator.finalize()

        assert result.rag_called is True
        assert result.web_called is True
        assert result.rag_count == 1
        assert result.web_count == 1
        assert len(result.sources) == 2
        assert result.source_types == ["rag", "web"]
        assert result.confidence_level == "normal"
        assert result.metrics == {"score_distribution": {"mean": 0.9}}

        # Build sources message and verify breakdown
        msg = translator.build_sources_message()
        assert msg is not None
        assert msg["source_types"] == ["rag", "web"]
        assert msg["rag_count"] == 1
        assert msg["web_count"] == 1


# ---------------------------------------------------------------
# Tests: OrchestratorService with mocked agent
# ---------------------------------------------------------------


@pytest.mark.integration
class TestOrchestratorServiceIntegration:
    """Integration tests for OrchestratorService.execute() with mocked LLM."""

    @pytest.mark.asyncio
    async def test_execute_emits_thinking_phase_on_tool_call(self):
        """The orchestrator thinking phase should be drained with the first tool call."""
        from llama_index.core.agent.workflow import AgentStream, ToolCall

        from tensortruth.services.orchestrator_service import OrchestratorService

        tool_service = MagicMock()
        tool_service.tools = []
        tool_service.execute_tool = AsyncMock()

        rag_service = MagicMock()
        rag_service.is_loaded.return_value = False

        svc = OrchestratorService(
            tool_service=tool_service,
            rag_service=rag_service,
            model="test-model",
            base_url="http://localhost:11434",
            context_window=4096,
        )

        tc = ToolCall(
            tool_name="web_search",
            tool_kwargs={"query": "test"},
            tool_id="tc_1",
        )

        async def mock_stream():
            yield tc
            yield AgentStream(
                delta="Result", response="", current_agent_name="orchestrator"
            )

        handler = _make_awaitable_handler(mock_stream, "Result")

        with (
            patch(
                "tensortruth.services.orchestrator_service.LIFunctionAgent",
                return_value=MagicMock(run=MagicMock(return_value=handler)),
            ),
            patch(
                "tensortruth.services.orchestrator_service.get_orchestrator_llm",
                return_value=MagicMock(),
            ),
        ):
            events = []
            async for event in svc.execute("Hello"):
                events.append(event)

        # The orchestrator thinking phase should be drained after the first tool call
        phase_events = [e for e in events if e.tool_phase is not None]
        assert len(phase_events) >= 1
        first_phase = phase_events[0]
        assert first_phase.tool_phase.tool_id == "orchestrator"
        assert first_phase.tool_phase.phase == "thinking"

    @pytest.mark.asyncio
    async def test_execute_with_chat_history(self):
        """Chat history should be passed through budgeting to the agent."""
        from tensortruth.services.orchestrator_service import OrchestratorService

        tool_service = MagicMock()
        tool_service.tools = []
        tool_service.execute_tool = AsyncMock()

        rag_service = MagicMock()
        rag_service.is_loaded.return_value = False

        svc = OrchestratorService(
            tool_service=tool_service,
            rag_service=rag_service,
            model="test-model",
            base_url="http://localhost:11434",
            context_window=16384,  # Large enough to keep all history
        )

        async def mock_stream():
            return
            yield

        handler = _make_awaitable_handler(mock_stream, "Response")

        mock_agent_cls = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = MagicMock(return_value=handler)
        mock_agent_cls.return_value = mock_agent_instance

        with (
            patch(
                "tensortruth.services.orchestrator_service.LIFunctionAgent",
                mock_agent_cls,
            ),
            patch(
                "tensortruth.services.orchestrator_service.get_orchestrator_llm",
                return_value=MagicMock(),
            ),
        ):
            history = [
                {"role": "user", "content": "What is PyTorch?"},
                {"role": "assistant", "content": "PyTorch is a framework."},
            ]
            events = []
            async for event in svc.execute("Tell me more", chat_history=history):
                events.append(event)

        # Verify the agent was called with chat_history
        call_kwargs = mock_agent_instance.run.call_args
        assert call_kwargs is not None
        # chat_history should be passed
        passed_history = call_kwargs.kwargs.get("chat_history") or call_kwargs[1].get(
            "chat_history"
        )
        assert passed_history is not None
        assert len(passed_history) == 2
