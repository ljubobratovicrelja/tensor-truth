"""Tests for orchestrator event translation and source accumulation.

Verifies that OrchestratorEvent objects are correctly translated into
WebSocket message dicts, and that sources from RAG and web tools are
properly accumulated and batched.
"""

import json
from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, TextNode

from tensortruth.services.models import RAGRetrievalResult, ToolProgress
from tensortruth.services.orchestrator_service import OrchestratorEvent
from tensortruth.services.orchestrator_stream import (
    OrchestratorStreamTranslator,
    translate_event,
)

# ---------------------------------------------------------------
# translate_event() unit tests
# ---------------------------------------------------------------


class TestTranslateEvent:
    """Test the stateless translate_event function."""

    def test_token_event(self):
        event = OrchestratorEvent(token="Hello")
        msg = translate_event(event)
        assert msg == {"type": "token", "content": "Hello"}

    def test_tool_call_event(self):
        event = OrchestratorEvent(
            tool_call={"tool": "web_search", "params": {"query": "test"}}
        )
        msg = translate_event(event)
        assert msg == {
            "type": "tool_progress",
            "tool": "web_search",
            "action": "calling",
            "params": {"query": "test"},
        }

    def test_tool_call_result_success(self):
        event = OrchestratorEvent(
            tool_call_result={
                "tool": "web_search",
                "params": {"query": "test"},
                "output": "some results",
                "is_error": False,
            }
        )
        msg = translate_event(event)
        assert msg["type"] == "tool_progress"
        assert msg["action"] == "completed"
        assert msg["output"] == "some results"
        assert msg["is_error"] is False

    def test_tool_call_result_failure(self):
        event = OrchestratorEvent(
            tool_call_result={
                "tool": "fetch_page",
                "params": {"url": "http://example.com"},
                "output": "Error: timeout",
                "is_error": True,
            }
        )
        msg = translate_event(event)
        assert msg["action"] == "failed"
        assert msg["is_error"] is True

    def test_tool_phase_event(self):
        tp = ToolProgress(
            tool_id="rag",
            phase="retrieving",
            message="Searching knowledge base...",
            metadata={"query": "test"},
        )
        event = OrchestratorEvent(tool_phase=tp)
        msg = translate_event(event)
        assert msg == {
            "type": "tool_phase",
            "tool_id": "rag",
            "phase": "retrieving",
            "message": "Searching knowledge base...",
            "metadata": {"query": "test"},
        }

    def test_empty_event_returns_none(self):
        event = OrchestratorEvent()
        msg = translate_event(event)
        assert msg is None


# ---------------------------------------------------------------
# OrchestratorStreamTranslator tests
# ---------------------------------------------------------------


class TestStreamTranslator:
    """Test the stateful OrchestratorStreamTranslator."""

    def test_token_accumulation(self):
        translator = OrchestratorStreamTranslator()

        events = [
            OrchestratorEvent(token="Hello"),
            OrchestratorEvent(token=" world"),
        ]
        for e in events:
            translator.process_event(e)

        result = translator.finalize()
        assert result.full_response == "Hello world"

    def test_tool_steps_accumulated(self):
        translator = OrchestratorStreamTranslator()

        event = OrchestratorEvent(
            tool_call_result={
                "tool": "web_search",
                "params": {"query": "test"},
                "output": "results",
                "is_error": False,
            }
        )
        translator.process_event(event)

        result = translator.finalize()
        assert len(result.tool_steps) == 1
        assert result.tool_steps[0]["tool"] == "web_search"

    def test_web_sources_extracted(self):
        translator = OrchestratorStreamTranslator()

        web_results = json.dumps(
            [
                {
                    "url": "http://example.com/1",
                    "title": "Page 1",
                    "snippet": "Snippet 1",
                },
                {
                    "url": "http://example.com/2",
                    "title": "Page 2",
                    "snippet": "Snippet 2",
                },
            ]
        )

        # First the tool call
        translator.process_event(
            OrchestratorEvent(
                tool_call={"tool": "web_search", "params": {"query": "test"}}
            )
        )
        # Then the result
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "test"},
                    "output": web_results,
                    "is_error": False,
                }
            )
        )

        result = translator.finalize()
        assert result.web_called is True
        assert result.web_count == 2
        assert "web" in result.source_types
        assert len(result.sources) == 2
        assert result.sources[0]["metadata"]["source_url"] == "http://example.com/1"
        assert result.sources[0]["metadata"]["doc_type"] == "web"
        assert result.sources[0]["metadata"]["fetch_status"] == "found"

    def test_rag_sources_via_injected_result(self):
        """Test RAG source extraction using an injected RAGRetrievalResult."""
        mock_chat_service = MagicMock()
        mock_chat_service.extract_sources.return_value = [
            {
                "text": "Source content",
                "score": 0.95,
                "metadata": {"doc_type": "paper"},
            },
        ]

        translator = OrchestratorStreamTranslator(chat_service=mock_chat_service)

        # Inject a RAG result with real nodes
        node = NodeWithScore(
            node=TextNode(text="Source content", metadata={"doc_type": "paper"}),
            score=0.95,
        )
        rag_result = RAGRetrievalResult(
            source_nodes=[node],
            confidence_level="normal",
            metrics={"score_distribution": {"mean": 0.9}},
            condensed_query="test query",
            num_sources=1,
        )
        translator.set_rag_retrieval_result(rag_result)

        # Process rag_query tool call + result
        translator.process_event(
            OrchestratorEvent(
                tool_call={"tool": "rag_query", "params": {"query": "test"}}
            )
        )
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "rag_query",
                    "params": {"query": "test"},
                    "output": "Found 1 sources (confidence: normal)...",
                    "is_error": False,
                }
            )
        )

        result = translator.finalize()
        assert result.rag_called is True
        assert result.rag_count == 1
        assert "rag" in result.source_types
        assert result.metrics == {"score_distribution": {"mean": 0.9}}
        assert result.confidence_level == "normal"
        mock_chat_service.extract_sources.assert_called_once_with([node])

    def test_rag_sources_via_source_converter_fallback(self):
        """Test RAG source extraction via SourceConverter when no ChatService."""
        translator = OrchestratorStreamTranslator(chat_service=None)

        node = NodeWithScore(
            node=TextNode(
                text="Test content",
                metadata={"display_name": "TestDoc", "doc_type": "paper"},
            ),
            score=0.85,
        )
        rag_result = RAGRetrievalResult(
            source_nodes=[node],
            confidence_level="low",
            metrics={"quality": {"high_confidence_ratio": 0.5}},
            condensed_query="test",
            num_sources=1,
        )
        translator.set_rag_retrieval_result(rag_result)

        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "rag_query",
                    "params": {"query": "test"},
                    "output": "Found 1 sources (confidence: low)...",
                    "is_error": False,
                }
            )
        )

        result = translator.finalize()
        assert result.rag_count == 1
        assert result.confidence_level == "low"
        # Verify source was converted via SourceConverter
        assert result.sources[0]["metadata"]["display_name"] == "TestDoc"

    def test_confidence_high_when_no_rag(self):
        """Confidence defaults to 'high' when only web tools called."""
        translator = OrchestratorStreamTranslator()

        translator.process_event(
            OrchestratorEvent(
                tool_call={"tool": "web_search", "params": {"query": "test"}}
            )
        )
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "test"},
                    "output": "[]",
                    "is_error": False,
                }
            )
        )

        result = translator.finalize()
        assert result.confidence_level == "high"

    def test_confidence_high_when_no_tools(self):
        """Confidence defaults to 'high' when no tools called (direct response)."""
        translator = OrchestratorStreamTranslator()
        translator.process_event(OrchestratorEvent(token="Just chatting"))

        result = translator.finalize()
        assert result.confidence_level == "high"
        assert result.rag_called is False
        assert result.web_called is False

    def test_mixed_sources(self):
        """Test accumulation of both RAG and web sources."""
        mock_chat_service = MagicMock()
        mock_chat_service.extract_sources.return_value = [
            {"text": "RAG source", "score": 0.9, "metadata": {"doc_type": "paper"}},
        ]

        translator = OrchestratorStreamTranslator(chat_service=mock_chat_service)

        # Inject RAG result
        node = NodeWithScore(
            node=TextNode(text="RAG source", metadata={"doc_type": "paper"}),
            score=0.9,
        )
        rag_result = RAGRetrievalResult(
            source_nodes=[node],
            confidence_level="normal",
            metrics={"test": True},
            condensed_query="q",
            num_sources=1,
        )
        translator.set_rag_retrieval_result(rag_result)

        # Process RAG tool result
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "rag_query",
                    "params": {"query": "q"},
                    "output": "...",
                    "is_error": False,
                }
            )
        )

        # Process web search result
        web_json = json.dumps(
            [
                {"url": "http://example.com", "title": "Web", "snippet": "Web snippet"},
            ]
        )
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "q"},
                    "output": web_json,
                    "is_error": False,
                }
            )
        )

        result = translator.finalize()
        assert result.rag_count == 1
        assert result.web_count == 1
        assert len(result.sources) == 2
        assert result.source_types == ["rag", "web"]
        assert result.confidence_level == "normal"

    def test_build_sources_message(self):
        """Test building the batched sources WebSocket message."""
        translator = OrchestratorStreamTranslator()

        web_json = json.dumps(
            [
                {"url": "http://example.com", "title": "Test", "snippet": "snippet"},
            ]
        )
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "q"},
                    "output": web_json,
                    "is_error": False,
                }
            )
        )

        msg = translator.build_sources_message()
        assert msg is not None
        assert msg["type"] == "sources"
        assert len(msg["data"]) == 1
        assert msg["metrics"] is None  # No RAG metrics for web-only

    def test_build_sources_message_empty(self):
        """No sources message when no sources collected."""
        translator = OrchestratorStreamTranslator()
        translator.process_event(OrchestratorEvent(token="Direct answer"))

        msg = translator.build_sources_message()
        assert msg is None

    def test_build_done_message(self):
        """Test building the done WebSocket message."""
        translator = OrchestratorStreamTranslator()
        translator.process_event(OrchestratorEvent(token="Final answer"))

        msg = translator.build_done_message(title_pending=True)
        assert msg["type"] == "done"
        assert msg["content"] == "Final answer"
        assert msg["confidence_level"] == "high"
        assert msg["title_pending"] is True

    def test_build_done_message_no_title(self):
        translator = OrchestratorStreamTranslator()
        translator.process_event(OrchestratorEvent(token="Answer"))

        msg = translator.build_done_message(title_pending=False)
        assert "title_pending" not in msg

    def test_rag_text_fallback_confidence_extraction(self):
        """Test confidence extraction from text when no RAGRetrievalResult."""
        translator = OrchestratorStreamTranslator()

        output = (
            "Found 3 sources (confidence: low)\n"
            "Query used: test query\n\n"
            "--- Source 1: Doc (score: 0.8) ---\ncontent\n\n"
            "Metrics: avg_score=0.7500, max_score=0.9000"
        )

        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "rag_query",
                    "params": {"query": "test"},
                    "output": output,
                    "is_error": False,
                }
            )
        )

        result = translator.finalize()
        assert result.rag_called is True
        assert result.confidence_level == "low"
        # Text-based extraction captures metrics summary
        assert result.metrics is not None
        assert "parsed_summary" in result.metrics

    def test_web_search_with_invalid_json(self):
        """Gracefully handle non-JSON web_search output."""
        translator = OrchestratorStreamTranslator()

        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "test"},
                    "output": "Error: search failed",
                    "is_error": True,
                }
            )
        )

        result = translator.finalize()
        assert result.web_called is True
        assert result.web_count == 0  # No sources extracted from error output

    def test_full_output_stripped_from_tool_steps(self):
        """full_output should be stripped from tool_steps to keep storage lean."""
        translator = OrchestratorStreamTranslator()

        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "test"},
                    "output": "truncated",
                    "full_output": "full output text that is much longer",
                    "is_error": False,
                }
            )
        )

        result = translator.finalize()
        assert len(result.tool_steps) == 1
        assert "full_output" not in result.tool_steps[0]
        assert result.tool_steps[0]["output"] == "truncated"

    def test_extract_web_sources_prefers_full_output(self):
        """_extract_web_sources should use full_output when available."""
        translator = OrchestratorStreamTranslator()

        full_json = json.dumps(
            [
                {
                    "url": "http://example.com/full",
                    "title": "Full",
                    "snippet": "Full snippet",
                },
            ]
        )
        truncated = full_json[:10]  # Broken JSON

        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "test"},
                    "output": truncated,
                    "full_output": full_json,
                    "is_error": False,
                }
            )
        )

        result = translator.finalize()
        assert result.web_count == 1
        assert result.sources[0]["metadata"]["source_url"] == "http://example.com/full"
        assert result.sources[0]["metadata"]["fetch_status"] == "found"

    def test_set_web_source_nodes(self):
        """set_web_source_nodes should convert SourceNodes to API-compatible dicts."""
        from tensortruth.core.source import SourceNode, SourceStatus, SourceType

        translator = OrchestratorStreamTranslator()

        node = SourceNode(
            id="s1",
            title="Example Page",
            source_type=SourceType.WEB,
            url="https://example.com",
            content="Full content",
            snippet="A snippet",
            score=0.87,
            status=SourceStatus.SUCCESS,
            content_chars=500,
        )
        translator.set_web_source_nodes([node])

        result = translator.finalize()
        assert result.web_called is True
        assert result.web_count == 1
        assert result.sources[0]["score"] == 0.87
        assert result.sources[0]["metadata"]["source_url"] == "https://example.com"
        assert result.sources[0]["metadata"]["display_name"] == "Example Page"
        assert result.sources[0]["metadata"]["relevance_score"] == 0.87

    def test_set_web_source_nodes_replaces_json_parsed_sources(self):
        """set_web_source_nodes should replace any JSON-parsed web sources."""
        from tensortruth.core.source import SourceNode, SourceStatus, SourceType

        translator = OrchestratorStreamTranslator()

        # First, process a web_search result (creates JSON-parsed sources)
        web_json = json.dumps(
            [{"url": "http://old.com", "title": "Old", "snippet": "old"}]
        )
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "test"},
                    "output": web_json,
                    "is_error": False,
                }
            )
        )

        # Then inject proper SourceNodes (should replace the old ones)
        node = SourceNode(
            id="s1",
            title="New Page",
            source_type=SourceType.WEB,
            url="https://new.com",
            score=0.9,
            status=SourceStatus.SUCCESS,
        )
        translator.set_web_source_nodes([node])

        result = translator.finalize()
        assert result.web_count == 1
        assert result.sources[0]["metadata"]["source_url"] == "https://new.com"

    def test_sources_message_with_mixed_types_includes_breakdown(self):
        """Mixed source types include type breakdown in sources message."""
        mock_chat_service = MagicMock()
        mock_chat_service.extract_sources.return_value = [
            {"text": "RAG", "score": 0.9, "metadata": {"doc_type": "paper"}},
        ]

        translator = OrchestratorStreamTranslator(chat_service=mock_chat_service)

        # Inject RAG result
        node = NodeWithScore(
            node=TextNode(text="RAG", metadata={"doc_type": "paper"}), score=0.9
        )
        rag_result = RAGRetrievalResult(
            source_nodes=[node],
            confidence_level="normal",
            num_sources=1,
        )
        translator.set_rag_retrieval_result(rag_result)

        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "rag_query",
                    "params": {"query": "q"},
                    "output": "...",
                    "is_error": False,
                }
            )
        )

        web_json = json.dumps(
            [
                {"url": "http://ex.com", "title": "Web", "snippet": "s"},
            ]
        )
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "q"},
                    "output": web_json,
                    "is_error": False,
                }
            )
        )

        msg = translator.build_sources_message()
        assert msg is not None
        assert msg["source_types"] == ["rag", "web"]
        assert msg["rag_count"] == 1
        assert msg["web_count"] == 1

    def test_promote_web_source_on_fetch_page(self):
        """fetch_page result promotes a web source from 'found' to 'success'."""
        translator = OrchestratorStreamTranslator()

        # First, web_search creates sources with "found" status
        web_json = json.dumps(
            [
                {"url": "http://example.com/1", "title": "Page 1", "snippet": "s1"},
                {"url": "http://example.com/2", "title": "Page 2", "snippet": "s2"},
            ]
        )
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "test"},
                    "output": web_json,
                    "is_error": False,
                }
            )
        )

        # Verify both start as "found"
        assert translator._web_sources[0]["metadata"]["fetch_status"] == "found"
        assert translator._web_sources[1]["metadata"]["fetch_status"] == "found"

        # Then fetch_page completes for one URL
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "fetch_page",
                    "params": {"url": "http://example.com/1"},
                    "output": "Full page content here",
                    "is_error": False,
                }
            )
        )

        result = translator.finalize()
        # First source promoted to "success" with updated content_chars
        assert result.sources[0]["metadata"]["fetch_status"] == "success"
        assert result.sources[0]["metadata"]["content_chars"] == len(
            "Full page content here"
        )
        # Second source remains "found"
        assert result.sources[1]["metadata"]["fetch_status"] == "found"

    def test_promote_web_source_on_fetch_page_error(self):
        """fetch_page error marks a web source as 'failed'."""
        translator = OrchestratorStreamTranslator()

        web_json = json.dumps(
            [
                {"url": "http://example.com/1", "title": "Page 1", "snippet": "s1"},
            ]
        )
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "web_search",
                    "params": {"query": "test"},
                    "output": web_json,
                    "is_error": False,
                }
            )
        )

        # fetch_page fails
        translator.process_event(
            OrchestratorEvent(
                tool_call_result={
                    "tool": "fetch_page",
                    "params": {"url": "http://example.com/1"},
                    "output": "Error: timeout",
                    "is_error": True,
                }
            )
        )

        result = translator.finalize()
        assert result.sources[0]["metadata"]["fetch_status"] == "failed"
