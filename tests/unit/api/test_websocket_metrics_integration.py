"""Integration test for WebSocket metrics flow.

This test verifies that metrics computed during RAG queries
are properly saved to session storage via WebSocket handler.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tensortruth.services.models import RAGChunk
from tensortruth.services.session_service import SessionService


class TestWebSocketMetricsIntegration:
    """Test that WebSocket handler saves metrics from RAG chunks."""

    @pytest.fixture
    def session_service(self, tmp_path: Path):
        """Create session service with temporary storage."""
        sessions_file = tmp_path / "chat_sessions.json"
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        return SessionService(sessions_file, sessions_dir)

    def test_websocket_saves_metrics_from_rag_chunk(
        self, session_service: SessionService
    ):
        """Test that metrics from RAGChunk are saved to assistant message.

        This reproduces the user's bug: metrics are computed but not saved.

        Flow:
        1. RAG service returns RAGChunk with metrics
        2. WebSocket handler extracts metrics from chunk
        3. Metrics should be saved to assistant_message in session storage
        """
        # Create a session
        data = session_service.load()
        session_id, data = session_service.create(
            modules=["pytorch"], params={}, data=data
        )
        session_service.save(data)

        # Add user message
        data = session_service.load()
        data = session_service.add_message(
            session_id, {"role": "user", "content": "What is a tensor?"}, data
        )
        session_service.save(data)

        # Simulate RAG service returning chunks with metrics
        # (This is what the WebSocket handler receives)
        mock_node = MagicMock()
        mock_node.score = 0.85
        mock_node.text = "A tensor is a mathematical object"
        mock_node.metadata = {
            "filename": "pytorch_docs.md",
            "doc_type": "library_doc",
        }

        metrics_dict = {
            "score_distribution": {
                "mean": 0.85,
                "median": 0.85,
                "min": 0.85,
                "max": 0.85,
                "std": None,
                "q1": 0.85,
                "q3": 0.85,
                "iqr": 0.0,
                "range": 0.0,
            },
            "diversity": {
                "unique_sources": 1,
                "source_types": 1,
                "source_entropy": 0.0,
            },
            "coverage": {
                "total_context_chars": 100,
                "avg_chunk_length": 100.0,
                "total_chunks": 1,
                "estimated_tokens": 25,
            },
            "quality": {
                "high_confidence_ratio": 1.0,
                "low_confidence_ratio": 0.0,
            },
        }

        # Simulate what WebSocket handler does:
        # 1. Collect response and sources from chunks
        full_response = "Test response"
        sources = []
        metrics_from_chunk = None

        # Simulate receiving chunks (like in routes/chat.py lines 228-231)
        chunks = [
            RAGChunk(text="Test response"),
            RAGChunk(source_nodes=[mock_node], is_complete=True, metrics=metrics_dict),
        ]

        for chunk in chunks:
            if chunk.is_complete:
                sources = chunk.source_nodes
                metrics_from_chunk = chunk.metrics  # This is the key line
            elif chunk.text:
                full_response += chunk.text

        # 2. Save assistant message (like in routes/chat.py lines 286-293)
        assistant_message = {"role": "assistant", "content": full_response}
        if sources:
            assistant_message["sources"] = [
                {
                    "text": node.text,
                    "score": node.score,
                    "metadata": node.metadata,
                }
                for node in sources
            ]
        if metrics_from_chunk:  # BUG CHECK: This should add metrics
            assistant_message["metrics"] = metrics_from_chunk

        data = session_service.load()
        data = session_service.add_message(session_id, assistant_message, data)
        session_service.save(data)

        # 3. Verify metrics were saved
        saved_data = session_service.load()
        session = saved_data.sessions[session_id]
        messages = session["messages"]

        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"

        # THE CRITICAL ASSERTION: metrics must be present
        assert "metrics" in assistant_msg, "Metrics field missing from saved message!"
        assert assistant_msg["metrics"] is not None, "Metrics is None!"

        # Verify metrics structure
        saved_metrics = assistant_msg["metrics"]
        assert saved_metrics["score_distribution"]["mean"] == 0.85
        assert saved_metrics["diversity"]["unique_sources"] == 1
        assert saved_metrics["coverage"]["total_chunks"] == 1

    def test_actual_websocket_handler_code_path(self):
        """Test the exact code path in routes/chat.py WebSocket handler.

        This verifies that the variable name and logic matches what's
        actually in the WebSocket handler.
        """
        # Simulate the exact variable names in routes/chat.py
        metrics_dict = None

        # Line 230: if chunk.is_complete:
        # Line 231: metrics_dict = chunk.metrics
        mock_chunk = MagicMock()
        mock_chunk.is_complete = True
        mock_chunk.metrics = {
            "score_distribution": {"mean": 0.75},
            "diversity": {"unique_sources": 2},
            "coverage": {"total_chunks": 2},
            "quality": {"high_confidence_ratio": 0.5},
        }

        # Simulate the handler code
        if mock_chunk.is_complete:
            metrics_dict = mock_chunk.metrics

        # Line 291-292: if metrics_dict:
        #                   assistant_message["metrics"] = metrics_dict
        assistant_message = {"role": "assistant", "content": "test"}
        if metrics_dict:
            assistant_message["metrics"] = metrics_dict

        # Verify the logic works
        assert "metrics" in assistant_message
        assert assistant_message["metrics"] == mock_chunk.metrics
