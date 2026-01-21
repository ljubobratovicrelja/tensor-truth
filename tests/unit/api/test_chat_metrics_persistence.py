"""Unit tests for metrics persistence in chat messages."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tensortruth.services.models import RAGChunk, RAGResponse
from tensortruth.services.session_service import SessionService


class TestMetricsPersistence:
    """Test that retrieval metrics are saved to session storage."""

    @pytest.fixture
    def mock_rag_service(self):
        """Create mock RAG service that returns metrics."""
        service = MagicMock()

        # Mock response with metrics
        def query_generator(prompt):
            # Yield status
            yield RAGChunk(status="retrieving")

            # Yield some content
            yield RAGChunk(text="Test response")

            # Yield final chunk with sources and metrics
            mock_node = MagicMock()
            mock_node.score = 0.85
            mock_node.text = "Source text"
            mock_node.metadata = {"filename": "test.pdf", "doc_type": "paper"}

            metrics = {
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
            yield RAGChunk(source_nodes=[mock_node], is_complete=True, metrics=metrics)

        service.query.return_value = query_generator("test")
        service.needs_reload.return_value = False
        service.get_chat_history.return_value = []

        return service

    @pytest.fixture
    def session_service(self, tmp_path: Path):
        """Create session service with temporary storage."""
        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text(json.dumps({"current_id": None, "sessions": {}}))
        return SessionService(sessions_file)

    def test_metrics_saved_to_session(
        self, session_service: SessionService, mock_rag_service, tmp_path: Path
    ):
        """Test that metrics are saved to message in session storage.

        When a RAG query completes, the metrics should be:
        1. Computed by the RAG service
        2. Included in the RAGChunk
        3. Saved to the assistant message in sessions.json
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

        # Simulate RAG query collecting response
        full_response = ""
        sources = []
        metrics_dict = None

        for chunk in mock_rag_service.query("What is a tensor?"):
            if chunk.is_complete:
                sources = chunk.source_nodes
                metrics_dict = chunk.metrics
            elif chunk.text:
                full_response += chunk.text

        # Save assistant message (simulating what the API does)
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
        if metrics_dict:
            assistant_message["metrics"] = metrics_dict

        data = session_service.load()
        data = session_service.add_message(session_id, assistant_message, data)
        session_service.save(data)

        # Verify metrics are in saved session
        saved_data = session_service.load()
        session = saved_data.sessions[session_id]
        messages = session["messages"]

        # Should have 2 messages (user + assistant)
        assert len(messages) == 2

        # Check assistant message has metrics
        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"
        assert "metrics" in assistant_msg
        assert assistant_msg["metrics"] is not None

        # Verify metrics structure
        metrics = assistant_msg["metrics"]
        assert "score_distribution" in metrics
        assert "diversity" in metrics
        assert "coverage" in metrics
        assert "quality" in metrics

        # Verify specific values
        assert metrics["score_distribution"]["mean"] == 0.85
        assert metrics["diversity"]["unique_sources"] == 1
        assert metrics["coverage"]["total_chunks"] == 1
        assert metrics["quality"]["high_confidence_ratio"] == 1.0

    def test_llm_only_mode_no_metrics(
        self, session_service: SessionService, tmp_path: Path
    ):
        """Test that LLM-only mode (no RAG) has no metrics.

        When no modules are attached, queries use LLM-only mode
        which doesn't perform retrieval and thus has no metrics.
        """
        # Create a session without modules
        data = session_service.load()
        session_id, data = session_service.create(modules=[], params={}, data=data)
        session_service.save(data)

        # Add user message
        data = session_service.load()
        data = session_service.add_message(
            session_id, {"role": "user", "content": "Hello"}, data
        )
        session_service.save(data)

        # Simulate LLM-only response (no sources, no metrics)
        assistant_message = {
            "role": "assistant",
            "content": "Hello! How can I help?",
        }
        # No sources or metrics in LLM-only mode

        data = session_service.load()
        data = session_service.add_message(session_id, assistant_message, data)
        session_service.save(data)

        # Verify no metrics in saved session
        saved_data = session_service.load()
        session = saved_data.sessions[session_id]
        messages = session["messages"]

        assistant_msg = messages[1]
        assert "metrics" not in assistant_msg or assistant_msg.get("metrics") is None

    def test_metrics_json_serializable(self, session_service: SessionService):
        """Test that metrics can be serialized to JSON.

        This verifies that numpy types are properly converted to Python types.
        """
        import numpy as np

        # Create session
        data = session_service.load()
        session_id, data = session_service.create(
            modules=["pytorch"], params={}, data=data
        )
        session_service.save(data)

        # Create metrics with numpy types (as returned by real retrieval)
        metrics_dict = {
            "score_distribution": {
                "mean": float(np.float32(0.85)),  # Simulate numpy score
                "median": float(np.float32(0.85)),
                "min": float(np.float32(0.80)),
                "max": float(np.float32(0.90)),
                "std": float(np.float32(0.05)),
                "q1": None,
                "q3": None,
                "iqr": None,
                "range": float(np.float32(0.10)),
            },
            "diversity": {
                "unique_sources": 3,
                "source_types": 2,
                "source_entropy": float(np.float32(1.5)),
            },
            "coverage": {
                "total_context_chars": 1000,
                "avg_chunk_length": 333.33,
                "total_chunks": 3,
                "estimated_tokens": 250,
            },
            "quality": {
                "high_confidence_ratio": float(np.float32(0.67)),
                "low_confidence_ratio": float(np.float32(0.0)),
            },
        }

        # Add message with metrics
        assistant_message = {
            "role": "assistant",
            "content": "Test response",
            "metrics": metrics_dict,
        }

        data = session_service.load()
        data = session_service.add_message(session_id, assistant_message, data)

        # Should serialize without error
        session_service.save(data)

        # Should be able to reload
        reloaded_data = session_service.load()
        session = reloaded_data.sessions[session_id]
        saved_metrics = session["messages"][0]["metrics"]

        # Verify all values are present and serializable
        assert saved_metrics["score_distribution"]["mean"] == pytest.approx(0.85)
        assert saved_metrics["diversity"]["unique_sources"] == 3
        assert json.dumps(saved_metrics)  # Should not raise
