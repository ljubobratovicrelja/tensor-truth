"""Test that messages API endpoint returns metrics field.

This test reproduces the user's bug: metrics are saved to chat_sessions.json
but not returned by the GET /sessions/{id}/messages endpoint.
"""

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from tensortruth.api.main import create_app


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


class TestMessagesEndpointMetrics:
    """Test that messages endpoint includes metrics in response."""

    @pytest.mark.asyncio
    async def test_get_messages_includes_metrics(self, client, tmp_path, monkeypatch):
        """Test that GET /sessions/{id}/messages returns metrics field.

        This reproduces the bug where:
        1. Metrics are saved to chat_sessions.json ✓
        2. But NOT returned by the messages API endpoint ✗
        3. So frontend shows undefined ✗
        """
        # Setup test session file with metrics
        sessions_file = tmp_path / "chat_sessions.json"
        session_data = {
            "current_id": "test-session-123",
            "sessions": {
                "test-session-123": {
                    "title": "Test Chat",
                    "created_at": "2026-01-21",
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is a neural network?",
                        },
                        {
                            "role": "assistant",
                            "content": "A neural network is...",
                            "sources": [
                                {
                                    "text": "Neural networks are...",
                                    "score": 0.85,
                                    "metadata": {"filename": "nn.pdf"},
                                }
                            ],
                            "metrics": {
                                "score_distribution": {
                                    "mean": 0.85,
                                    "median": 0.85,
                                    "min": 0.85,
                                    "max": 0.85,
                                    "std": None,
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
                            },
                        },
                    ],
                    "modules": ["test"],
                    "params": {},
                }
            },
        }
        sessions_file.write_text(json.dumps(session_data))

        # Mock the session service to use our test file
        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        # Call the messages endpoint
        response = await client.get("/api/sessions/test-session-123/messages")

        assert response.status_code == 200
        data = response.json()

        # Verify we got messages
        assert "messages" in data
        assert len(data["messages"]) == 2

        # Check the assistant message
        assistant_msg = data["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["sources"]) == 1

        # THE CRITICAL ASSERTION: metrics must be in the API response
        assert (
            "metrics" in assistant_msg
        ), "❌ BUG: metrics field missing from API response!"
        assert (
            assistant_msg["metrics"] is not None
        ), "❌ BUG: metrics is None in API response!"

        # Verify metrics structure
        metrics = assistant_msg["metrics"]
        assert "score_distribution" in metrics
        assert metrics["score_distribution"]["mean"] == 0.85
        assert "diversity" in metrics
        assert metrics["diversity"]["unique_sources"] == 1
        assert "coverage" in metrics
        assert metrics["coverage"]["total_chunks"] == 1
        assert "quality" in metrics
        assert metrics["quality"]["high_confidence_ratio"] == 1.0

    @pytest.mark.asyncio
    async def test_messages_without_metrics_returns_none(
        self, client, tmp_path, monkeypatch
    ):
        """Test that old messages without metrics return metrics=None."""
        sessions_file = tmp_path / "chat_sessions.json"
        session_data = {
            "current_id": "old-session",
            "sessions": {
                "old-session": {
                    "title": "Old Chat",
                    "created_at": "2026-01-20",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello",
                        },
                        {
                            "role": "assistant",
                            "content": "Hi there!",
                            "sources": [
                                {
                                    "text": "Greeting response",
                                    "score": 0.9,
                                    "metadata": {},
                                }
                            ],
                            # No metrics field (old message)
                        },
                    ],
                    "modules": [],
                    "params": {},
                }
            },
        }
        sessions_file.write_text(json.dumps(session_data))

        monkeypatch.setattr(
            "tensortruth.api.deps.get_sessions_file", lambda: sessions_file
        )
        from tensortruth.api.deps import get_session_service

        get_session_service.cache_clear()

        response = await client.get("/api/sessions/old-session/messages")
        assert response.status_code == 200

        data = response.json()
        assistant_msg = data["messages"][1]

        # Old messages should have metrics field but with None value
        assert "metrics" in assistant_msg
        assert assistant_msg["metrics"] is None
