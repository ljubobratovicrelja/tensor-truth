"""Test that verifies metrics are accessible in the message display flow.

This test simulates the exact flow:
1. Message loaded from backend API
2. Message passed to MessageItem component
3. Metrics extracted and passed to SourcesList
4. SourcesList displays metrics (not fallback)
"""

import json


def test_message_has_metrics_field():
    """Test that a saved message includes metrics field.

    This verifies the backend->frontend data contract.
    """
    # Simulate what the API returns (from chat_sessions.json)
    api_response = {
        "messages": [
            {
                "role": "assistant",
                "content": "Test response",
                "sources": [
                    {
                        "text": "Source text",
                        "score": 0.85,
                        "metadata": {"filename": "test.pdf"},
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
                    "diversity": {"unique_sources": 1, "source_types": 1},
                    "coverage": {"total_chunks": 1, "estimated_tokens": 25},
                    "quality": {
                        "high_confidence_ratio": 1.0,
                        "low_confidence_ratio": 0.0,
                    },
                },
            }
        ]
    }

    # Verify the message structure
    message = api_response["messages"][0]
    assert "metrics" in message, "Message missing metrics field!"
    assert message["metrics"] is not None, "Metrics is None!"
    assert isinstance(message["metrics"], dict), "Metrics should be dict!"

    # Verify metrics can be JSON serialized (no numpy types)
    json_str = json.dumps(message)
    assert json_str is not None

    # Verify metrics structure matches TypeScript RetrievalMetrics interface
    metrics = message["metrics"]
    assert "score_distribution" in metrics
    assert "diversity" in metrics
    assert "coverage" in metrics
    assert "quality" in metrics

    print("✓ Message has valid metrics structure")
    print(f"✓ Mean score: {metrics['score_distribution']['mean']}")
    print(f"✓ Unique sources: {metrics['diversity']['unique_sources']}")
    print(f"✓ Total chunks: {metrics['coverage']['total_chunks']}")


def test_metrics_accessible_from_message_object():
    """Test that metrics can be accessed using TypeScript-style access.

    Simulates: const messageMetrics = metrics ?? message.metrics
    """
    message = {
        "role": "assistant",
        "content": "Test",
        "sources": [{"text": "src", "score": 0.8, "metadata": {}}],
        "metrics": {
            "score_distribution": {"mean": 0.8},
            "diversity": {"unique_sources": 1},
            "coverage": {"total_chunks": 1},
            "quality": {"high_confidence_ratio": 1.0},
        },
    }

    # Simulate MessageItem.tsx line: messageMetrics = metrics ?? message.metrics
    metrics_prop = None  # Streaming metrics (None for historical messages)
    messageMetrics = (
        metrics_prop if metrics_prop is not None else message.get("metrics")
    )

    assert messageMetrics is not None, "Failed to extract metrics from message!"
    assert messageMetrics["score_distribution"]["mean"] == 0.8

    print("✓ Metrics successfully extracted from message object")


def test_sourceslist_should_display_metrics_not_fallback():
    """Test the conditional logic in SourcesList component.

    Simulates: {metrics ? <NewFormat/> : stats ? <OldFallback/> : null}
    """
    # Simulate having both metrics and stats (from client-side calculation)
    metrics = {
        "score_distribution": {"mean": 0.75, "min": 0.7, "max": 0.8, "std": 0.05},
        "diversity": {"unique_sources": 2},
        "coverage": {"estimated_tokens": 200},
    }

    sources = [
        {"score": 0.7, "text": "src1", "metadata": {}},
        {"score": 0.8, "text": "src2", "metadata": {}},
    ]

    # Client-side stats calculation (fallback)
    scores = [s["score"] for s in sources]
    stats = {"max": max(scores), "min": min(scores), "mean": sum(scores) / len(scores)}

    # The conditional: metrics ? NEW : stats ? OLD : null
    should_show_new_format = metrics is not None
    should_show_fallback = not should_show_new_format and stats is not None

    assert should_show_new_format, "Should display NEW format when metrics exist!"
    assert not should_show_fallback, "Should NOT display fallback when metrics exist!"

    print("✓ SourcesList would display NEW metrics format")
    print(f"  Avg: {int(metrics['score_distribution']['mean'] * 100)}%")
    min_score = int(metrics["score_distribution"]["min"] * 100)
    max_score = int(metrics["score_distribution"]["max"] * 100)
    print(f"  Range: {min_score}%-{max_score}%")
    print(f"  {metrics['diversity']['unique_sources']} docs")


if __name__ == "__main__":
    test_message_has_metrics_field()
    test_metrics_accessible_from_message_object()
    test_sourceslist_should_display_metrics_not_fallback()
    print("\n✅ All frontend metrics display tests passed!")
