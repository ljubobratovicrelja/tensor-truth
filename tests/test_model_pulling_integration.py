"""Integration tests for model pulling functionality."""

from unittest.mock import MagicMock, Mock, patch

import requests

from tensortruth.core.ollama import get_available_models, pull_model


class TestModelPullingIntegration:
    """Integration tests for model pulling with real config."""

    def test_pull_model_with_progress_callback(self):
        """Test pull_model function with progress callback."""
        # Mock the requests.post to simulate a model pull
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        # Simulate progress updates
        mock_response.iter_lines.return_value = [
            b'{"status":"pulling manifest"}',
            b'{"status":"pulling abc123","total":1000,"completed":250}',
            b'{"status":"pulling abc123","total":1000,"completed":750}',
            b'{"status":"verifying sha256 digest"}',
            b'{"status":"writing manifest"}',
            b'{"status":"success"}',
        ]

        # Track callback calls
        callback_results = []

        def progress_callback(status, progress, message):
            callback_results.append((status, progress, message))

        with patch("requests.post", return_value=mock_response):
            result = pull_model("test-model:1b", progress_callback)

        # Verify success
        assert result is True

        # Verify callback was called
        assert len(callback_results) > 0

        # Verify we got different status updates
        statuses = [status for status, _, _ in callback_results]
        assert "pulling manifest" in statuses
        assert "success" in statuses

        # Verify progress was tracked
        progresses = [progress for _, progress, _ in callback_results if progress > 0]
        assert len(progresses) > 0
        assert any(p > 0.2 for p in progresses)  # At least one progress > 20%

    def test_pull_model_network_failure(self):
        """Test pull_model handles network failures gracefully."""
        with patch(
            "requests.post",
            side_effect=requests.exceptions.ConnectionError("No connection"),
        ):
            result = pull_model("test-model:1b")
            assert result is False

    def test_get_available_models_when_ollama_unavailable(self):
        """Test get_available_models returns empty list when API fails."""
        # Mock API failure
        with patch("requests.get", side_effect=Exception("API unavailable")):
            result = get_available_models()

            # Should return empty list when Ollama is unavailable
            assert isinstance(result, list)
            assert result == []

    def test_get_available_models_success(self):
        """Test get_available_models with successful API response."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "deepseek-r1:14b"},
                {"name": "test-model:1b"},
            ]
        }

        with patch("requests.get", return_value=mock_response):
            result = get_available_models()

            # Should return sorted list of model names
            assert result == ["deepseek-r1:14b", "llama3.1:8b", "test-model:1b"]


class TestModelPullingScenarios:
    """Test different scenarios for model pulling logic."""

    def test_model_availability_checking(self):
        """Test the logic for determining which models need to be pulled."""
        # Test data
        required_models = ["deepseek-r1:14b", "deepseek-r1:8b", "llama3.1:8b"]

        # Scenario 1: All models available
        available_models = [
            "deepseek-r1:14b",
            "deepseek-r1:8b",
            "llama3.1:8b",
            "extra-model:1b",
        ]
        missing = [model for model in required_models if model not in available_models]
        assert len(missing) == 0

        # Scenario 2: Some models missing
        available_models = ["deepseek-r1:14b", "extra-model:1b"]  # Missing 2 models
        missing = [model for model in required_models if model not in available_models]
        assert len(missing) == 2
        assert "deepseek-r1:8b" in missing
        assert "llama3.1:8b" in missing

        # Scenario 3: All models missing
        available_models = ["extra-model:1b"]
        missing = [model for model in required_models if model not in available_models]
        assert len(missing) == 3
        assert all(model in missing for model in required_models)

    def test_duplicate_model_handling(self):
        """Test that duplicate models are handled correctly."""
        # Test removing duplicates while preserving order
        models = ["deepseek-r1:14b", "deepseek-r1:8b", "deepseek-r1:14b", "llama3.1:8b"]
        unique_models = list(dict.fromkeys(models))

        assert len(unique_models) == 3
        assert unique_models == ["deepseek-r1:14b", "deepseek-r1:8b", "llama3.1:8b"]


class TestModelPullingErrorHandling:
    """Test error handling in model pulling."""

    def test_pull_model_with_malformed_json(self):
        """Test handling of malformed JSON in pull response."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"status":"pulling manifest"}',
            b"invalid json",  # This should be handled gracefully
            b'{"status":"success"}',
        ]

        with patch("requests.post", return_value=mock_response):
            # Should not crash on malformed JSON
            result = pull_model("test-model:1b")
            assert result is True  # Should still complete successfully

    def test_pull_model_with_empty_response(self):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = []  # Empty response

        with patch("requests.post", return_value=mock_response):
            # Should handle empty response gracefully
            result = pull_model("test-model:1b")
            assert result is True  # Should complete without error
