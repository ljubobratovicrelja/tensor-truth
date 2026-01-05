"""Integration tests for browse agent.

These tests require:
- Ollama running locally
- Network connectivity
- Real LLM model (e.g., llama3.2:3b)
"""

import pytest

from tensortruth.utils.browse_agent import browse_agent


@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.requires_ollama
class TestBrowseAgentIntegration:
    """Integration tests with real Ollama and network."""

    def test_simple_factual_goal(self):
        """Test agent with simple factual goal."""
        result = browse_agent(
            goal="What is the capital of France?",
            model_name="llama3.2:3b",
            ollama_url="http://localhost:11434",
            max_iterations=3,  # Should conclude quickly
            context_window=16384,
        )

        # Check result structure
        assert result.final_answer is not None
        assert len(result.final_answer) > 0

        # Check Paris is mentioned (case insensitive)
        assert "paris" in result.final_answer.lower()

        # Check at least one search was performed
        assert len(result.searches_performed) > 0

        # Check termination (no more goal_satisfied since CONCLUDE was removed)
        assert result.termination_reason in ["max_iterations", "timeout"]

    def test_multi_step_research_goal(self):
        """Test agent with goal requiring multiple iterations."""
        result = browse_agent(
            goal="Compare Python and JavaScript for web development",
            model_name="llama3.2:3b",
            ollama_url="http://localhost:11434",
            max_iterations=5,
            context_window=16384,
        )

        # Check result structure
        assert result.final_answer is not None

        # Check both languages are mentioned
        assert "python" in result.final_answer.lower()
        assert "javascript" in result.final_answer.lower()

        # Check multiple searches were likely performed
        assert len(result.searches_performed) >= 1

        # Check some pages might have been visited
        # (not guaranteed, but likely for comparison task)
        assert result.current_iteration > 0

    def test_iteration_budget_enforcement(self):
        """Test that agent respects iteration limit."""
        result = browse_agent(
            goal="Comprehensive analysis of artificial intelligence history and future",
            model_name="llama3.2:3b",
            ollama_url="http://localhost:11434",
            max_iterations=2,  # Very low limit
            context_window=16384,
        )

        # Check iteration budget was respected
        assert result.current_iteration <= 2

        # Check termination reason (no more goal_satisfied since CONCLUDE was removed)
        assert result.termination_reason in [
            "max_iterations",
            "timeout",
        ]

        # Check still provided some answer
        assert result.final_answer is not None

    def test_technical_documentation_search(self):
        """Test searching for technical documentation."""
        result = browse_agent(
            goal="Find Python asyncio documentation and explain coroutines",
            model_name="llama3.2:3b",
            ollama_url="http://localhost:11434",
            max_iterations=5,
            context_window=16384,
        )

        # Check result mentions key concepts
        answer_lower = result.final_answer.lower()
        assert "asyncio" in answer_lower or "coroutine" in answer_lower

        # Check searches were performed
        assert len(result.searches_performed) > 0

    def test_agent_with_callbacks(self):
        """Test agent execution with progress and thinking callbacks."""
        thinking_updates = []
        progress_updates = []

        def thinking_callback(text):
            thinking_updates.append(text)

        def progress_callback(text):
            progress_updates.append(text)

        result = browse_agent(
            goal="What is Node.js?",
            model_name="llama3.2:3b",
            ollama_url="http://localhost:11434",
            max_iterations=3,
            thinking_callback=thinking_callback,
            progress_callback=progress_callback,
            context_window=16384,
        )

        # Check callbacks were invoked
        assert len(thinking_updates) > 0
        assert len(progress_updates) > 0

        # Check thinking updates have expected format
        assert any("Iteration" in update for update in thinking_updates)

        # Check progress updates indicate actions
        progress_text = " ".join(progress_updates).lower()
        assert "searching" in progress_text or "fetching" in progress_text

        # Check result is valid
        assert result.final_answer is not None


@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.slow
class TestBrowseAgentEdgeCases:
    """Integration tests for edge cases and error handling."""

    def test_very_specific_query(self):
        """Test with very specific technical query."""
        result = browse_agent(
            goal="What are the parameters of Python's asyncio.gather function?",
            model_name="llama3.2:3b",
            ollama_url="http://localhost:11434",
            max_iterations=4,
            context_window=16384,
        )

        # Should still provide an answer
        assert result.final_answer is not None
        assert len(result.final_answer) > 0

    def test_ambiguous_goal(self):
        """Test with somewhat ambiguous goal."""
        result = browse_agent(
            goal="Tell me about Rust",
            model_name="llama3.2:3b",
            ollama_url="http://localhost:11434",
            max_iterations=4,
            context_window=16384,
        )

        # Agent should handle ambiguity and search
        assert result.final_answer is not None
        assert len(result.searches_performed) > 0

    def test_agent_thinking_history(self):
        """Test that thinking history is properly recorded."""
        result = browse_agent(
            goal="What is Docker?",
            model_name="llama3.2:3b",
            ollama_url="http://localhost:11434",
            max_iterations=3,
            context_window=16384,
        )

        # Check thinking history was recorded
        assert len(result.thinking_history) > 0

        # Check each history entry has expected fields
        for entry in result.thinking_history:
            assert "iteration" in entry
            assert "thinking" in entry
            assert "action" in entry

        # Check iterations are sequential
        iterations = [entry["iteration"] for entry in result.thinking_history]
        assert iterations == sorted(iterations)
        assert iterations[0] == 1
