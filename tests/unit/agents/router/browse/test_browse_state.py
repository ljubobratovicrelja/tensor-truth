"""Tests for BrowseState."""

from tensortruth.agents.router.browse.state import BrowseState, WorkflowPhase


class TestBrowseStateInitialization:
    """Test BrowseState initialization."""

    def test_browse_state_initialization(self):
        """Should initialize with required fields and defaults."""
        state = BrowseState(
            query="test query",
            phase=WorkflowPhase.INITIAL,
        )

        assert state.query == "test query"
        assert state.phase == WorkflowPhase.INITIAL
        assert state.search_results is None
        assert state.pages is None
        assert state.min_pages_required == 5
        assert state.max_content_chars == 0
        assert state.total_content_chars == 0
        assert state.content_overflow is False
        assert state.fetch_iterations == 0
        assert state.max_fetch_iterations == 3
        assert state.next_url_index == 0
        assert state.reranker_model is None

    def test_browse_state_with_custom_values(self):
        """Should allow custom initialization values."""
        state = BrowseState(
            query="test",
            phase=WorkflowPhase.SEARCHED,
            min_pages_required=5,
            max_content_chars=100000,
            reranker_model="test-reranker",
        )

        assert state.min_pages_required == 5
        assert state.max_content_chars == 100000
        assert state.reranker_model == "test-reranker"


class TestWorkflowPhaseTransitions:
    """Test workflow phase transitions."""

    def test_workflow_phase_transitions(self):
        """Should track workflow phases correctly."""
        state = BrowseState(query="test", phase=WorkflowPhase.INITIAL)
        assert state.phase == WorkflowPhase.INITIAL

        state.phase = WorkflowPhase.SEARCHED
        assert state.phase == WorkflowPhase.SEARCHED

        state.phase = WorkflowPhase.FETCHED
        assert state.phase == WorkflowPhase.FETCHED

        state.phase = WorkflowPhase.COMPLETE
        assert state.phase == WorkflowPhase.COMPLETE


class TestOverflowDetection:
    """Test overflow detection logic."""

    def test_overflow_detection(self):
        """Should detect content overflow."""
        state = BrowseState(
            query="test",
            phase=WorkflowPhase.INITIAL,
            max_content_chars=1000,
        )

        assert state.content_overflow is False

        # Simulate overflow
        state.total_content_chars = 1200
        state.content_overflow = True

        assert state.content_overflow is True
        assert state.total_content_chars > state.max_content_chars


class TestMinPagesEnforcement:
    """Test min pages enforcement."""

    def test_min_pages_enforcement(self):
        """Should track min_pages_required."""
        state = BrowseState(
            query="test",
            phase=WorkflowPhase.INITIAL,
            min_pages_required=3,
        )

        assert state.min_pages_required == 3

        # Simulate fetching pages
        state.pages = [{"url": "url1"}, {"url": "url2"}]
        assert len(state.pages) < state.min_pages_required

        state.pages.append({"url": "url3"})
        assert len(state.pages) == state.min_pages_required


class TestIsCompleteLogic:
    """Test is_complete() logic."""

    def test_is_complete_logic(self):
        """Should check if workflow is complete."""
        state = BrowseState(query="test", phase=WorkflowPhase.INITIAL)
        assert state.is_complete() is False

        state.phase = WorkflowPhase.SEARCHED
        assert state.is_complete() is False

        state.phase = WorkflowPhase.FETCHED
        assert state.is_complete() is False

        state.phase = WorkflowPhase.COMPLETE
        assert state.is_complete() is True


class TestToDictSerialization:
    """Test to_dict() serialization."""

    def test_to_dict_serialization(self):
        """Should serialize state to dict."""
        state = BrowseState(
            query="test query",
            phase=WorkflowPhase.FETCHED,
            min_pages_required=3,
        )
        state.search_results = [{"url": "url1"}, {"url": "url2"}]
        state.pages = [{"url": "url1", "status": "success"}]
        state.total_content_chars = 5000
        state.content_overflow = False
        state.actions_taken = ["search", "fetch"]
        state.iteration_count = 2

        result = state.to_dict()

        assert result["query"] == "test query"
        assert result["phase"] == "fetched"
        assert result["actions_taken"] == ["search", "fetch"]
        assert result["iteration_count"] == 2
        assert result["search_result_count"] == 2
        assert result["page_count"] == 1
        assert result["content_overflow"] is False
        assert result["total_content_chars"] == 5000

    def test_to_dict_with_no_results(self):
        """Should handle None search results and pages."""
        state = BrowseState(
            query="test",
            phase=WorkflowPhase.INITIAL,
        )

        result = state.to_dict()

        assert result["search_result_count"] == 0
        assert result["page_count"] == 0
