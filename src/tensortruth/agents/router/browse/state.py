"""Browse agent state with overflow tracking."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from tensortruth.agents.router.state import RouterState


class WorkflowPhase(Enum):
    """Browse workflow phases."""

    INITIAL = "initial"
    SEARCHED = "searched"
    FETCHED = "fetched"
    COMPLETE = "complete"


@dataclass
class BrowseState(RouterState):
    """State for browse agent with overflow tracking.

    Extends RouterState with browse-specific fields for tracking search
    results, fetched pages, and content overflow protection.

    Attributes:
        query: User's query (inherited from RouterState)
        phase: Current workflow phase (WorkflowPhase enum)
        actions_taken: List of actions executed (inherited)
        iteration_count: Number of iterations (inherited)
        max_iterations: Maximum iterations allowed (inherited)
        search_results: Search results from DuckDuckGo
        pages: Fetched pages with content
        min_pages_required: Minimum pages to fetch before synthesis
        max_content_chars: Maximum content size (calculated from context window)
        total_content_chars: Current total content size
        content_overflow: Whether content limit was exceeded
        fetch_iterations: Number of fetch attempts
        max_fetch_iterations: Maximum fetch retries
        next_url_index: Next URL index to fetch
        reranker_model: Reranker model to use (if any)
    """

    # Browse-specific fields
    search_results: Optional[List[Dict]] = None
    pages: Optional[List[Dict]] = None
    min_pages_required: int = 5
    max_content_chars: int = field(default=0)  # Calculated from context window
    total_content_chars: int = 0
    content_overflow: bool = False
    fetch_iterations: int = 0
    max_fetch_iterations: int = 3
    next_url_index: int = 0
    reranker_model: Optional[str] = None
    rag_device: Optional[str] = None

    def is_complete(self) -> bool:
        """Check if workflow is complete.

        Returns:
            True if phase is COMPLETE, False otherwise
        """
        return self.phase == WorkflowPhase.COMPLETE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for logging.

        Returns:
            Dict representation of state with key metrics
        """
        return {
            "query": self.query,
            "phase": self.phase.value,
            "actions_taken": self.actions_taken,
            "iteration_count": self.iteration_count,
            "search_result_count": (
                len(self.search_results) if self.search_results else 0
            ),
            "page_count": len(self.pages) if self.pages else 0,
            "content_overflow": self.content_overflow,
            "total_content_chars": self.total_content_chars,
        }
