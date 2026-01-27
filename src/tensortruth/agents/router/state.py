"""Base state class for router-based agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RouterState(ABC):
    """Base class for router agent state.

    This abstract class defines the minimal state required for router-based
    agents that use iterative workflows. Subclasses should add domain-specific
    fields for their workflows.

    Note: Not all RouterAgent subclasses need this - simple agents like
    ChatAgent may use minimal state or none at all. This is designed for
    agents with iterative route → execute loops like BrowseAgent.

    Attributes:
        query: The user's original query/request
        phase: Current workflow phase (subclass-specific Enum)
        actions_taken: List of actions executed so far
        iteration_count: Number of iterations completed
        max_iterations: Maximum iterations allowed
    """

    query: str
    phase: Any  # Subclasses should use specific Enum types
    actions_taken: List[str] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 10

    @abstractmethod
    def is_complete(self) -> bool:
        """Check if workflow is complete.

        Subclasses implement their specific completion logic (e.g., check
        if phase == COMPLETE or if certain conditions are met).

        Returns:
            True if workflow should stop, False otherwise
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for logging and debugging.

        Subclasses should include all relevant fields in serialization.

        Returns:
            Dict representation of state
        """
        pass
