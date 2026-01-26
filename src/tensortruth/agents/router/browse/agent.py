"""BrowseAgent - Router-based web research agent."""

import logging
from typing import Dict, List, Optional, cast

from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.config import AgentCallbacks, AgentResult
from tensortruth.agents.router.base import RouterAgent
from tensortruth.agents.router.state import RouterState
from tensortruth.core.sources import SourceNode
from tensortruth.core.synthesis import (
    CitationStyle,
    SynthesisConfig,
    synthesize_with_llm_stream,
)

from .executor import BrowseExecutor
from .router import BrowseRouter
from .state import BrowseState, WorkflowPhase

logger = logging.getLogger(__name__)


class BrowseAgent(RouterAgent):
    """Browse agent using router pattern.

    Implements a two-phase architecture:
    1. Router (fast small model) - Routes workflow decisions
    2. Synthesis (session model) - Generates final answer

    Features:
    - Overflow protection based on context window
    - Deterministic fallback for reliability
    - Reranking for better search results
    """

    def __init__(
        self,
        router_llm: Ollama,
        synthesis_llm: Ollama,
        tools: Dict[str, FunctionTool],
        min_pages_required: int = 5,
        max_iterations: int = 10,
        reranker_model: Optional[str] = None,
        rag_device: str = "cpu",
        context_window: int = 16384,
    ):
        """Initialize BrowseAgent.

        Args:
            router_llm: Fast model for routing decisions
            synthesis_llm: Session model for answer synthesis
            tools: Dictionary of tool name to FunctionTool
            min_pages_required: Minimum pages to fetch before synthesis
            max_iterations: Maximum workflow iterations
            reranker_model: Optional reranker model for search results
            rag_device: Device for reranker model (cpu, cuda, mps)
            context_window: Context window of synthesis model
        """
        super().__init__(router_llm, synthesis_llm, tools, max_iterations)

        self.router = BrowseRouter(router_llm)
        self.executor = BrowseExecutor(tools)
        self.min_pages_required = min_pages_required
        self.reranker_model = reranker_model
        self.rag_device = rag_device

        # Calculate max content from synthesis model's context window
        # Reserve space for: prompt template (~2000 tokens), output (~2000 tokens)
        # Rough estimate: 1 token ≈ 4 chars
        prompt_overhead = 2000 * 4  # ~8000 chars for prompt
        output_buffer = 2000 * 4  # ~8000 chars for output
        self.max_content_chars = (context_window * 4) - prompt_overhead - output_buffer

        logger.info(
            f"BrowseAgent initialized: context_window={context_window}, "
            f"max_content_chars={self.max_content_chars}, "
            f"min_pages={min_pages_required}"
        )

    def _create_initial_state(self, query: str) -> RouterState:
        """Create initial BrowseState with calculated max_content.

        Args:
            query: User query

        Returns:
            Initial browse state
        """
        return BrowseState(
            query=query,
            phase=WorkflowPhase.INITIAL,
            min_pages_required=self.min_pages_required,
            max_content_chars=self.max_content_chars,
            reranker_model=self.reranker_model,
            rag_device=self.rag_device,
        )

    async def route(self, state: RouterState) -> str:
        """Route using BrowseRouter.

        Args:
            state: Current state

        Returns:
            Next action to execute
        """
        return await self.router.route(cast(BrowseState, state))

    async def execute(
        self,
        action: str,
        state: RouterState,
        callbacks: Optional[AgentCallbacks] = None,
    ) -> RouterState:
        """Execute action using BrowseExecutor.

        Args:
            action: Action to execute
            state: Current state
            callbacks: Optional callbacks for tool calls

        Returns:
            Updated state

        Raises:
            ValueError: If action is unknown
        """
        browse_state = cast(BrowseState, state)
        if action == "search_web":
            # Build params for callback
            queries = self.executor._generate_queries(browse_state.query)
            if callbacks and callbacks.on_tool_call:
                callbacks.on_tool_call("search_web", {"queries": queries})
            return await self.executor.execute_search(browse_state)
        elif action == "fetch_sources":
            # Build params for callback - show all search results
            if callbacks and callbacks.on_tool_call:
                url_count = (
                    len(browse_state.search_results)
                    if browse_state.search_results
                    else 0
                )
                callbacks.on_tool_call(
                    "fetch_sources",
                    {
                        "search_result_count": url_count,
                        "target_pages": browse_state.min_pages_required,
                    },
                )
            # Pass callbacks to enable pipeline progress reporting
            return await self.executor.execute_fetch(browse_state, callbacks)
        elif action == "done":
            browse_state.phase = WorkflowPhase.COMPLETE
            return browse_state
        else:
            raise ValueError(f"Unknown action: {action}")

    def _build_result(self, state: RouterState, final_answer: str) -> AgentResult:
        """Build AgentResult with rich source metadata.

        Overrides base class to add SourceNode objects with titles, scores, etc.

        Args:
            state: Final browse state
            final_answer: Synthesized answer

        Returns:
            AgentResult with sources populated
        """
        return AgentResult(
            final_answer=final_answer,
            iterations=state.iteration_count,
            tools_called=state.actions_taken,
            urls_browsed=self._extract_urls(state),
            sources=self._extract_sources(state),
        )

    async def _synthesize(self, state: RouterState, callbacks: AgentCallbacks) -> str:
        """Synthesize answer using core synthesis engine.

        Args:
            state: Final browse state
            callbacks: Agent callbacks for streaming

        Returns:
            Final synthesized answer
        """
        browse_state = cast(BrowseState, state)

        # Build synthesis config
        synthesis_config = SynthesisConfig(
            query=browse_state.query,
            context_window=self.synthesis_llm.context_window,
            citation_style=CitationStyle.HYPERLINK,  # Match web command format
        )

        # Send progress update before synthesis
        if callbacks.on_progress:
            callbacks.on_progress("Synthesizing answer from fetched pages...")

        # Use core synthesis engine
        logger.info("Starting answer synthesis with core synthesis engine...")

        # Ensure pages is not None (it should be populated by this point)
        pages = browse_state.pages if browse_state.pages else []

        # Debug: Log titles being passed to synthesis
        logger.info(f"Browse agent passing {len(pages)} pages to synthesis:")
        for i, p in enumerate(pages, 1):
            logger.info(
                f"  {i}. title='{p.get('title', 'NO TITLE')}' url={p.get('url', 'NO URL')[:50]}..."
            )

        full_answer = ""
        async for token in synthesize_with_llm_stream(
            self.synthesis_llm,
            synthesis_config,
            pages,  # type: ignore[arg-type]
        ):
            full_answer += token
            if callbacks.on_token:
                callbacks.on_token(token)

        logger.info(f"Synthesis complete: {len(full_answer)} chars")
        return full_answer

    def _extract_urls(self, state: RouterState) -> List[str]:
        """Extract URLs from browse state.

        Args:
            state: Browse state

        Returns:
            List of all fetched URLs (successful and failed)
        """
        browse_state = cast(BrowseState, state)
        if not browse_state.pages:
            return []
        # Return all pages regardless of status for transparency
        return [p["url"] for p in browse_state.pages]

    def _extract_sources(self, state: RouterState) -> List[SourceNode]:
        """Extract rich source metadata from browse state.

        The pipeline (SourceFetchPipeline) already built SourceNode objects
        with all metadata (titles, scores, status, errors). We just return them.

        Args:
            state: Browse state

        Returns:
            List of SourceNode objects with titles, scores, content, etc.
        """
        browse_state = cast(BrowseState, state)

        # Pipeline already built source_nodes with all metadata
        if browse_state.source_nodes:
            logger.info(
                f"_extract_sources: Returning {len(browse_state.source_nodes)} "
                f"sources from pipeline"
            )
            return browse_state.source_nodes

        # Fallback if no source_nodes (shouldn't happen with pipeline)
        logger.warning(
            "_extract_sources: No source_nodes from pipeline, returning empty"
        )
        return []

    def get_metadata(self) -> Dict:
        """Get BrowseAgent metadata.

        Returns:
            Dict with agent metadata
        """
        return {
            "name": "browse",
            "description": "Router-based web research agent",
            "agent_type": "router",
            "capabilities": [
                "web_search",
                "content_synthesis",
                "overflow_protection",
                "reranking",
            ],
        }
