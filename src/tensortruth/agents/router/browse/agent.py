"""BrowseAgent - Router-based web research agent."""

import logging
from typing import Dict, List, Optional, cast

from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.config import AgentCallbacks
from tensortruth.agents.router.base import RouterAgent
from tensortruth.agents.router.state import RouterState

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
        self, action: str, state: RouterState, callbacks: AgentCallbacks = None
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
        elif action == "fetch_pages_batch":
            # Build params for callback
            urls = self.executor._select_urls(browse_state)
            if callbacks and callbacks.on_tool_call:
                callbacks.on_tool_call("fetch_pages_batch", {"urls": urls})
            return await self.executor.execute_fetch(browse_state)
        elif action == "done":
            browse_state.phase = WorkflowPhase.COMPLETE
            return browse_state
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _synthesize(self, state: RouterState, callbacks: AgentCallbacks) -> str:
        """Synthesize answer from browse state with content trimming.

        Args:
            state: Final browse state
            callbacks: Agent callbacks for streaming

        Returns:
            Final synthesized answer
        """
        browse_state = cast(BrowseState, state)

        # Format pages for synthesis
        pages_content = self._format_pages_for_synthesis(browse_state.pages)

        # Check if content needs trimming
        content_size = len(pages_content)
        if content_size > browse_state.max_content_chars:
            logger.warning(
                f"Content size ({content_size} chars) exceeds limit "
                f"({browse_state.max_content_chars} chars). Trimming to fit context window."
            )
            pages_content = pages_content[: browse_state.max_content_chars]
            pages_content += "\n\n[...Content trimmed to fit context window...]"

        # Build synthesis prompt
        prompt_template = """Synthesize a comprehensive answer from the research results below.

Query: {query}

Research Results ({page_count} pages fetched):
{pages_content}

Instructions:
- Provide a clear, comprehensive answer to the query
- Use inline citations with [Source N] format
- Synthesize information from multiple sources
- Include relevant details and examples
- If information is conflicting, note the differences

Answer:"""

        prompt = prompt_template.format(
            query=browse_state.query,
            page_count=len(browse_state.pages) if browse_state.pages else 0,
            pages_content=pages_content,
        )

        # Verify final prompt fits in context window
        prompt_size = len(prompt)
        expected_output_size = 2000 * 4  # ~8000 chars
        total_size = prompt_size + expected_output_size

        if total_size > (self.synthesis_llm.context_window * 4):
            logger.error(
                f"Prompt too large: {prompt_size} chars + {expected_output_size} output "
                f"> {self.synthesis_llm.context_window * 4} context limit"
            )
            # Further trim if needed
            allowed_content = browse_state.max_content_chars - (
                prompt_size - content_size
            )
            if allowed_content > 0:
                pages_content = pages_content[:allowed_content]
                pages_content += "\n\n[...Content trimmed to fit context window...]"
                prompt = prompt_template.format(
                    query=browse_state.query,
                    page_count=len(browse_state.pages) if browse_state.pages else 0,
                    pages_content=pages_content,
                )

        # Stream synthesis
        logger.info("Starting answer synthesis...")

        # Send progress update before synthesis
        if callbacks.on_progress:
            callbacks.on_progress("Synthesizing answer from fetched pages...")

        full_answer = ""
        async for chunk in await self.synthesis_llm.astream_complete(prompt):
            if chunk.delta:
                full_answer += chunk.delta
                if callbacks.on_token:
                    callbacks.on_token(chunk.delta)

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

    def _format_pages_for_synthesis(self, pages: Optional[List[Dict]]) -> str:
        """Format pages for synthesis prompt.

        Args:
            pages: List of page dictionaries

        Returns:
            Formatted string for prompt
        """
        if not pages:
            return "No pages fetched."

        formatted = []
        for i, page in enumerate(pages, 1):
            if page["status"] == "success" and page.get("content"):
                formatted.append(
                    f"## Source {i}: {page['title']}\n"
                    f"URL: {page['url']}\n\n{page['content']}\n"
                )

        if not formatted:
            return "No successful pages to synthesize."

        return "\n---\n\n".join(formatted)

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
