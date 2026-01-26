"""BrowseExecutor - Executes browse actions using enhanced tools."""

import json
import logging
from typing import Any, Dict, List

from llama_index.core.tools import FunctionTool

from .state import BrowseState, WorkflowPhase

logger = logging.getLogger(__name__)


class BrowseExecutor:
    """Executes browse actions using enhanced tools."""

    def __init__(self, tools: Dict[str, FunctionTool]):
        """Initialize executor with tools.

        Args:
            tools: Dictionary of tool name to FunctionTool
        """
        self.tools = tools

    def _parse_tool_result(self, result: Any) -> Any:
        """Parse tool result handling ToolOutput, JSON string, or parsed object.

        LlamaIndex FunctionTool.acall() can return:
        1. ToolOutput object with .content or .raw_output attribute
        2. JSON string (when tool returns json.dumps(...))
        3. Pre-parsed Python object (when LlamaIndex auto-parses the JSON)
        4. List of TextContent objects (from MCP servers)

        Args:
            result: Raw result from tool.acall()

        Returns:
            Parsed Python object (dict or list)

        Raises:
            ValueError: If result cannot be parsed
        """
        # Debug logging
        logger.info(f"_parse_tool_result: result type = {type(result)}")

        # Case 1: ToolOutput object - try raw_output first, then content
        if not isinstance(result, str) and hasattr(result, "raw_output"):
            result_data = result.raw_output
            logger.info(
                f"_parse_tool_result: extracted raw_output, type = {type(result_data)}"
            )
        elif not isinstance(result, str) and hasattr(result, "content"):
            result_data = result.content
            logger.info(
                f"_parse_tool_result: extracted content, type = {type(result_data)}"
            )
        else:
            # Case 2, 3, or 4: Direct result
            result_data = result
            logger.info("_parse_tool_result: using result directly")

        # Handle MCP CallToolResult type (has .content attribute)
        if hasattr(result_data, "content") and not isinstance(result_data, str):
            result_data = result_data.content
            logger.info(
                f"_parse_tool_result: extracted from CallToolResult, type = {type(result_data)}"
            )

        # Case 4: List of TextContent objects (from MCP servers)
        if isinstance(result_data, list) and len(result_data) > 0:
            # Check if it's a list of TextContent objects
            first_item = result_data[0]
            logger.info(f"_parse_tool_result: first_item type = {type(first_item)}")
            if hasattr(first_item, "text") and hasattr(first_item, "type"):
                # Extract text from first TextContent object
                result_data = first_item.text
                logger.info("Extracted text from TextContent object")

        # Now handle string vs already-parsed
        logger.info(f"_parse_tool_result: final result_data type = {type(result_data)}")
        if isinstance(result_data, str):
            # Case 2: JSON string - parse it
            try:
                return json.loads(result_data)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Tool result is string but not valid JSON: {e}. "
                    f"First 200 chars: {result_data[:200]!r}"
                )
        elif isinstance(result_data, (dict, list)):
            # Case 3: Already parsed - return as-is
            return result_data
        else:
            # Unexpected type
            raise ValueError(
                f"Unexpected tool result type: {type(result_data)}. "
                f"Expected str, dict, or list, got {result_data!r}"
            )

    def _rerank_search_results(
        self, state: BrowseState, search_results: List[Dict]
    ) -> List[Dict]:
        """Rerank search results by relevance if reranker enabled.

        Args:
            state: Current browse state
            search_results: List of search result dicts

        Returns:
            Reranked search results (or original if reranker disabled/failed)
        """
        if not state.reranker_model:
            logger.info("Reranker disabled, using DDG order")
            return search_results

        if not search_results:
            return search_results

        try:
            logger.info(f"Reranking {len(search_results)} results")

            # Get device from state (default to 'cpu')
            device = getattr(state, "rag_device", "cpu") or "cpu"

            # Get reranker from ModelManager singleton
            from tensortruth.utils.web_search import (
                get_reranker_for_web,
                rerank_search_results,
            )

            reranker = get_reranker_for_web(
                model_name=state.reranker_model,
                device=device,
                top_n=len(search_results),
            )

            # Rerank by title + snippet relevance
            ranked_results = rerank_search_results(
                query=state.query,
                results=search_results,
                top_n=len(search_results),
                reranker=reranker,
            )

            # Attach scores to results for later use
            reranked: List[Dict] = []
            for result, score in ranked_results:
                # Add relevance_score to result dict
                result_with_score: Dict = dict(result)  # type: ignore[assignment]
                result_with_score["relevance_score"] = float(score)
                reranked.append(result_with_score)
                logger.debug(
                    f"Reranked: {result.get('url')} -> score={float(score):.2f}"
                )
            top_title = reranked[0]["title"]
            top_score = reranked[0].get("relevance_score", 0.0)
            logger.info(
                f"Reranking complete: top = {top_title} (score: {top_score:.2f})"
            )

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return search_results  # Fallback to DDG order

    async def execute_search(self, state: BrowseState) -> BrowseState:
        """Execute search_web and update state.

        Args:
            state: Current browse state

        Returns:
            Updated browse state with search results
        """
        # Generate 3 diverse queries
        queries = self._generate_queries(state.query)
        logger.info(f"Executing search with {len(queries)} queries: {queries}")

        # Call search_web tool
        search_tool = self.tools["search_web"]
        result = await search_tool.acall(queries=queries)

        logger.info(f"Search returned {len(str(result))} chars")

        # Parse result (handles all three cases)
        search_results = self._parse_tool_result(result)

        # Rerank results by relevance
        search_results = self._rerank_search_results(state, search_results)

        state.search_results = search_results
        state.phase = WorkflowPhase.SEARCHED
        state.actions_taken.append("search_web")

        logger.info(f"Parsed {len(state.search_results)} search results")

        return state

    async def execute_fetch(self, state: BrowseState) -> BrowseState:
        """Execute unified fetch+rerank+fit pipeline.

        Args:
            state: Current browse state

        Returns:
            Updated browse state with fitted pages and sources
        """
        if not state.search_results:
            logger.warning("No search results to fetch")
            state.phase = WorkflowPhase.COMPLETE
            return state

        logger.info(
            f"Executing unified pipeline: {len(state.search_results)} search results"
        )

        # Import and create pipeline
        from tensortruth.core.source_pipeline import SourceFetchPipeline

        # Create pipeline instance
        pipeline = SourceFetchPipeline(
            query=state.query,
            max_pages=state.min_pages_required,
            context_window=state.max_content_chars // 4,  # Convert chars to tokens
            reranker_model=state.reranker_model,
            reranker_device=state.rag_device or "cpu",
            rerank_content_threshold=0.1,  # Can be made configurable
            max_source_context_pct=0.15,
            input_context_pct=0.6,
            custom_instructions=None,
            progress_callback=None,
        )

        try:
            # Execute pipeline: fetch + rerank + fit
            fitted_pages, source_nodes, allocations = await pipeline.execute(
                state.search_results
            )

            logger.info(
                f"Pipeline complete: {len(fitted_pages)} fitted pages, "
                f"{len(source_nodes)} total sources"
            )

            # Convert fitted tuples to dict format for state.pages
            state.pages = [
                {
                    "url": url,
                    "title": title,
                    "content": content,
                    "status": "success",
                }
                for url, title, content in fitted_pages
            ]

            # Store source nodes for later extraction
            state.source_nodes = source_nodes  # type: ignore[attr-defined]

            # Calculate total content chars from allocations
            state.total_content_chars = sum(allocations.values())
            state.content_overflow = False  # Pipeline handles fitting

            state.phase = WorkflowPhase.FETCHED
            state.fetch_iterations += 1
            state.actions_taken.append("fetch_sources")

            logger.info(
                f"Fetch completed: {len(state.pages)} pages, "
                f"{state.total_content_chars} chars fitted to context"
            )

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            state.pages = []
            state.source_nodes = []  # type: ignore[attr-defined]
            state.phase = WorkflowPhase.COMPLETE

        return state

    def _generate_queries(self, query: str) -> List[str]:
        """Generate 3 diverse queries.

        Args:
            query: Original user query

        Returns:
            List of 3 diverse search queries
        """
        return [
            f"{query} overview",
            f"{query} technical details",
            f"{query} recent 2026",
        ]
