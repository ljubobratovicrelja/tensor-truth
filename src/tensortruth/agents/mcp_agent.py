"""MCP-based web browsing agent using LlamaIndex FunctionAgent."""

import asyncio
import logging
import re
import warnings
from typing import Callable, Optional

from llama_index.core.agent.workflow.function_agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from .config import AgentResult, MCPServerConfig
from .server_registry import create_default_registry

logger = logging.getLogger(__name__)


# Pattern to detect max iterations errors from LlamaIndex
# This covers variations in error message formatting
_MAX_ITERATIONS_PATTERN = re.compile(
    r"(max[_\s]?iterations?|iteration[_\s]?limit|exceeded.*iterations?)",
    re.IGNORECASE,
)


def _is_max_iterations_error(error: Exception) -> bool:
    """Check if an exception indicates max iterations was reached.

    Args:
        error: The exception to check

    Returns:
        True if this appears to be a max iterations error
    """
    # Check exception type name (handles subclasses without importing)
    error_type = type(error).__name__.lower()
    if "iteration" in error_type or "maxiter" in error_type:
        return True

    # Fall back to message pattern matching
    error_msg = str(error)
    if _MAX_ITERATIONS_PATTERN.search(error_msg):
        return True

    # Log unrecognized errors for debugging
    logger.debug(
        f"Unrecognized agent error type: {type(error).__name__}, message: {error_msg[:200]}"
    )
    return False


class ToolTracker:
    """Tracks tool calls during agent execution."""

    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None):
        self.tools_called: list[str] = []
        self.urls_browsed: list[str] = []
        self.search_queries: list[str] = []
        self.progress_callback = progress_callback

    def _report_progress(self, message: str) -> None:
        """Report progress if callback is configured."""
        if self.progress_callback:
            self.progress_callback(message)
        logger.debug(message)

    def wrap_tools(self, tools: list) -> list:
        """Wrap tools to track calls and report progress.

        Args:
            tools: List of LlamaIndex tools

        Returns:
            List of wrapped tools that track usage
        """
        wrapped = []
        for tool in tools:
            if tool.metadata.name == "fetch_page":
                wrapped.append(self._wrap_fetch_page(tool))
            elif tool.metadata.name == "search_web":
                wrapped.append(self._wrap_search_web(tool))
            else:
                wrapped.append(tool)
        return wrapped

    def _wrap_fetch_page(self, original_tool) -> FunctionTool:
        """Wrap fetch_page tool to track URLs."""

        async def tracked_fetch_page(url: str, timeout: int = 10) -> str:
            """Fetch a web page and convert it to markdown."""
            # Track the URL
            if url not in self.urls_browsed:
                self.urls_browsed.append(url)
            self.tools_called.append("fetch_page")

            # Report progress
            self._report_progress(f"📄 Fetching: {url}")

            # Call original tool
            return await original_tool.acall(url=url, timeout=timeout)

        return FunctionTool.from_defaults(
            async_fn=tracked_fetch_page,
            name="fetch_page",
            description=original_tool.metadata.description,
        )

    def _wrap_search_web(self, original_tool) -> FunctionTool:
        """Wrap search_web tool to track queries."""

        async def tracked_search_web(query: str, max_results: int = 10) -> str:
            """Search DuckDuckGo for information."""
            # Track the query
            self.search_queries.append(query)
            self.tools_called.append("search_web")

            # Report progress
            self._report_progress(f"🔎 Searching: {query}")

            # Call original tool
            return await original_tool.acall(query=query, max_results=max_results)

        return FunctionTool.from_defaults(
            async_fn=tracked_search_web,
            name="search_web",
            description=original_tool.metadata.description,
        )


# System prompt for the web research agent
AGENT_SYSTEM_PROMPT = """\
You are a web research agent with access to tools for searching and fetching web content.

IMPORTANT: You MUST use the available tools to gather information. Do NOT answer from memory.

Available tools:
- search_web: Search DuckDuckGo for information. Always start with this.
- fetch_page: Fetch and read a specific URL to get detailed content.

Required workflow:
1. ALWAYS call search_web first to find relevant sources
2. Call fetch_page on 2-3 of the most relevant URLs from search results
3. Synthesize information from the fetched pages
4. Include source citations as markdown links: [Title](url)

Your final answer MUST include:
- Information gathered from the web (not from memory)
- Source citations with URLs for every fact
- A "Sources:" section at the end listing all URLs consulted

Never provide an answer without first using the tools to gather current information.
"""


class MCPBrowseAgent:
    """Web browsing agent using LlamaIndex FunctionAgent with MCP tools.

    This agent:
    1. Connects to configured MCP servers to load tools
    2. Uses FunctionAgent for autonomous reasoning and tool selection
    3. Synthesizes findings into a comprehensive answer

    Example:
        agent = MCPBrowseAgent(
            model_name="llama3.1:8b",
            ollama_url="http://localhost:11434"
        )
        result = await agent.run("What are the latest Python 3.12 features?")
        print(result.final_answer)
    """

    def __init__(
        self,
        model_name: str,
        ollama_url: str,
        mcp_servers: Optional[list[MCPServerConfig]] = None,
        synthesis_model: Optional[str] = None,
        max_iterations: int = 10,
        context_window: int = 8192,
        progress_callback: Optional[Callable[[str], None]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the MCP browse agent.

        Args:
            model_name: Ollama model for reasoning (e.g., "llama3.1:8b")
            ollama_url: Ollama API base URL
            mcp_servers: List of MCP server configs. If None, uses defaults.
            synthesis_model: Optional different model for final synthesis.
                           If None, uses model_name for everything.
            max_iterations: Maximum reasoning iterations (default: 10)
            context_window: Context window size (default: 8192)
            progress_callback: Optional callback for progress updates
            stream_callback: Optional callback for streaming final answer tokens
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.synthesis_model = synthesis_model or model_name
        self.max_iterations = max_iterations
        self.context_window = context_window
        self.progress_callback = progress_callback
        self.stream_callback = stream_callback

        # Set up server registry
        self._registry = create_default_registry()
        if mcp_servers:
            for server in mcp_servers:
                self._registry.register(server)

        self._agent: Optional[FunctionAgent] = None
        self._tool_tracker: Optional[ToolTracker] = None

    def _report_progress(self, message: str) -> None:
        """Report progress if callback is configured."""
        if self.progress_callback:
            self.progress_callback(message)
        logger.info(message)

    async def _load_tools(self) -> list[FunctionTool]:
        """Load tools from all configured MCP servers."""
        self._report_progress("Loading MCP tools...")
        tools = await self._registry.load_tools()
        self._report_progress(f"Loaded {len(tools)} tools from MCP servers")
        return tools

    def _create_llm(self, model_name: str) -> Ollama:
        """Create an Ollama LLM instance."""
        return Ollama(
            model=model_name,
            base_url=self.ollama_url,
            temperature=0.2,
            context_window=self.context_window,
            additional_kwargs={
                "num_ctx": self.context_window
            },  # Explicitly set Ollama context
            request_timeout=120.0,
        )

    def _create_synthesis_llm(self) -> Ollama:
        """Create an LLM instance for final synthesis (higher quality)."""
        return Ollama(
            model=self.synthesis_model,
            base_url=self.ollama_url,
            temperature=0.1,  # More deterministic for final answer
            context_window=self.context_window,
            additional_kwargs={
                "num_ctx": self.context_window
            },  # Explicitly set Ollama context
            request_timeout=120.0,
        )

    def _build_partial_response(self, goal: str) -> str:
        """Build a partial response from gathered information when max iterations hit.

        Args:
            goal: The original research goal

        Returns:
            A partial response summarizing what was found
        """
        parts = [f"Research goal: {goal}\n"]

        if self._tool_tracker:
            if self._tool_tracker.search_queries:
                parts.append("Searches performed:")
                for query in self._tool_tracker.search_queries:
                    parts.append(f"- {query}")
                parts.append("")

            if self._tool_tracker.urls_browsed:
                parts.append("Pages consulted:")
                for url in self._tool_tracker.urls_browsed:
                    parts.append(f"- {url}")
                parts.append("")

        parts.append(
            "Note: The agent was gathering information when the iteration limit was reached. "
            "The synthesis below is based on partially gathered data."
        )

        return "\n".join(parts)

    async def run(self, goal: str) -> AgentResult:
        """Execute the agent with the given goal.

        Args:
            goal: The research goal/question to answer

        Returns:
            AgentResult with the final answer and metadata
        """
        if not goal or not goal.strip():
            return AgentResult(
                final_answer="Error: Please provide a research goal.",
                error="Empty goal provided",
            )

        try:
            self._report_progress(f"Starting research: {goal}")

            # Load tools from MCP servers
            tools = await self._load_tools()

            if not tools:
                return AgentResult(
                    final_answer="Error: No tools available. Check MCP server configuration.",
                    error="No tools loaded from MCP servers",
                )

            # Create tool tracker and wrap tools for progress reporting
            self._tool_tracker = ToolTracker(progress_callback=self.progress_callback)
            wrapped_tools = self._tool_tracker.wrap_tools(tools)

            # Create LLM for reasoning (fast model)
            reasoning_llm = self._create_llm(self.model_name)

            self._report_progress("Initializing reasoning agent...")

            # Create FunctionAgent with reasoning model (uses native tool calling)
            self._agent = FunctionAgent(
                tools=wrapped_tools,
                llm=reasoning_llm,
                system_prompt=AGENT_SYSTEM_PROMPT,
            )

            self._report_progress("Executing research...")

            # Execute the agent with reasoning model
            hit_max_iterations = False
            try:
                response = await self._agent.run(
                    user_msg=goal,
                    max_iterations=self.max_iterations,
                )
                intermediate_answer = str(response)
            except Exception as agent_error:
                # Check if this is a max iterations error using robust detection
                if _is_max_iterations_error(agent_error):
                    hit_max_iterations = True
                    self._report_progress(
                        f"⚠️ Reached iteration limit ({self.max_iterations}). "
                        "Synthesizing available findings..."
                    )
                    # Create a partial response from what we've gathered
                    intermediate_answer = self._build_partial_response(goal)
                else:
                    # Re-raise other errors
                    raise

            # If synthesis model is different from reasoning model, use it for final answer
            if self.synthesis_model != self.model_name:
                self._report_progress("Synthesizing final answer...")

                # Create synthesis LLM (quality model) for final answer refinement
                synthesis_llm = self._create_synthesis_llm()

                # Create a synthesis prompt that includes the research findings
                synthesis_prompt = (
                    "You are an expert research assistant. Refine and improve "
                    "the following research findings into a comprehensive, "
                    "well-structured final answer.\n\n"
                    f"Research Goal: {goal}\n\n"
                    f"Research Findings:\n{intermediate_answer}\n\n"
                    "Guidelines:\n"
                    "1. Organize the answer with clear sections\n"
                    "2. Add proper citations and references\n"
                    "3. Ensure the answer is comprehensive and addresses "
                    "all aspects of the goal\n"
                    "4. Improve readability and structure\n"
                    "5. Add any missing context or explanations\n"
                    "6. Keep the original facts but present them more "
                    "professionally\n\n"
                    "Final Answer:"
                )

                # Use streaming if callback is provided
                if self.stream_callback:
                    final_answer = ""
                    for chunk in synthesis_llm.stream_complete(synthesis_prompt):
                        if chunk.delta:
                            final_answer += chunk.delta
                            self.stream_callback(chunk.delta)
                    final_answer = str(final_answer)
                else:
                    # Non-streaming fallback
                    completion = await synthesis_llm.acomplete(synthesis_prompt)
                    final_answer = str(completion)
            else:
                # Use the original response if same model
                final_answer = intermediate_answer

            # Inject sources section if URLs were browsed
            if self._tool_tracker and self._tool_tracker.urls_browsed:
                sources_md = "\n\n---\n\n**Sources consulted:**\n"
                for url in self._tool_tracker.urls_browsed:
                    sources_md += f"- {url}\n"
                final_answer = final_answer + sources_md

                # Stream the sources section too
                if self.stream_callback:
                    self.stream_callback(sources_md)

            # Add warning if max iterations was hit
            if hit_max_iterations:
                warning_md = (
                    f"\n\n⚠️ **Note:** The research agent reached its iteration limit "
                    f"({self.max_iterations}) before completing all searches. "
                    f"The answer above is based on partial findings. "
                    f"You can increase the limit in your preset settings "
                    f"(`agent_max_iterations`) for more thorough research.\n"
                )
                final_answer = final_answer + warning_md

                if self.stream_callback:
                    self.stream_callback(warning_md)

            self._report_progress("Research complete!")

            # Get tracked values
            tools_called = self._tool_tracker.tools_called if self._tool_tracker else []
            urls_browsed = self._tool_tracker.urls_browsed if self._tool_tracker else []

            return AgentResult(
                final_answer=final_answer,
                iterations=len(tools_called),  # Approximate iterations by tool calls
                tools_called=tools_called,
                urls_browsed=urls_browsed,
            )

        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            return AgentResult(
                final_answer=f"Error during research: {str(e)}",
                error=str(e),
            )
        finally:
            # Clean up MCP server connections
            if hasattr(self._registry, "close_all_connections"):
                try:
                    await self._registry.close_all_connections()
                except Exception as e:
                    logger.warning(f"Error closing MCP connections: {e}")


def browse_agent(
    goal: str,
    model_name: str,
    ollama_url: str,
    synthesis_model: Optional[str] = None,
    max_iterations: int = 10,
    progress_callback: Optional[Callable[[str], None]] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    context_window: int = 8192,
    # Deprecated parameters kept for backward compatibility
    min_required_pages: int = 5,
    thinking_callback: Optional[Callable] = None,
) -> AgentResult:
    """Synchronous wrapper for MCPBrowseAgent.

    Args:
        goal: Research goal/question
        model_name: Ollama model for reasoning
        ollama_url: Ollama API URL
        synthesis_model: Optional model for final synthesis
        max_iterations: Maximum reasoning iterations
        progress_callback: Optional progress callback for status updates
        stream_callback: Optional callback for streaming final answer tokens
        context_window: Context window size
        min_required_pages: (Deprecated, ignored)
        thinking_callback: (Deprecated, ignored)

    Returns:
        AgentResult with the final answer
    """
    # Deprecation warnings for unused parameters
    if min_required_pages != 5:
        warnings.warn(
            "The 'min_required_pages' parameter is deprecated and ignored. "
            "The MCP-based agent uses autonomous reasoning to determine when sufficient "
            "information has been gathered.",
            DeprecationWarning,
            stacklevel=2,
        )

    if thinking_callback is not None:
        warnings.warn(
            "The 'thinking_callback' parameter is deprecated and ignored. "
            "Use 'progress_callback' for progress updates instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    agent = MCPBrowseAgent(
        model_name=model_name,
        ollama_url=ollama_url,
        synthesis_model=synthesis_model,
        max_iterations=max_iterations,
        context_window=context_window,
        progress_callback=progress_callback,
        stream_callback=stream_callback,
    )

    # Use default asyncio policy to avoid uvloop subprocess issues
    original_policy = asyncio.get_event_loop_policy()
    try:
        # Force default policy for subprocess compatibility
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        return asyncio.run(agent.run(goal))
    finally:
        asyncio.set_event_loop_policy(original_policy)
