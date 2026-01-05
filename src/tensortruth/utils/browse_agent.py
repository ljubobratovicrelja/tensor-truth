"""Autonomous web browsing agent.

This agent iteratively searches the web, fetches pages, and makes decisions
about what to explore next based on the user's goal.

Usage:
    from tensortruth.utils.browse_agent import browse_agent

    result = browse_agent(
        goal="Find Python documentation for asyncio",
        model_name="llama3.2:3b",
        ollama_url="http://localhost:11434",
    )
    print(result.final_answer)
"""

import asyncio
import logging
import re
from typing import Callable, Optional, Tuple

import aiohttp
from llama_index.llms.ollama import Ollama

from .agent_framework import (
    AgentAction,
    AgentActionType,
    AgentState,
    BaseAgent,
    register_agent,
)
from .web_search import fetch_page_as_markdown, search_duckduckgo

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def normalize_url(url: str) -> str:
    """Normalize URL for duplicate detection.

    Handles common variations:
    - Trailing slashes
    - HTTP vs HTTPS (keeps as-is, but strips fragments)
    - URL fragments (#section)
    - Query parameters (keeps them)

    Args:
        url: Raw URL string

    Returns:
        Normalized URL string
    """
    from urllib.parse import urlparse, urlunparse

    # Parse URL
    parsed = urlparse(url.strip())

    # Rebuild without fragment, normalize path (remove trailing slash)
    # For root path ("/"), also remove it to get empty string
    path = parsed.path.rstrip("/")

    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc.lower(),  # Lowercase domain
            path,
            parsed.params,
            parsed.query,
            "",  # Remove fragment
        )
    )

    return normalized


# =============================================================================
# Prompt Templates
# =============================================================================

REASONING_PROMPT_TEMPLATE = """You are a web research agent helping answer: {goal}

=== YOUR CURRENT PROGRESS ===
Iteration: {iteration}/{max_iterations}
Searches: {num_searches} | Pages visited: {num_pages} | Failed fetches: {num_failed}

=== SEARCH RESULTS (URLs you can fetch) ===
{search_history}

=== PAGES YOU VISITED ===
{pages_summary}

=== FAILED FETCHES (Don't retry these) ===
{failed_fetches}

=== WHAT YOU LEARNED ===
{gathered_info}

=== YOUR NEXT DECISION ===
Choose ONE action:
• SEARCH - Search DuckDuckGo with new query
• FETCH_PAGE - Fetch a URL from search results above

STRICT RULES - FOLLOW THESE:
1. If searches=0 and pages=0: You MUST do SEARCH
2. If searches>0 and pages=0: You MUST do FETCH_PAGE (pick a URL from search results)
3. If pages<5: You MUST do FETCH_PAGE (need at least 5 credible sources)
4. If pages>=5: You MAY do SEARCH or FETCH_PAGE to gather more information

Your situation: Searches={num_searches}, Pages={num_pages}
→ {required_action}

Respond with EXACTLY this format:

THINKING: [explain your decision]
ACTION: [SEARCH or FETCH_PAGE]
QUERY: [if SEARCH, what to search]
URL: [if FETCH_PAGE, which URL]
REASON: [why this action]

=== EXAMPLES ===

Example 1 (first iteration, no searches):
THINKING: I have no information yet about the topic. I need to start with a search.
ACTION: SEARCH
QUERY: convolutional neural networks basics
REASON: Initial search to gather foundational information

Example 2 (have search results, choosing URL from above):
THINKING: I found several relevant pages in the search results. \
    The Wikipedia article looks comprehensive. Let me fetch it to \
    get detailed information.
ACTION: FETCH_PAGE
URL: https://en.wikipedia.org/wiki/Convolutional_neural_network
REASON: Wikipedia article provides comprehensive technical overview

Example 3 (have 5+ sources, looking for more depth):
THINKING: I have 5 sources already, but I want to find more recent research on this topic.
ACTION: SEARCH
QUERY: convolutional neural networks recent advances 2024
REASON: Seeking more current information to complement existing sources

Now respond for YOUR situation:"""

PAGE_SUMMARY_PROMPT = """Summarize the following webpage content into key \
points relevant to the user's goal.

GOAL: {goal}

PAGE CONTENT:
{content}

Extract 3-5 key points that are most relevant to answering the user's goal.
Focus on facts, data, explanations, and actionable information.

Summary:"""

FINAL_SYNTHESIS_PROMPT = """You are a research assistant synthesizing \
findings from multiple web sources.

USER'S GOAL: {goal}

SEARCH QUERIES PERFORMED:
{search_history}

PAGES ANALYZED:
{pages_with_summaries}

TASK: Provide a comprehensive answer to the user's goal based on the \
gathered information.

CRITICAL CITATION RULES - READ CAREFULLY:
1. **ALWAYS cite using markdown hyperlinks**, NEVER use plain [1] or [2] style
2. **Correct citation format**: "According to [Source Title](url), the \
key point is..."
3. **Example**: "The [Python asyncio documentation]\
(https://docs.python.org/3/library/asyncio.html) explains..."
4. **WRONG**: "According to [1], the algorithm..." ❌
5. **RIGHT**: "According to [Python Docs](https://docs.python.org), the algorithm..." ✓
6. **Link technical terms** to their definitions when sources provide them

REQUIREMENTS:
1. Start with a direct answer to the goal
2. Organize findings by topic/theme
3. Cite sources using markdown hyperlinks: [Source Title](url)
4. Include relevant details and evidence
5. Note any limitations or conflicting information
6. End with key takeaways

FORMAT:
### Answer
[Direct response to goal]

### Detailed Findings
[Organized information with inline hyperlinked citations]

### Key Takeaways
[Important points]

### Sources
[Numbered list of all sources with URLs as markdown links]

Begin your response:"""


# =============================================================================
# BrowseAgent Implementation
# =============================================================================


@register_agent("browse")
class BrowseAgent(BaseAgent):
    """Autonomous web browsing agent.

    This agent iteratively searches the web and fetches pages to answer
    user-specified goals. It uses LLM reasoning to decide when to search,
    when to fetch specific pages, and when it has gathered enough information.
    """

    def _build_search_history_summary(self, state: AgentState) -> str:
        """Build formatted search history with available URLs.

        Args:
            state: Current agent state

        Returns:
            Formatted search history string
        """
        if not state.searches_performed:
            return "None yet"

        # Only show last 3 searches to save tokens
        recent_searches = state.searches_performed[-3:]
        search_history_lines = []

        # Normalize all URLs for duplicate detection
        failed_urls = {normalize_url(url) for url, _ in state.failed_fetches}
        visited_urls = {normalize_url(url) for url, _, _ in state.pages_visited}

        for query, results in recent_searches:
            search_history_lines.append(
                f"- Query: '{query}' → {len(results)} results"
            )
            # Show top URLs (filter out failed/visited ones, deprioritize YouTube)
            if results:
                # Filter out failed URLs, already visited URLs, and prioritize non-YouTube
                available_results = [
                    r
                    for r in results
                    if normalize_url(r.get("url", "")) not in failed_urls
                    and normalize_url(r.get("url", "")) not in visited_urls
                ]
                # Sort: non-YouTube first
                prioritized = sorted(
                    available_results[:5],
                    key=lambda r: (
                        "youtube.com" in r.get("url", "").lower(),
                        r.get("url", ""),
                    ),
                )

                for i, result in enumerate(prioritized[:3], 1):
                    title = result.get("title", "No title")[:60]
                    url = result.get("url", "")
                    search_history_lines.append(f"  {i}. [{title}]({url})")

        search_history = "\n".join(search_history_lines)
        if len(state.searches_performed) > 3:
            search_history = (
                f"[{len(state.searches_performed) - 3} earlier searches]\n"
                + search_history
            )
        return search_history

    def _build_pages_summary(self, state: AgentState) -> str:
        """Build formatted summary of visited pages.

        Args:
            state: Current agent state

        Returns:
            Formatted pages summary string
        """
        if not state.pages_visited:
            return "None yet"

        # Only show last 3 pages with very short summaries
        recent_pages = state.pages_visited[-3:]
        pages_summary = "\n".join(
            [f"- [{title[:60]}]({url})" for url, title, _ in recent_pages]
        )
        if len(state.pages_visited) > 3:
            pages_summary = (
                f"[{len(state.pages_visited) - 3} earlier pages]\n" + pages_summary
            )
        return pages_summary

    def _build_failed_fetches_summary(self, state: AgentState) -> str:
        """Build formatted summary of failed page fetches.

        Args:
            state: Current agent state

        Returns:
            Formatted failed fetches summary string
        """
        if not state.failed_fetches:
            return "None"

        # Show recent failed fetches so agent knows what didn't work
        recent_failed = state.failed_fetches[-5:]  # Last 5 failures
        failed_lines = []
        for url, error in recent_failed:
            # Extract domain for cleaner display
            from urllib.parse import urlparse

            domain = urlparse(url).netloc or url
            # Truncate error message
            error_short = error[:50] if error else "Unknown error"
            failed_lines.append(f"- {domain}: {error_short}")

        failed_fetches = "\n".join(failed_lines)
        if len(state.failed_fetches) > 5:
            failed_fetches = (
                f"[{len(state.failed_fetches) - 5} earlier failures]\n"
                + failed_fetches
            )
        return failed_fetches

    def _build_gathered_info_summary(self, state: AgentState) -> str:
        """Build formatted summary of gathered information.

        Args:
            state: Current agent state

        Returns:
            Formatted gathered info summary string
        """
        if state.information_gathered:
            return "\n".join([f"- {info}" for info in state.information_gathered])
        return "No specific insights extracted yet"

    def _determine_required_action(self, state: AgentState) -> str:
        """Determine what action the agent must/should take based on current state.

        Args:
            state: Current agent state

        Returns:
            Guidance string for the agent's next action
        """
        num_searches = len(state.searches_performed)
        num_pages = len(state.pages_visited)
        min_required_pages = 5  # Target 5 credible sources

        if num_searches == 0:
            return "You MUST do SEARCH (you have no data yet)"
        elif num_pages == 0:
            return (
                "You MUST do FETCH_PAGE (you have search results "
                "but haven't fetched any pages)"
            )
        elif num_pages < min_required_pages:
            return (
                "You MUST do FETCH_PAGE (you need "
                f"at least {min_required_pages} sources, "
                f"currently have {num_pages})"
            )
        else:
            return (
                "You MAY do SEARCH or FETCH_PAGE to gather more "
                f"information (you have {num_pages} sources)"
            )

    async def reason_next_action(
        self,
        state: AgentState,
        model_name: str,
        ollama_url: str,
        context_window: int,
    ) -> Tuple[str, AgentAction]:
        """Use LLM to reason about next action."""
        # Build all prompt sections
        search_history = self._build_search_history_summary(state)
        pages_summary = self._build_pages_summary(state)
        failed_fetches = self._build_failed_fetches_summary(state)
        gathered_info = self._build_gathered_info_summary(state)
        required_action = self._determine_required_action(state)

        # Build prompt
        prompt = REASONING_PROMPT_TEMPLATE.format(
            goal=state.goal,
            iteration=state.current_iteration + 1,
            max_iterations=state.max_iterations,
            num_searches=len(state.searches_performed),
            num_pages=len(state.pages_visited),
            num_failed=len(state.failed_fetches),
            search_history=search_history,
            pages_summary=pages_summary,
            failed_fetches=failed_fetches,
            gathered_info=gathered_info,
            required_action=required_action,
        )

        # Truncate prompt if too long (use 50% of context for reasoning)
        max_prompt_chars = int(context_window * 0.5 * 4)  # 4 chars per token
        if len(prompt) > max_prompt_chars:
            logger.warning(
                f"Prompt too long ({len(prompt)} chars), truncating to {max_prompt_chars}"
            )
            prompt = prompt[:max_prompt_chars] + "\n\n[Previous content truncated...]"

        # Call LLM
        llm = Ollama(
            model=model_name,
            base_url=ollama_url,
            request_timeout=60.0,
            temperature=0.2,
            context_window=context_window,
            num_ctx=context_window,
        )

        try:
            response = await llm.acomplete(prompt)
            thinking_text = response.text.strip()

            # Parse action from response
            action = self._parse_action(thinking_text, state)

            return thinking_text, action
        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
            # Fallback: conclude if we have any data, otherwise try one more search
            if state.searches_performed or state.pages_visited:
                return (
                    f"LLM timeout/error: {str(e)}. Concluding with available data.",
                    AgentAction(
                        type=AgentActionType.CONCLUDE,
                        reasoning=f"LLM error: {str(e)}",
                    ),
                )
            else:
                # No data yet, try a basic search
                return (
                    f"LLM timeout/error on first iteration: {str(e)}. Attempting fallback search.",
                    AgentAction(
                        type=AgentActionType.SEARCH,
                        query=state.goal,
                        reasoning="Fallback search due to LLM error",
                    ),
                )

    def _parse_action(self, llm_response: str, state: AgentState) -> AgentAction:
        """Parse LLM response into structured action.

        Args:
            llm_response: Raw LLM response text
            state: Current agent state (for fallback logic)

        Returns:
            Parsed AgentAction (falls back to CONCLUDE on parse errors)
        """
        try:
            # Extract ACTION line (case-insensitive)
            action_match = re.search(
                r"ACTION:\s*(SEARCH|FETCH_PAGE|CONCLUDE)",
                llm_response,
                re.IGNORECASE,
            )

            if not action_match:
                logger.warning(
                    f"No ACTION found in LLM response, defaulting to CONCLUDE. "
                    f"LLM response was:\n{llm_response[:500]}"
                )
                return AgentAction(
                    type=AgentActionType.CONCLUDE,
                    reasoning="Could not parse action from response",
                )

            action_type = AgentActionType[action_match.group(1).upper()]

            # Extract REASON (case-insensitive)
            reason_match = re.search(r"REASON:\s*(.+)", llm_response, re.IGNORECASE)
            reasoning = reason_match.group(1).strip() if reason_match else None

            # Handle SEARCH action
            if action_type == AgentActionType.SEARCH:
                query_match = re.search(r"QUERY:\s*(.+)", llm_response, re.IGNORECASE)
                if not query_match:
                    logger.warning(
                        "SEARCH action missing QUERY, defaulting to CONCLUDE"
                    )
                    return AgentAction(
                        type=AgentActionType.CONCLUDE,
                        reasoning="SEARCH action without query",
                    )

                query = query_match.group(1).strip()
                # Remove quotes if present
                query = query.strip("\"'")

                return AgentAction(
                    type=AgentActionType.SEARCH, query=query, reasoning=reasoning
                )

            # Handle FETCH_PAGE action
            elif action_type == AgentActionType.FETCH_PAGE:
                url_match = re.search(r"URL:\s*(.+)", llm_response, re.IGNORECASE)
                if not url_match:
                    logger.warning(
                        "FETCH_PAGE action missing URL, defaulting to CONCLUDE"
                    )
                    return AgentAction(
                        type=AgentActionType.CONCLUDE,
                        reasoning="FETCH_PAGE action without URL",
                    )

                url = url_match.group(1).strip()
                # Normalize URL to prevent duplicate fetches
                url = normalize_url(url)
                return AgentAction(
                    type=AgentActionType.FETCH_PAGE, url=url, reasoning=reasoning
                )

            # Handle CONCLUDE action
            else:
                return AgentAction(type=AgentActionType.CONCLUDE, reasoning=reasoning)

        except Exception as e:
            logger.error(f"Failed to parse action: {e}", exc_info=True)
            return AgentAction(
                type=AgentActionType.CONCLUDE,
                reasoning=f"Parse error: {str(e)}",
            )

    async def execute_action(
        self,
        action: AgentAction,
        state: AgentState,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Execute an action and update state."""
        if action.type == AgentActionType.SEARCH:
            await self._execute_search(action, state, progress_callback)
        elif action.type == AgentActionType.FETCH_PAGE:
            await self._execute_fetch_page(action, state, progress_callback)

    async def _execute_search(
        self,
        action: AgentAction,
        state: AgentState,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Execute a web search."""
        # Check if query already searched (case-insensitive, normalized)
        normalized_query = action.query.lower().strip()
        previous_queries = {q.lower().strip() for q, _ in state.searches_performed}

        if normalized_query in previous_queries:
            logger.warning(f"Skipping duplicate search query: {action.query}")
            if progress_callback:
                progress_callback(f"⚠️ **Skipping duplicate search:** {action.query}")
            return

        if progress_callback:
            progress_callback(f"🔎 **Searching for:** {action.query}")

        results = await search_duckduckgo(
            action.query, max_results=10, progress_callback=None
        )

        state.searches_performed.append((action.query, results))

        # Don't report number of results found - just that search completed
        if not results and progress_callback:
            progress_callback("⚠️ No search results found")

    async def _execute_fetch_page(
        self,
        action: AgentAction,
        state: AgentState,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Fetch and summarize a web page."""
        # Check if URL already visited (defensive check)
        normalized_url = normalize_url(action.url)
        visited_urls = {normalize_url(url) for url, _, _ in state.pages_visited}

        if normalized_url in visited_urls:
            logger.warning(f"Skipping duplicate URL: {action.url}")
            if progress_callback:
                progress_callback(f"⚠️ **Skipping duplicate URL:** {action.url}")
            return

        # Create clickable link with truncated display text
        # Truncate URL for display if too long (keep as clickable link)
        display_url = action.url if len(action.url) <= 80 else action.url[:77] + "..."

        if progress_callback:
            progress_callback(
                f"📄 **Fetching page from:** [{display_url}]({action.url})"
            )

        async with aiohttp.ClientSession() as session:
            content, status, error_msg = await fetch_page_as_markdown(
                action.url, session, timeout=10
            )

            if status == "success" and content:
                # Summarize page content to preserve context budget
                # For now, just truncate to first 2000 chars
                # TODO: Use LLM summarization for better quality
                summary = content[:2000]
                if len(content) > 2000:
                    summary += "\n\n[Content truncated...]"

                # Extract title from action or URL
                title = action.title or action.url.split("/")[-1]

                state.pages_visited.append((action.url, title, summary))

                if progress_callback:
                    progress_callback(
                        f"✅ **Successfully retrieved:** [{display_url}]({action.url})"
                    )
            else:
                state.failed_fetches.append((action.url, error_msg or status))
                if progress_callback:
                    progress_callback(
                        f"❌ **Failed to retrieve:** [{display_url}]({action.url})"
                    )

    async def synthesize_final_answer(
        self,
        state: AgentState,
        model_name: str,
        ollama_url: str,
        context_window: int,
        progress_callback: Optional[Callable[[str], None]] = None,
        synthesis_model: Optional[str] = None,
    ) -> str:
        """Synthesize final answer from agent state.

        Args:
            state: Agent state with gathered information
            model_name: Model for synthesis (fallback if synthesis_model not provided)
            ollama_url: Ollama API URL
            context_window: Context window size
            progress_callback: Optional progress callback
            synthesis_model: Optional separate model for final synthesis (uses model_name if None)
        """
        # Handle case where agent terminated without gathering info
        if not state.searches_performed and not state.pages_visited:
            return (
                f"❌ **Unable to gather information**\n\n"
                f"Agent terminated ({state.termination_reason}) before "
                f"performing any searches or fetching pages."
            )

        # Build search history
        search_history = "\n".join(
            [f"{i+1}. {query}" for i, (query, _) in enumerate(state.searches_performed)]
        )

        # Build pages summary with content
        pages_with_summaries = []
        for i, (url, title, summary) in enumerate(state.pages_visited, 1):
            pages_with_summaries.append(
                f"### Source {i}: [{title}]({url})\n\n{summary}\n\n---\n"
            )

        pages_text = "\n".join(pages_with_summaries)

        # If no pages were fetched, use search snippets instead
        if not pages_text and state.searches_performed:
            snippets = []
            for query, results in state.searches_performed:
                for i, result in enumerate(results[:5], 1):  # Top 5 per search
                    snippets.append(
                        f"### Result {i} from '{query}'\n"
                        f"**[{result['title']}]({result['url']})**\n\n"
                        f"{result['snippet']}\n\n---\n"
                    )
            pages_text = "\n".join(snippets)

        # Build prompt
        prompt = FINAL_SYNTHESIS_PROMPT.format(
            goal=state.goal,
            search_history=search_history or "None",
            pages_with_summaries=pages_text or "No content available",
        )

        # Truncate if needed (use 60% of context for synthesis input)
        max_input_chars = int(context_window * 0.6 * 4)
        if len(prompt) > max_input_chars:
            logger.warning(
                f"Synthesis prompt too long ({len(prompt)} chars), truncating"
            )
            prompt = prompt[:max_input_chars] + "\n\n[Content truncated for length...]"

        # Use synthesis model if provided, otherwise use reasoning model
        final_model = synthesis_model or model_name
        if synthesis_model:
            logger.info(f"Using synthesis model: {synthesis_model}")

        # Call LLM for final synthesis
        llm = Ollama(
            model=final_model,
            base_url=ollama_url,
            request_timeout=120.0,
            temperature=0.3,  # Slightly higher for synthesis
            context_window=context_window,
            num_ctx=context_window,
        )

        try:
            response = await llm.acomplete(prompt)
            summary = response.text.strip()

            # Add termination metadata
            metadata_lines = []
            if state.termination_reason:
                reason_display = {
                    "goal_satisfied": "Agent determined goal was satisfied",
                    "max_iterations": f"Reached maximum iterations ({state.max_iterations})",
                    "timeout": "Execution timeout reached",
                    "error": "Error occurred during execution",
                }.get(state.termination_reason, state.termination_reason)

                metadata_lines.append(f"\n\n---\n*{reason_display}*")
                metadata_lines.append(
                    f"*Searches: {len(state.searches_performed)} | "
                    f"Pages visited: {len(state.pages_visited)} | "
                    f"Iterations: {state.current_iteration}*"
                )

            return summary + "".join(metadata_lines)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            return (
                f"❌ **Failed to synthesize answer:** {str(e)}\n\n"
                f"Agent gathered {len(state.searches_performed)} searches and "
                f"{len(state.pages_visited)} pages, but could not generate summary."
            )


# =============================================================================
# Public API
# =============================================================================


def browse_agent(
    goal: str,
    model_name: str,
    ollama_url: str,
    max_iterations: int = 10,
    thinking_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    context_window: int = 16384,
    synthesis_model: Optional[str] = None,
) -> AgentState:
    """Execute the autonomous browsing agent (sync wrapper).

    Args:
        goal: User's goal to achieve
        model_name: Ollama model for reasoning (MUST be 7b+, e.g., "llama3.1:8b")
                   Note: 3b models insufficient for structured reasoning
        ollama_url: Ollama API base URL
        max_iterations: Maximum iterations allowed (default: 10, target 5 \
            sources with room for failures)
        thinking_callback: Callback for streaming thinking updates
        progress_callback: Callback for progress updates
        context_window: Model context window size (default: 16384, capped at 8k in practice)
        synthesis_model: Optional model for final synthesis (e.g., main chat model for quality)

    Returns:
        AgentState with final answer and execution history

    Example:
        # Use fast model for reasoning, quality model for synthesis
        result = browse_agent(
            goal="Find Python asyncio documentation",
            model_name="llama3.2:latest",  # Fast for decisions
            synthesis_model="deepseek-r1:8b",  # Quality for final answer
            ollama_url="http://localhost:11434",
        )
        print(result.final_answer)
    """
    agent = BrowseAgent(name="browse", description=BrowseAgent.__doc__)

    # Run async agent in sync context
    return asyncio.run(
        agent.run(
            goal=goal,
            model_name=model_name,
            ollama_url=ollama_url,
            max_iterations=max_iterations,
            context_window=context_window,
            thinking_callback=thinking_callback,
            progress_callback=progress_callback,
            synthesis_model=synthesis_model,
        )
    )
