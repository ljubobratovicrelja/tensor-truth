#!/usr/bin/env python3
"""
ROUTER-BASED BROWSE AGENT - Real Implementation
==============================================

This version uses:
- REAL DuckDuckGo searches (via ddgs library)
- REAL page fetching (HTTP requests with readability)
- REAL synthesis (llama3.1:8b via Ollama)

No mocks - this is the actual working implementation.

Usage:
    pip install ddgs httpx readability-lxml lxml_html_clean
    python test_router_browse_agent_real.py
"""

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import httpx
import requests
from ddgs import DDGS
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, TextNode
from readability import Document

# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================


class WorkflowPhase(Enum):
    """Phases of the browse workflow."""

    INITIAL = "initial"
    SEARCHED = "searched"
    FETCHED = "fetched"
    COMPLETE = "complete"


@dataclass
class BrowseState:
    """
    Explicit state tracking for browse workflow.

    New fields for retry logic:
    - fetch_iterations: Number of fetch attempts (max 3)
    - next_url_index: Index in search_results for next batch of URLs
    - min_pages_required: Now 5 (increased from 3)
    - total_content_chars: Cumulative content size across all fetched pages
    - max_content_chars: Max content before overflow (default 25000)
    - content_overflow: Flag indicating we stopped due to context overflow
    """

    query: str
    phase: WorkflowPhase = WorkflowPhase.INITIAL
    search_results: Optional[List[Dict]] = None
    pages: Optional[List[Dict]] = None
    min_pages_required: int = 5  # Changed from 3 to 5
    search_depth: str = "thorough"
    actions_taken: List[str] = field(default_factory=list)
    fetch_iterations: int = 0  # Track how many times we've fetched
    next_url_index: int = 0  # Track which URL to fetch next
    max_fetch_iterations: int = 3  # Max number of fetch attempts
    total_content_chars: int = 0  # Cumulative content size
    max_content_chars: int = 25000  # Max before overflow (~safe for llama3.1:8b)
    content_overflow: bool = False  # True if stopped due to overflow

    @property
    def is_complete(self) -> bool:
        return self.phase == WorkflowPhase.COMPLETE

    @property
    def has_search_results(self) -> bool:
        return self.search_results is not None and len(self.search_results) > 0

    @property
    def has_pages(self) -> bool:
        return self.pages is not None and len(self.pages) > 0

    @property
    def page_count(self) -> int:
        return len(self.pages) if self.pages else 0

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "phase": self.phase.value,
            "search_depth": self.search_depth,
            "actions_taken": self.actions_taken,
            "has_search_results": self.has_search_results,
            "search_result_count": (
                len(self.search_results) if self.search_results else 0
            ),
            "has_pages": self.has_pages,
            "page_count": self.page_count,
            "fetch_iterations": self.fetch_iterations,
            "next_url_index": self.next_url_index,
            "total_content_chars": self.total_content_chars,
            "max_content_chars": self.max_content_chars,
            "content_overflow": self.content_overflow,
        }


# ==============================================================================
# ROUTER (same as POC)
# ==============================================================================


class BrowseRouter:
    """Router using llama3.2:3b with structured output."""

    def __init__(
        self, model: str = "llama3.2:3b", ollama_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.ollama_url = ollama_url

    def _build_router_prompt(self, state: BrowseState) -> str:
        """Build routing prompt with structured output."""
        history_str = (
            " → ".join(state.actions_taken) if state.actions_taken else "START"
        )
        last_action = state.actions_taken[-1] if state.actions_taken else "none"

        return f"""You are a workflow router. Choose the next action based on the current state.

Respond ONLY with valid JSON in this format: {{"action": "search_web"}} or {{"action": "fetch_pages_batch"}} or {{"action": "done"}}

Respond only as shown, with no additional discursive or explanatory text.

EXAMPLES:

State: Results=0, Pages=0/3, Last=none
Output: {{"action": "search_web"}}

State: Results=5, Pages=0/3, Last=search_web
Output: {{"action": "fetch_pages_batch"}}

State: Results=5, Pages=3/3, Last=fetch_pages_batch
Output: {{"action": "done"}}

State: Results=4, Pages=1/1, Last=fetch_pages_batch
Output: {{"action": "done"}}

CRITICAL RULES:
- If Results>0 and Pages=0: MUST use fetch_pages_batch
- If Pages >= required: MUST use done
- NEVER repeat the last action
- NEVER search_web twice

YOUR TURN:

State: Results={len(state.search_results) if state.search_results else 0}, Pages={state.page_count}/{state.min_pages_required}, Last={last_action}
Output:"""

    async def route(self, state: BrowseState) -> str:
        """Make routing decision with structured output."""
        prompt = self._build_router_prompt(state)

        print("\n" + "=" * 80)
        print("ROUTER DECISION")
        print("=" * 80)
        print(f"State: {json.dumps(state.to_dict(), indent=2)}")
        print(f"\nRouting with {self.model}...")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search_web", "fetch_pages_batch", "done"],
                        "description": "The next action to take",
                    }
                },
                "required": ["action"],
            },
            "options": {
                "temperature": 0.0,
                "num_predict": 50,
            },
        }

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=30
            )
            result = resp.json()
            response_text = result.get("response", "").strip()

            print(f"Router raw response: '{response_text}'")

            # Parse JSON response
            try:
                response_json = json.loads(response_text)
                action = response_json.get("action")

                if action in ["search_web", "fetch_pages_batch", "done"]:
                    print(f"✅ Parsed action from LLM: {action}")
                    return action
                else:
                    print(f"⚠️  Invalid action: '{action}'")
                    return self._deterministic_route(state)

            except json.JSONDecodeError:
                print("⚠️  Response not valid JSON")
                return self._deterministic_route(state)

        except Exception as e:
            print(f"❌ Router error: {e}")
            return self._deterministic_route(state)

    def _deterministic_route(self, state: BrowseState) -> str:
        """
        Deterministic fallback routing with retry logic.

        New logic:
        - If < 5 pages and < 3 fetch iterations and more URLs available → retry fetch
        - Stop if content overflow OR >= 5 pages OR hit max iterations
        """
        if not state.has_search_results:
            return "search_web"

        if not state.has_pages:
            # First fetch attempt
            return "fetch_pages_batch"

        # Check stopping conditions
        if state.content_overflow:
            # Hit context window limit
            return "done"

        # Have some pages - check if we need more
        if state.page_count < state.min_pages_required:
            # Need more pages
            if state.fetch_iterations < state.max_fetch_iterations:
                # Haven't hit max iterations yet
                if state.next_url_index < len(state.search_results):
                    # Still have URLs to try
                    return "fetch_pages_batch"

        # Either have enough pages, hit max iterations, or no more URLs
        return "done"


# ==============================================================================
# EXECUTOR - REAL IMPLEMENTATIONS
# ==============================================================================


class BrowseExecutor:
    """Executes actions with REAL DuckDuckGo and page fetching."""

    async def execute_search_web(self, state: BrowseState) -> BrowseState:
        """
        REAL DuckDuckGo search implementation.

        Generates 3 diverse queries and searches DuckDuckGo.
        """
        print("\n" + "=" * 80)
        print("EXECUTING: search_web (REAL DuckDuckGo)")
        print("=" * 80)

        # Generate 3 diverse queries
        queries = [
            f"{state.query} overview",
            f"{state.query} technical details",
            f"{state.query} recent 2026",
        ]

        print(f"Queries: {queries}")
        print("Searching DuckDuckGo...")

        all_results = []
        seen_urls = set()

        try:
            with DDGS() as ddgs:
                for query in queries:
                    print(f"  Searching: {query}")
                    results = list(ddgs.text(query, max_results=10))

                    # Deduplicate by URL
                    for result in results:
                        url = result.get("href", result.get("url", ""))
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(
                                {
                                    "url": url,
                                    "title": result.get("title", ""),
                                    "snippet": result.get("body", ""),
                                }
                            )

                    print(f"    Found {len(results)} results")

            print(f"✅ Got {len(all_results)} unique search results")

            # RERANK results using cross-encoder before fetching
            if len(all_results) > 0:
                print(f"\n🔄 Reranking {len(all_results)} results...")
                print("   Using: BAAI/bge-reranker-v2-m3")

                # Convert to LlamaIndex nodes
                nodes = []
                for result in all_results:
                    text = f"{result.get('title', '')}. {result.get('snippet', '')}"
                    node = TextNode(text=text)
                    node.metadata = result
                    nodes.append(NodeWithScore(node=node, score=1.0))

                # Rerank
                reranker = SentenceTransformerRerank(
                    model="BAAI/bge-reranker-v2-m3",
                    top_n=min(20, len(all_results))  # Keep top 20 or all if less
                )
                reranked_nodes = reranker.postprocess_nodes(
                    nodes, query_str=state.query
                )

                # Convert back to search result format with scores
                all_results = []
                for node_with_score in reranked_nodes:
                    result = node_with_score.node.metadata.copy()
                    result['relevance_score'] = node_with_score.score
                    all_results.append(result)

                print(f"   ✅ Reranked! Top score: {all_results[0]['relevance_score']:.3f}")
                print(f"   Top result: {all_results[0]['title'][:60]}")

            # Update state with reranked results
            state.search_results = all_results  # Already limited to top 20 by reranker
            state.phase = WorkflowPhase.SEARCHED

            return state

        except Exception as e:
            print(f"❌ Search error: {e}")
            # Return empty results rather than failing
            state.search_results = []
            state.phase = WorkflowPhase.SEARCHED
            return state

    async def execute_fetch_pages_batch(self, state: BrowseState) -> BrowseState:
        """
        REAL page fetching with HTTP requests and readability extraction.

        NEW: Fetches pages ONE BY ONE and stops when:
        1. Max pages reached (min_pages_required)
        2. Context window would overflow (total_content_chars > max_content_chars)
        3. No more URLs available
        """
        print("\n" + "=" * 80)
        print(f"EXECUTING: fetch_pages_batch (Iteration {state.fetch_iterations + 1}/3)")
        print("=" * 80)

        # Increment fetch iteration
        state.fetch_iterations += 1

        # Calculate how many pages we still need
        current_count = len(state.pages) if state.pages else 0
        current_content = state.total_content_chars

        print(f"Current pages: {current_count}/{state.min_pages_required}")
        print(f"Current content: {current_content}/{state.max_content_chars} chars")
        print(f"Content headroom: {state.max_content_chars - current_content} chars")

        # Initialize pages list if needed
        if state.pages is None:
            state.pages = []

        # Fetch pages one by one until we hit a stopping condition
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            pages_fetched_this_iteration = 0

            while state.next_url_index < len(state.search_results):
                # Check stopping conditions BEFORE fetching
                if current_count >= state.min_pages_required:
                    print(f"✅ Reached min_pages_required ({state.min_pages_required})")
                    break

                # Get next URL
                url = state.search_results[state.next_url_index]["url"]
                state.next_url_index += 1

                try:
                    print(f"  [{current_count + 1}] Fetching {url[:80]}")
                    response = await client.get(url)

                    if response.status_code == 200:
                        # Extract readable content with readability
                        doc = Document(response.text)
                        title = doc.title()
                        content = doc.summary()  # HTML content

                        # Convert HTML to plain text (simple version)
                        import re
                        from html import unescape

                        text_content = re.sub("<[^<]+?>", "", content)
                        text_content = unescape(text_content)
                        text_content = re.sub(r"\s+", " ", text_content).strip()

                        # Truncate to reasonable length
                        max_chars = 5000
                        if len(text_content) > max_chars:
                            text_content = text_content[:max_chars] + "..."

                        page_length = len(text_content)

                        # CHECK: Would this overflow the context window?
                        if current_content + page_length > state.max_content_chars:
                            print(f"    ⚠️  OVERFLOW: Adding this page ({page_length} chars) "
                                  f"would exceed limit")
                            print(f"       Current: {current_content} + New: {page_length} "
                                  f"> Max: {state.max_content_chars}")
                            state.content_overflow = True
                            break

                        # Safe to add this page
                        state.pages.append(
                            {
                                "url": url,
                                "title": title,
                                "status": "success",
                                "content": text_content,
                                "length": page_length,
                            }
                        )
                        state.total_content_chars += page_length
                        current_count += 1
                        current_content += page_length
                        pages_fetched_this_iteration += 1

                        print(f"    ✅ Success ({page_length} chars)")
                        print(f"       Total content now: {current_content}/{state.max_content_chars} chars")
                    else:
                        print(f"    ❌ HTTP {response.status_code} - skipping")

                except Exception as e:
                    print(f"    ❌ Error: {e} - skipping")

        print(f"\n✅ Fetched {pages_fetched_this_iteration} pages in this iteration")
        print(f"📊 Total pages now: {len(state.pages)}/{state.min_pages_required}")
        print(f"📊 Total content: {state.total_content_chars}/{state.max_content_chars} chars")

        if state.content_overflow:
            print(f"⚠️  CONTEXT OVERFLOW - stopped fetching to prevent overflow")

        # Set phase to FETCHED (router will decide if we need more)
        state.phase = WorkflowPhase.FETCHED

        return state

    async def execute(self, action: str, state: BrowseState) -> BrowseState:
        """Execute the routed action and return updated state."""
        state.actions_taken.append(action)

        if action == "search_web":
            return await self.execute_search_web(state)
        elif action == "fetch_pages_batch":
            return await self.execute_fetch_pages_batch(state)
        elif action == "done":
            state.phase = WorkflowPhase.COMPLETE
            return state
        else:
            raise ValueError(f"Unknown action: {action}")


# ==============================================================================
# SYNTHESIS - REAL LLM
# ==============================================================================


class BrowseSynthesizer:
    """Synthesizes final answer using llama3.1:8b."""

    def __init__(
        self, model: str = "llama3.1:8b", ollama_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.ollama_url = ollama_url

    async def synthesize(
        self, query: str, pages: List[Dict], content_overflow: bool = False
    ) -> str:
        """
        REAL synthesis with llama3.1:8b.

        Takes fetched pages and synthesizes a comprehensive answer.

        Args:
            query: User's question
            pages: List of fetched pages
            content_overflow: True if we stopped fetching due to context overflow
        """
        print("\n" + "=" * 80)
        print(f"SYNTHESIS (REAL {self.model})")
        print("=" * 80)
        print(f"Input: {len(pages)} pages")
        print(f"Total content: {sum(p['length'] for p in pages)} characters")
        if content_overflow:
            print("⚠️  OVERFLOW WARNING: Some sources were skipped due to context limits")

        # Build context from pages
        context_parts = []
        for i, page in enumerate(pages, 1):
            context_parts.append(f"""
Source {i}: {page['title']}
URL: {page['url']}
Content: {page['content'][:2000]}...
""")

        context = "\n\n".join(context_parts)

        # Build overflow warning for prompt
        overflow_warning = ""
        if content_overflow:
            overflow_warning = """
⚠️  IMPORTANT: Due to context window limitations, only the first {len(pages)} sources
were included. Additional relevant sources were found but could not be processed.
Please review the source list and visit additional URLs if you need more comprehensive
information.
"""

        # Build synthesis prompt
        prompt = f"""You are a research assistant. Based on the web pages below, provide a comprehensive answer to the user's question.

USER QUESTION:
{query}

WEB PAGES:
{context}

INSTRUCTIONS:
1. Synthesize information from all sources
2. Provide a clear, comprehensive answer
3. Cite sources by number [Source 1], [Source 2], etc.
4. Be factual and accurate
5. If information is insufficient, state that
{overflow_warning}

YOUR ANSWER:"""

        print("Synthesizing with LLM...")

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1000,
                },
            }

            resp = requests.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=120
            )
            result = resp.json()
            answer = result.get("response", "").strip()

            # Prepend overflow warning to answer if applicable
            if content_overflow:
                overflow_msg = (
                    f"⚠️  CONTEXT OVERFLOW WARNING: This answer is based on "
                    f"{len(pages)} sources. Additional relevant sources were found "
                    f"but could not be included due to context window limitations. "
                    f"Review the source list for additional information.\n\n"
                )
                answer = overflow_msg + answer

            print(f"✅ Synthesis complete ({len(answer)} chars)")
            return answer

        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            return f"Error synthesizing answer: {e}"


# ==============================================================================
# MAIN AGENT
# ==============================================================================


class RouterBrowseAgent:
    """Main agent orchestrating router + executor + synthesizer."""

    def __init__(
        self,
        router_model: str = "llama3.2:3b",
        synthesis_model: str = "llama3.1:8b",
        min_pages_required: int = 5,  # Changed from 3 to 5
        search_depth: str = "thorough",
    ):
        self.router = BrowseRouter(model=router_model)
        self.executor = BrowseExecutor()
        self.synthesizer = BrowseSynthesizer(model=synthesis_model)
        self.min_pages_required = min_pages_required
        self.search_depth = search_depth

    async def run(self, query: str) -> Dict:
        """Run the browse workflow with REAL search, fetch, and synthesis."""
        print("\n" + "=" * 80)
        print("ROUTER BROWSE AGENT - Real Implementation")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"Search depth: {self.search_depth}")
        print(f"Min pages required: {self.min_pages_required}")

        # Initialize state
        state = BrowseState(
            query=query,
            min_pages_required=self.min_pages_required,
            search_depth=self.search_depth,
        )

        # Main loop: route → execute → update state
        iteration = 0
        max_iterations = 5

        while not state.is_complete and iteration < max_iterations:
            iteration += 1
            print(f"\n{'=' * 80}")
            print(f"ITERATION {iteration}")
            print("=" * 80)

            # 1. Router decides next action
            action = await self.router.route(state)

            # 2. Executor performs action
            state = await self.executor.execute(action, state)

            # 3. Check if done
            if action == "done":
                break

        # Synthesis phase
        if state.pages and len(state.pages) > 0:
            answer = await self.synthesizer.synthesize(
                state.query, state.pages, content_overflow=state.content_overflow
            )
        else:
            answer = "No pages were successfully fetched. Unable to provide an answer."

        return {
            "answer": answer,
            "sources": state.pages,
            "iterations": iteration,
            "final_state": state.to_dict(),
        }


# ==============================================================================
# TEST EXECUTION
# ==============================================================================


async def main(query: str = None):
    """Run the test with real DuckDuckGo, fetching, and synthesis.

    Args:
        query: User's search query. If None, uses default query.
    """
    # Use default query if none provided
    if query is None:
        query = "What are convolutional neural networks?"

    print("\n" + "=" * 80)
    print("ROUTER BROWSE AGENT - REAL IMPLEMENTATION TEST")
    print("=" * 80)
    print("\nFeatures:")
    print("  - REAL DuckDuckGo searches (3 diverse queries)")
    print("  - RERANKING with BAAI/bge-reranker-v2-m3 cross-encoder")
    print("  - ONE-BY-ONE fetching with context overflow protection")
    print("  - Stops when: 5 pages OR content overflow OR 3 iterations")
    print("  - REAL HTTP page fetching with readability extraction")
    print("  - REAL llama3.1:8b synthesis with overflow warnings")

    # Test thorough search with 5 pages minimum
    agent = RouterBrowseAgent(
        router_model="llama3.2:3b",
        synthesis_model="llama3.1:8b",
        min_pages_required=5,  # Changed from 3 to 5
        search_depth="thorough",
    )

    result = await agent.run(query)

    # Display results
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"\nIterations: {result['iterations']}")
    print(f"Sources: {len(result['sources'])} pages")
    print(f"\nFinal state: {json.dumps(result['final_state'], indent=2)}")
    print(f"\n{'=' * 80}")
    print("ANSWER:")
    print("=" * 80)
    print(result["answer"])
    print(f"\n{'=' * 80}")
    print("SOURCES:")
    print("=" * 80)
    for i, source in enumerate(result["sources"], 1):
        print(f"{i}. {source['title']}")
        print(f"   {source['url']}")
        print(f"   ({source['length']} chars)")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    # Check for command-line query argument
    if len(sys.argv) > 1:
        # Join all arguments after script name as the query
        user_query = " ".join(sys.argv[1:])
        asyncio.run(main(user_query))
    else:
        # Use default query
        asyncio.run(main())
