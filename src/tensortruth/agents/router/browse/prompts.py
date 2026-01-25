"""Prompts for browse router agent."""

ROUTER_PROMPT_TEMPLATE = """You are a workflow router for web research.
Based on the current state, decide the next action.

**Current State:**
- Search results available: {n_results}
- Pages fetched: {page_count}/{min_pages}
- Last action: {last_action}

**Available Actions:**
1. search_web - Search the web for information
2. fetch_pages_batch - Fetch and download web pages
3. done - Complete research and synthesize answer

**Decision Rules:**
- If search results = 0: MUST use "search_web"
- If search results > 0 AND pages fetched = 0: MUST use "fetch_pages_batch"
- If pages fetched >= min pages: MUST use "done"
- NEVER repeat the last action

**Examples:**

Example 1:
State: Results=0, Pages=0/3, Last=none
Decision: {{"action": "search_web"}}

Example 2:
State: Results=10, Pages=0/3, Last=search_web
Decision: {{"action": "fetch_pages_batch"}}

Example 3:
State: Results=10, Pages=3/3, Last=fetch_pages_batch
Decision: {{"action": "done"}}

**Your Turn:**
State: Results={n_results}, Pages={page_count}/{min_pages}, Last={last_action}
Decision:"""
