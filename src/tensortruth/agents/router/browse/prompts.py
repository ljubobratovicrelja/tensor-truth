"""Prompts for browse router agent."""

QUERY_GENERATION_PROMPT_TEMPLATE = """You are a search query generator for web research.

**Context:**
User's query: "{query}"

Recent conversation history:
{history_context}
{failure_context}
**Your Task:**
Generate 3 diverse, specific search queries to comprehensively research this topic.

**Requirements:**
1. Resolve pronouns (this/that/it/these) using conversation history
2. Extract custom instructions from query (e.g., "focus on SOTA", "recent only", "ignore history")
3. Generate 3 diverse angles appropriate for the query type:
   - For people: research/work, publications/projects, bio/profile
   - For technologies: overview, applications/use-cases, recent advances
   - For concepts: definition/explanation, examples/applications, current state
4. Keep queries concise (3-7 words each)
5. Make queries search-engine friendly (no questions, use keywords)

**Output Format (CRITICAL - Output ONLY valid JSON):**
{{
  "queries": [
    "query 1 here",
    "query 2 here",
    "query 3 here"
  ],
  "custom_instructions": "extracted instructions or null"
}}

**Examples:**

Example 1 - No history:
User query: "neural networks overview"
History: (No conversation history)
Output:
{{
  "queries": [
    "neural networks overview introduction",
    "neural networks architecture backpropagation",
    "neural networks 2026 recent advances"
  ],
  "custom_instructions": null
}}

Example 2 - With context:
User query: "browse more about this"
History:
user: What is backpropagation?
assistant: Backpropagation is an algorithm...
Output:
{{
  "queries": [
    "backpropagation algorithm overview",
    "backpropagation gradient descent implementation",
    "backpropagation improvements 2026"
  ],
  "custom_instructions": null
}}

Example 3 - With custom instructions:
User query: "Make me an overview of transformer methods, focus on SOTA only"
History: (No conversation history)
Output:
{{
  "queries": [
    "transformer architecture state of the art",
    "transformer SOTA methods 2026",
    "latest transformer research papers 2026"
  ],
  "custom_instructions": "focus on state-of-the-art methods only"
}}

Example 4 - Context + instructions:
User query: "about these methods. Ignore history, focus on technical details."
History:
user: Tell me about attention mechanisms
assistant: Attention mechanisms allow models...
Output:
{{
  "queries": [
    "attention mechanisms technical details",
    "attention mechanism implementation mathematics",
    "attention mechanism algorithms computation"
  ],
  "custom_instructions": "focus on technical implementation details only"
}}

Example 5 - Person/researcher query:
User query: "professor at MIT machine learning"
History: (No conversation history)
Output:
{{
  "queries": [
    "MIT machine learning professor research",
    "MIT professor publications machine learning",
    "MIT machine learning faculty profiles"
  ],
  "custom_instructions": null
}}

**Important Rules:**
1. ALWAYS resolve "this", "that", "it", "these" using conversation history
2. Extract instructions like "focus on X", "recent only", "ignore Y"
3. Make queries diverse - different angles of same topic
4. Output ONLY the JSON (no markdown, no explanation)
5. Custom instructions should summarize modifiers (or be null if none)

Now generate queries. Output ONLY the JSON:"""

ROUTER_PROMPT_TEMPLATE = """You are a workflow router for web research.
Based on the current state, decide the next action.

**Current State:**
- Search results available: {n_results}
- Pages fetched: {page_count}/{min_pages}
- Last action: {last_action}

**Available Actions:**
1. generate_queries - Generate search queries (only at start)
2. search_web - Search the web for information
3. fetch_sources - Fetch and download web pages
4. done - Complete research and synthesize answer

**Decision Rules:**
- If at start with no queries generated: MUST use "generate_queries"
- If search results = 0: MUST use "search_web"
- If search results > 0 AND pages fetched = 0: MUST use "fetch_sources"
- If pages fetched >= min pages: MUST use "done"
- NEVER repeat the last action

**Examples:**

Example 1:
State: Results=0, Pages=0/3, Last=none
Decision: {{"action": "generate_queries"}}

Example 2:
State: Results=0, Pages=0/3, Last=generate_queries
Decision: {{"action": "search_web"}}

Example 3:
State: Results=10, Pages=0/3, Last=search_web
Decision: {{"action": "fetch_sources"}}

Example 4:
State: Results=10, Pages=3/3, Last=fetch_sources
Decision: {{"action": "done"}}

**Your Turn:**
State: Results={n_results}, Pages={page_count}/{min_pages}, Last={last_action}
Decision:"""
