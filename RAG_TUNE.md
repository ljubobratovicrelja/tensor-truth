# RAG Tuning and Debug Context Feature

## Overview

This document explains recent work on the RAG (Retrieval-Augmented Generation) pipeline debugging and query condensing mechanism. It serves as a handoff for further tuning and optimization work.

## What Was Implemented

### 1. `--debug-context` CLI Flag

Added a comprehensive debugging feature to visualize exactly what the LLM receives during RAG inference.

**Usage:**
```bash
tensor-truth --debug-context
```

**What It Shows:**

For RAG mode, displays 5 sections:
1. **Retrieval Query** - User input vs. condensed query (after LLM processing)
2. **Retrieval Summary** - Nodes retrieved, scores, confidence status
3. **Context String** - Raw text from retrieved documents (exactly what's sent to LLM)
4. **Current Prompt** - Formatted prompt for this turn (context + query)
5. **Complete Conversation** - Full conversation including chat history + current prompt

For Simple LLM mode (no RAG), displays:
1. **Model Configuration** - Model name, temperature, max_tokens, device
2. **Chat History** - Previous conversation messages
3. **Current Prompt** - User's input
4. **System Prompt** - If available

### 2. Architecture

**Environment Variable Approach:**
- `cli.py` - Parses `--debug-context` flag, sets `TENSOR_TRUTH_DEBUG_CONTEXT=1`
- `app_state.py` - Reads env var, initializes `st.session_state.debug_context`
- `chat_handler.py` - Captures debug data during inference
- `rendering_debug.py` - Live rendering during streaming (NEW FILE)
- `rendering.py` - Renders stored debug data from message history

**Key Design:**
- All debug info shown in **plain text** (not HTML) - user can copy actual prompts/context
- Debug data stored in message history (`debug_data` field) so it persists after streaming
- `build_chat_history()` intentionally excludes metadata - only `content` and `role` pass to LLM
- No code duplication - separate functions for live vs. stored rendering, but consistent structure

### 3. Files Modified

```
src/tensortruth/cli.py                           # Flag parsing
src/tensortruth/app_utils/app_state.py          # Session state init
src/tensortruth/app_utils/chat_handler.py       # Debug data capture
src/tensortruth/app_utils/rendering_debug.py    # Live debug rendering (NEW)
src/tensortruth/app_utils/rendering.py          # Stored debug rendering
src/tensortruth/app_utils/chat_utils.py         # Documentation added
```

## Critical Bug Fixed: Query Condensing

### The Problem

The RAG engine was configured with:
- **Memory**: `ChatMemoryBuffer` to store conversation history
- **Condense Prompt**: `CUSTOM_CONDENSE_PROMPT_TEMPLATE` to resolve references like "it", "this", "THESE"
- **Purpose**: Convert "use cases of THESE networks" → "use cases of CNNs" for better retrieval

**But** the code was calling:
```python
synthesizer, _, context_nodes = engine._run_c3(
    prompt, chat_history=None, streaming=True  # ❌ Bypasses condensing!
)
```

Passing `chat_history=None` explicitly tells `CondensePlusContextChatEngine` to **skip** the condensing step!

### The Fix

**Location:** `src/tensortruth/app_utils/chat_handler.py:92`

```python
# BEFORE (broken):
synthesizer, _, context_nodes = engine._run_c3(
    prompt, chat_history=None, streaming=True
)

# AFTER (fixed):
synthesizer, condensed_query, context_nodes = engine._run_c3(
    prompt, streaming=True
)
```

**Changes:**
1. Removed `chat_history=None` parameter - lets engine use its internal memory
2. Captured the second return value (previously ignored with `_`) - this is the **condensed query**
3. Display both user input and condensed query in debug output

### How It Works Now

**Phase 1: Condense (if chat history exists)**
- LLM receives: chat history + current user input + condense prompt
- LLM generates: standalone query with references resolved
- Example: "use cases of THESE" → "use cases of Convolutional Neural Networks (CNNs)"

**Phase 2: Retrieval**
- Vector DB searches using **condensed query**
- Returns: top-k documents with similarity scores
- Reranking and filtering applied

**Phase 3: Generation**
- LLM receives: chat history + retrieved context + formatted prompt
- Generates: answer based on context and conversation

### Verification

The debug output now shows:
```
🔎 Retrieval Query

User Input:
use cases of THESE networks

Condensed Query (used for retrieval):
🔄 Engine condensed the query with chat history to create a standalone search query
use cases of Convolutional Neural Networks (CNNs)
```

If no condensing happens (first message or standalone query), it shows:
```
*No condensing needed (first message or query already standalone)*
```

## What Still Needs Work

### 1. Verify Condensing Quality

The condense prompt is defined in `src/tensortruth/rag_engine.py:132`:

```python
CUSTOM_CONDENSE_PROMPT_TEMPLATE = (
    "Role: Technical Query Engineer.\n"
    "Task: Convert the user's follow-up input into a precise, standalone technical directive "
    "or search query based on the chat history.\n\n"
    # ... (see file for full prompt)
)
```

**Questions to investigate:**
- Is the condense prompt optimized for technical documentation retrieval?
- Does it preserve important keywords and technical terms?
- Does it handle multi-turn conversations well?
- Should we add examples or few-shot prompting?

**How to test:**
1. Run with `--debug-context`
2. Ask a technical question
3. Follow up with pronoun references ("it", "this", "THESE")
4. Check if condensed query is accurate and retrieval-friendly

### 2. Memory Window Size

Currently using `ChatMemoryBuffer.from_defaults(token_limit=3000)` in `rag_engine.py:426`.

**Questions:**
- Is 3000 tokens the right size for technical conversations?
- Should we use a different memory type (e.g., `VectorMemory` for long conversations)?
- How does memory size affect condensing quality?

### 3. Retrieval vs. Generation Context

**Current behavior:**
- Condense step uses chat history (now working)
- Retrieval uses condensed query (now working)
- Generation uses chat history + retrieved context (always worked)

**Questions:**
- Should we limit how much history goes to the condense step vs. generation step?
- Are we sending too much history to the generation LLM?
- Should we use different memory configurations for different phases?

### 4. Context Prompt Templates

There are 3 context prompts in `rag_engine.py`:
1. `CUSTOM_CONTEXT_PROMPT_TEMPLATE` - Normal RAG response (line 94)
2. `CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE` - Low confidence warning (line 104)
3. `CUSTOM_CONTEXT_PROMPT_NO_SOURCES` - No sources retrieved (line 113)

**Questions:**
- Are these prompts optimized?
- Should they include more guidance about using chat history?
- Do they encourage hallucination when sources are poor?

### 5. Confidence Thresholds

Current implementation in `chat_handler.py:_check_confidence_and_adjust_prompt()`:
- If `best_score < confidence_threshold`: Change prompt template, show warning
- If no nodes retrieved: Use fallback context, change template

**Questions:**
- Are the confidence thresholds well-calibrated?
- Should we adjust retrieval parameters based on confidence?
- Should we re-query with a different condensed query if confidence is low?

## Code Locations Reference

### RAG Engine Configuration
- **File:** `src/tensortruth/rag_engine.py`
- **Function:** `load_rag_engine()` (line 337)
- **Key sections:**
  - Memory setup: line 426
  - Condense prompt: line 460
  - Context prompts: lines 94, 104, 113

### Query Condensing
- **File:** `src/tensortruth/app_utils/chat_handler.py`
- **Function:** `_handle_rag_mode()` (line 75)
- **Key line:** 92 - `engine._run_c3()` call

### Debug Data Capture
- **File:** `src/tensortruth/app_utils/chat_handler.py`
- **RAG mode:** Lines 104-174
- **Simple LLM mode:** Lines 249-291

### Debug Rendering
- **Live rendering:** `src/tensortruth/app_utils/rendering_debug.py`
  - RAG: `render_debug_context()` (line 8)
  - Simple LLM: `render_debug_simple_llm()` (line 139)
- **Stored rendering:** `src/tensortruth/app_utils/rendering.py`
  - `render_debug_from_stored_data()` (line 273)

### Context String Assembly
- **File:** `src/tensortruth/app_utils/streaming.py`
- **Location:** Lines 147-161
- **Key line:** `context_str = "\n\n".join([n.get_content() for n in context_nodes])`

## Testing Recommendations

### Basic Flow Test
1. Start with `--debug-context` flag
2. Ask: "What are Convolutional Neural Networks?"
3. Follow up: "What are use cases of THESE networks?"
4. Verify condensed query shows "CNNs" instead of "THESE"

### Edge Cases
1. **No history:** First message should show "No condensing needed"
2. **Low confidence:** Query with no relevant docs should show warning
3. **Simple LLM mode:** Should show model config but no retrieval info

### Performance Test
1. Long conversation (10+ turns)
2. Check if memory management is working
3. Verify condensing still works correctly
4. Check if response quality degrades

## Data Storage

Debug data is stored in `~/.tensortruth/sessions.json` within each message:

```json
{
  "role": "assistant",
  "content": "...",
  "debug_data": {
    "mode": "rag",
    "user_query": "use cases of THESE",
    "condensed_query": "use cases of CNNs",
    "best_score": 0.8234,
    "confidence_threshold": 0.7,
    "num_nodes": 3,
    "node_scores": [...],
    "actual_context_str": "...",
    "actual_formatted_prompt": "...",
    "complete_conversation": "..."
  }
}
```

**Important:** This metadata is **never** passed to LLM. The `build_chat_history()` function (in `chat_utils.py`) only extracts `content` and `role` fields.

## Next Steps

1. **Validate condensing quality** - Use `--debug-context` to review condensed queries across various conversation patterns
2. **Tune prompts** - Experiment with condense and context prompt templates
3. **Optimize memory** - Test different memory sizes and types
4. **Measure impact** - Compare retrieval quality before/after condensing fix
5. **Add metrics** - Track condensing effectiveness (when it changes the query vs. when it doesn't)

## Backward Compatibility

The stored data rendering includes backward compatibility:
```python
user_query = debug_data.get("user_query") or debug_data.get("retrieval_query")
```

Old messages (before this fix) used `retrieval_query` field. New messages use both `user_query` and `condensed_query`.
