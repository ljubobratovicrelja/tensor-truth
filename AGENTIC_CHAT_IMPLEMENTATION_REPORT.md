# Agentic Chat Loop — Implementation Report

## Summary

All 12 stories (Stories 0-11) across 8 phases have been implemented on the `agentic-chat` branch. The implementation replaces the direct RAG pipeline with an Orchestrator Agent that processes every user prompt, decides whether to chat directly, query the knowledge base, search the web, fetch pages, or call MCP tools — then synthesizes a final response.

---

## Phase 1: Foundation (Stories 0, 1)

### Story 0: Orchestrator LLM Singleton
**File:** `src/tensortruth/core/ollama.py`

- Added module-level singleton variables: `_orchestrator_llm_instance`, `_orchestrator_llm_key`
- Added `get_orchestrator_llm(model, base_url, context_window=16384) -> Ollama`
  - Returns cached Ollama instance with `thinking=False`, `temperature=0.2`, `request_timeout=120.0`
  - Keyed by `(model, base_url)` tuple — replaced on model change
  - Deferred LlamaIndex import (only on first creation)
  - `additional_kwargs` sets `num_ctx` and `num_predict=-1` (unlimited generation)

### Story 1: Tool-Call Capability Detection
**File:** `src/tensortruth/core/ollama.py`

- Added `check_tool_call_support(model_name: str) -> bool` with `@lru_cache(maxsize=32)`
- Identical pattern to `check_thinking_support()` — queries `/api/show`, checks `"tools" in capabilities`
- Returns `False` on any error (safe default)

---

## Phase 2: Generalized Progress Reporting API (Story 2)

### Backend
**Files:** `services/models.py`, `api/schemas/chat.py`, `services/rag_service.py`, `services/chat_service.py`, `api/routes/chat.py`

- Added `ToolProgress` dataclass: `tool_id`, `phase`, `message`, `metadata`
- Added `StreamToolPhase` Pydantic model: `type="tool_phase"`, `tool_id`, `phase`, `message`, `metadata`
- Added `progress: Optional[ToolProgress]` field to `RAGChunk` (backward-compatible alongside `status`)
- RAGService yields both `status` (legacy) and `progress` (new) on every phase change
- WebSocket handler sends both `{"type": "status"}` and `{"type": "tool_phase"}` messages

### Frontend
**Files:** `api/types.ts`, `stores/chatStore.ts`, `hooks/useWebSocket.ts`, new `components/chat/ToolPhaseIndicator.tsx`, `components/chat/StreamingIndicator.tsx`

- Added `StreamToolPhase` interface to types and `StreamMessage` union
- Added `toolPhase` state + `setToolPhase` action to chatStore
- WebSocket handler processes `"tool_phase"` messages
- New `ToolPhaseIndicator` component with extensible `PHASE_ICON_MAP` (13 phases, lucide-react icons)
- `StreamingIndicator` delegates to `ToolPhaseIndicator` when `toolPhase` is present, falls back to legacy

---

## Phase 3: Tools (Stories 3, 4)

### Story 3: RAG as a Retrieval Tool
**Files:** `services/rag_service.py`, `services/models.py`, `services/orchestrator_tool_wrappers.py`

- Added `RAGRetrievalResult` dataclass: `source_nodes`, `confidence_level`, `metrics`, `condensed_query`, `num_sources`
- Added `RAGService.retrieve(query, params, session_messages, progress_callback) -> RAGRetrievalResult`
  - Non-streaming retrieval + reranking + confidence scoring — stops before LLM generation
- Created `create_rag_tool(rag_service, progress_emitter, session_params, session_messages) -> FunctionTool`
  - Async wrapper that runs `retrieve()` in executor thread
  - Returns formatted text with source summaries, scores, metrics
  - Source content truncated to 1500 chars per source

### Story 4: Tool Wrappers
**File:** `services/orchestrator_tool_wrappers.py`

- `create_web_search_tool(tool_service, progress_emitter) -> FunctionTool` — wraps `search_web`
- `create_fetch_page_tool(tool_service, progress_emitter) -> FunctionTool` — wraps `fetch_page`
- `create_fetch_pages_batch_tool(tool_service, progress_emitter) -> FunctionTool` — parallel URL fetching
- `create_all_tool_wrappers(...)` — convenience factory for all tools
- All use closure pattern: dependencies captured at construction time
- `ProgressEmitter` type alias, `_emit()` helper handles sync/async emitters
- Pydantic input schemas for each tool

---

## Phase 4: Core Orchestrator (Story 5)

**File:** `services/orchestrator_service.py` (651 lines)

### OrchestratorService class
- **`__init__`**: Accepts tool_service, rag_service, model config, session params, module_descriptions, custom_instructions, project_metadata, max_iterations
- **`_build_tools(progress_emitter)`**: Creates wrapped tools + filters/adds MCP tools (excludes `_WRAPPED_BUILTIN_TOOL_NAMES`)
- **`_build_system_prompt(tools)`**: 5-section prompt: role, module enumeration, tool routing guidance, project metadata, explicit tool list
- **`_budget_history(history, system_prompt, user_prompt)`**: 12% system / 18% history / 18% user / 50% response buffer, backward turn-based truncation
- **`execute(prompt, chat_history, progress_emitter)`**: Async generator yielding `OrchestratorEvent` objects. Creates FunctionAgent per-execution (LLM is singleton). Streams events via `handler.stream_events()`.
- **`get_tool_names()`**: Returns sorted list of registered tool names

### OrchestratorEvent dataclass
- Fields: `token`, `tool_call`, `tool_call_result`, `tool_phase` (exactly one populated per event)

### load_module_descriptions(modules, config)
- Standalone function loading ChromaDB metadata for each active module
- Falls back gracefully to bare module names on error

---

## Phase 5: Streaming Integration (Story 6)

**File:** `services/orchestrator_stream.py`

### translate_event(event) -> Optional[Dict]
- Stateless translation: token->StreamToken, tool_call->StreamToolProgress(calling), tool_call_result->StreamToolProgress(completed/failed), tool_phase->StreamToolPhase

### OrchestratorStreamTranslator class
- Stateful wrapper that accumulates response text, sources, metrics, confidence, tool_steps
- `process_event(event)` → WebSocket message dict
- `set_rag_retrieval_result(result)` → injects RAG result for proper source extraction
- `build_sources_message()` → batched sources with `source_types`, `rag_count`, `web_count`
- `build_done_message(title_pending)` → final done message with confidence
- RAG sources extracted via `ChatService.extract_sources()`, web sources parsed from JSON output

### RAG result callback
- Added `rag_result_callback` parameter to `create_rag_tool()` — stores raw `RAGRetrievalResult` on OrchestratorService for source extraction

---

## Phase 6: Integration & Config (Stories 7, 8, 10)

### Story 7: WebSocket Handler Refactor
**File:** `api/routes/chat.py`

- New `_is_orchestrator_enabled(session, model_name) -> bool`: checks config + model capability
- New `_run_orchestrator_path(websocket, context, session, ...)`: complete orchestrator execution path
  - Loads module descriptions, creates OrchestratorService, creates translator
  - Progress emitter bridges sync/async boundary via `loop.call_soon_threadsafe`
  - Streams events, sends sources, sends done message, saves assistant message with tool_steps
- Routing: `/command` -> CommandRegistry (unchanged) -> orchestrator check -> orchestrator path or ChatService fallback
- Graceful fallback: if orchestrator raises, falls through to ChatService path

### Story 8: Configuration & Settings

**Backend:**
- `app_utils/config_schema.py`: Added `orchestrator_enabled: bool = True` to `AgentConfig`
- `api/schemas/config.py`: Added to `AgentConfigSchema`
- `api/routes/config.py`: New `GET /api/config/model-capabilities?model=<name>` endpoint
- `services/session_service.py`: Default propagation to new sessions

**Frontend:**
- New `components/ui/switch.tsx` (shadcn/ui, requires `@radix-ui/react-switch`)
- `api/types.ts`: Added `orchestrator_enabled` to `AgentConfig`, new `ModelCapabilitiesResponse`
- `api/config.ts`: Added `getModelCapabilities()` API function
- `hooks/useConfig.ts`: Added `useModelCapabilities()` hook (5-min stale time)
- `components/config/SessionSettingsPanel.tsx`: "Agentic mode" toggle in Agent section, disabled when model lacks tool support

### Story 10: Tool Auto-Discovery
**Files:** `services/orchestrator_service.py`, `tests/unit/test_orchestrator_tool_discovery.py`

- Verified `_build_tools()` handles all edge cases correctly
- Added `get_tool_names()` method
- Added DEBUG-level per-tool description logging
- 23 unit tests verifying tool assembly, filtering, descriptions

---

## Phase 7: Mixed Source Types (Story 9)

**Backend:** `api/schemas/chat.py` — Added `source_types`, `rag_count`, `web_count` to `StreamSources`

**Frontend:**
- `api/types.ts`: Added optional fields to `StreamSources`
- `stores/chatStore.ts`: Added `streamingSourceTypes` state
- `hooks/useWebSocket.ts`: Passes `source_types` to `setSources`
- `components/chat/SourceCard.tsx`: Refactored `SourcesList` for grouped rendering
  - Mixed sources: "Knowledge Base Results" (BookOpen icon) + "Web Results" (Globe icon) sections
  - Single-type: renders exactly as before (full backward compat)

---

## Phase 8: Testing (Story 11)

### New test files:
- `tests/unit/test_orchestrator_service.py` (36 tests)
- `tests/integration/test_orchestrator_flow.py` (7 tests)

### Pre-existing test files (from earlier stories):
- `tests/unit/test_orchestrator_tool_wrappers.py` (16 tests)
- `tests/unit/test_orchestrator_tool_discovery.py` (23 tests)
- `tests/unit/test_orchestrator_stream.py` (21 tests)
- `tests/unit/test_rag_service.py` (7 new tests for `retrieve()`)

### Total new orchestrator test coverage: ~110 tests

---

## Files Created (New)

| File | Story | Purpose |
|------|-------|---------|
| `services/orchestrator_tool_wrappers.py` | 3, 4 | FunctionTool factories for RAG + web tools |
| `services/orchestrator_service.py` | 5 | Core orchestrator (FunctionAgent wrapper) |
| `services/orchestrator_stream.py` | 6 | Event translation + source accumulation |
| `components/chat/ToolPhaseIndicator.tsx` | 2 | Phase-aware streaming indicator |
| `components/ui/switch.tsx` | 8 | shadcn/ui Switch component |
| `tests/unit/test_orchestrator_service.py` | 11 | Orchestrator unit tests |
| `tests/unit/test_orchestrator_tool_wrappers.py` | 3, 4 | Tool wrapper tests |
| `tests/unit/test_orchestrator_tool_discovery.py` | 10 | Tool discovery tests |
| `tests/unit/test_orchestrator_stream.py` | 6 | Stream translation tests |
| `tests/integration/test_orchestrator_flow.py` | 11 | Integration tests |

## Files Modified

| File | Stories | Changes |
|------|---------|---------|
| `core/ollama.py` | 0, 1 | LLM singleton + tool-call detection |
| `services/models.py` | 2, 3 | ToolProgress + RAGRetrievalResult dataclasses |
| `services/rag_service.py` | 2, 3 | Progress emission + retrieve() method |
| `services/chat_service.py` | 2 | ToolProgress in loading_models status |
| `services/__init__.py` | 2, 5 | Exports for ToolProgress, OrchestratorService |
| `api/schemas/chat.py` | 2, 9 | StreamToolPhase + StreamSources extensions |
| `api/routes/chat.py` | 2, 7 | tool_phase messages + orchestrator routing |
| `api/routes/config.py` | 8 | model-capabilities endpoint |
| `api/schemas/config.py` | 8 | orchestrator_enabled field |
| `app_utils/config_schema.py` | 8 | AgentConfig.orchestrator_enabled |
| `services/session_service.py` | 8 | Default propagation |
| `api/types.ts` | 2, 8, 9 | StreamToolPhase + config + sources types |
| `stores/chatStore.ts` | 2, 9 | toolPhase + streamingSourceTypes state |
| `hooks/useWebSocket.ts` | 2, 9 | tool_phase handler + source_types |
| `hooks/useConfig.ts` | 8 | useModelCapabilities hook |
| `api/config.ts` | 8 | getModelCapabilities API function |
| `components/chat/StreamingIndicator.tsx` | 2 | ToolPhaseIndicator delegation |
| `components/chat/SourceCard.tsx` | 9 | Mixed source grouped rendering |
| `components/chat/index.ts` | 2 | ToolPhaseIndicator export |
| `components/config/SessionSettingsPanel.tsx` | 8 | Agentic mode toggle |
| `tests/unit/test_rag_service.py` | 3 | retrieve() tests |

## Dependencies Added
- `@radix-ui/react-switch` (frontend, for shadcn/ui Switch component)

## Known Design Notes
- Orchestrator is ON by default (`orchestrator_enabled: true`)
- Hard-disabled when model lacks `"tools"` capability
- FunctionAgent created per-execution (lightweight); LLM is the cached singleton
- `/command` dispatch bypasses orchestrator entirely
- Fallback to direct ChatService.query() on orchestrator failure
- RAG tool returns sources+confidence only; orchestrator synthesizes final response
- Context budget: 12% system / 18% history / 18% user / 50% response
