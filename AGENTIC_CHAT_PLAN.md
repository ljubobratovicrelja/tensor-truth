# Agentic Chat Loop — Implementation Plan

## Execution Model

Each implementation phase maps to **one Claude Code session** (one agent plan + execution cycle). Phases are scoped to fit within a single context window — no phase should require more files or changes than one session can handle without context overflow. If a phase is too large, it must be split.

### Instructions for Agents

1. **Before doing anything**, read `.agent/context.md` for general instructions on how to behave within this codebase (conventions, patterns, tooling, etc.).
2. After completing a phase, **launch a separate audit agent** that reads this plan and verifies the changes made are consistent with the plan's intent, correct in implementation, and don't break existing functionality. The audit agent should report any discrepancies before the phase is considered done.

## Overview

Replace the current direct RAG pipeline with an **Orchestrator Agent** that processes every user prompt, decides whether to chat directly, query the knowledge base, search the web, fetch pages, or call MCP tools — then synthesizes a final response.

The orchestrator is built on top of LlamaIndex's existing `FunctionAgent` (already proven in the codebase), not a custom loop.

## Architecture

```
User prompt (WebSocket)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Orchestrator (FunctionAgent with tool-calling)      │
│                                                      │
│  Tools available (auto-discovered at startup):       │
│  ├── rag_query(query)         → RAGService           │
│  ├── web_search(query)        → ToolService          │
│  ├── fetch_page(url)          → ToolService          │
│  ├── fetch_pages_batch(urls)  → ToolService          │
│  ├── <MCP tools>              → ToolService          │
│  └── ... (all registered tools)                      │
│                                                      │
│  Agentic loop (FunctionAgent handles internally):    │
│  call tools → inspect results → decide               │
│  → call more tools or produce final answer           │
└──────────────────────────────────────────────────────┘
    │
    ▼
Stream final response to frontend
(sources from tools forwarded in parallel)
```

### Key Principles

- **Every message goes through the orchestrator.** No pre-filter. Uniform path.
- **RAG is a retrieval tool**, not the default path. The orchestrator decides when to query indexed docs. RAG returns sources + confidence only — the orchestrator itself synthesizes the final response with all tool results in context.
- **Same model** as the session. No separate model for orchestration.
- **Thinking disabled** on the orchestrator LLM instance. Fast tool-call decisions only.
- **Orchestrator LLM is a hot singleton.** Created once, reloaded only on model change.
- **Uses existing FunctionAgent.** No custom agentic loop — reuse `FunctionAgentWrapper` pattern from `agents/function/`.
- **Tools only, not agents.** The orchestrator calls tools directly (RAG, web_search, fetch_page, MCP tools). Users invoke complex agents (browse, doc_researcher) explicitly via `/commands`. This avoids agent-in-agent nesting and iteration explosion.
- **The orchestrator subsumes BrowseAgent's capabilities.** With `web_search` + `fetch_page` tools, the orchestrator can autonomously perform multi-step web research — more powerfully than BrowseAgent, because it makes free agentic decisions rather than following a deterministic router.
- **Graceful degradation.** If the model doesn't support tool-calling, fall back to the current direct RAG path silently.
- **Opt-out via config.** `orchestrator_enabled` setting in UI + `config.yaml`. Default: `true`.

### Deprecations

- **IntentService**: Has an active `/intent` HTTP endpoint but is not used in the main chat flow (commands handle explicit tool/agent launching). Marked for deprecation — orchestrator replaces all intent routing. The `/intent` endpoint can be removed or kept for backward compatibility.

---

## Stories

### Story 0: Orchestrator LLM Singleton

**Goal:** Create and cache a non-thinking Ollama LLM instance for orchestration.

**Details:**
- New helper in `core/ollama.py`: `get_orchestrator_llm(model, base_url, context_window)`.
- Returns an `Ollama` instance with `thinking=False`, same model/base_url as the session LLM.
- Module-level singleton keyed by `(model, base_url)`. If the session model changes, the singleton is replaced with a new instance for the new model.
- On first call, the instance is created and kept hot. Subsequent calls return the cached instance. This avoids the LlamaIndex Ollama client setup overhead on every message.
- Ollama itself manages GPU memory — multiple LlamaIndex `Ollama` instances pointing to the same model share the single loaded model in VRAM.

**Files:** `src/tensortruth/core/ollama.py`

---

### Story 1: Tool-Call Capability Detection

**Goal:** Detect whether the active Ollama model supports native tool-calling.

**Details:**
- New function in `core/ollama.py`: `check_tool_call_support(model_name) -> bool`.
- Identical pattern to existing `check_thinking_support()`: query Ollama `/api/show`, check for `"tools" in capabilities`.
- Confirmed: `gpt-oss:120b` returns `capabilities: ["completion", "tools", "thinking"]`.
- Cache the result per model name (same LRU pattern as thinking check).
- If tool-calling is not supported, the orchestrator is hard-disabled for that session regardless of the config setting.

**Files:** `src/tensortruth/core/ollama.py`

---

### Story 2: Generalized Progress Reporting API

**Goal:** Replace the hardcoded pipeline status system with an extensible, tool-aware progress API that works across backend and frontend.

**Current problem:**
- Backend emits `{"type": "status", "status": "retrieving"}` with 5 hardcoded values.
- Frontend `PipelineStatus` is a narrow string union. `StreamingIndicator` maps these to hardcoded funny messages.
- No extensibility for new tools to report their own phases.

**New design — Backend:**
- New dataclass in `services/models.py`:
  ```python
  @dataclass
  class ToolProgress:
      tool_id: str     # "rag", "web_search", "orchestrator", etc.
      phase: str       # "retrieving", "searching", "fetching", etc.
      message: str     # Human-readable: "Searching your documents..."
      metadata: dict   # Phase-specific: {"pages_fetched": 3, "pages_target": 5}
  ```
- **New** WebSocket message schema in `api/schemas/chat.py` — note this is a **separate type** from the existing `StreamToolProgress` (which tracks tool execution steps with `tool`, `action`, `params`, `output`, `is_error`). The existing schema stays unchanged for tool step tracking. The new schema is for phase-level progress reporting:
  ```python
  class StreamToolPhase(BaseModel):
      type: Literal["tool_phase"] = "tool_phase"
      tool_id: str
      phase: str
      message: str
      metadata: dict = {}
  ```
- RAGService updated to yield `ToolProgress` objects instead of bare status strings. Existing `RAGChunk.status` kept temporarily for backward compat, with `progress` field added alongside.
- Each tool defines its own phases and messages — progress is co-located with the tool, not scattered across UI.

**New design — Frontend:**
- New `ToolPhase` interface in `api/types.ts` (distinct from existing `StreamToolProgress` which stays for tool step tracking):
  ```typescript
  interface StreamToolPhase {
    type: "tool_phase";
    tool_id: string;
    phase: string;
    message: string;
    metadata: Record<string, unknown>;
  }
  ```
- `chatStore.ts`: Add `toolPhase: StreamToolPhase | null` field alongside existing `pipelineStatus` (kept for backward compat during migration).
- New `ToolPhaseIndicator` component: renders `phase.message` with phase-appropriate icon and animation. Icons mapped by phase name (extensible record, not hardcoded switch). Falls back to generic spinner for unknown phases.
- `StreamingIndicator` delegates to `ToolPhaseIndicator` when `toolPhase` is present, falls back to legacy status rendering otherwise.
- Funny messages preserved: can be layered on top of specific phases (e.g., "generating" phase shows random fun messages).

**Relationship to existing `StreamToolProgress`:** The existing `StreamToolProgress` (`type: "tool_progress"`) with fields `tool`, `action`, `params`, `output`, `is_error` continues to serve its current role — tracking tool execution steps (calling/completed/failed) shown in `ToolSteps` component. The new `StreamToolPhase` (`type: "tool_phase"`) is a separate message type for real-time phase progress (e.g., "Searching knowledge base...", "Ranking results...") shown in the streaming indicator. Both coexist.

**Migration:** Non-breaking. Backend sends old `status` messages alongside new `tool_phase` messages during transition. Frontend handles both. Old `status` type deprecated after full migration.

**Files:**
- Backend: `services/models.py`, `api/schemas/chat.py`, `services/rag_service.py`, `api/routes/chat.py`
- Frontend: `api/types.ts`, `stores/chatStore.ts`, `hooks/useWebSocketChat.ts`, new `components/chat/ToolProgressIndicator.tsx`, `components/chat/StreamingIndicator.tsx`

---

### Story 3: RAG as a Retrieval Tool

**Goal:** Wrap the existing RAG retrieval pipeline as a callable tool for the orchestrator.

**Key design decision:** The RAG tool returns **sources + confidence only**, not a synthesized response. The orchestrator receives the retrieved context and synthesizes the final answer itself (with all tool results in context). This means:
- RAG tool does retrieval + reranking + confidence scoring — stops before LLM generation.
- The orchestrator's own LLM call acts as the synthesizer.
- Sources and metrics are forwarded to the frontend in parallel (same UX as current RAG: source cards, confidence badges, metrics panel).

**Details:**
- New method on `RAGService`: `retrieve(query, chat_context) -> RAGRetrievalResult` — a non-streaming method that performs retrieval, reranking, and confidence scoring, returning sources and metrics without calling the LLM for generation.
- `rag_query` `FunctionTool` wraps this method:
  - **Input:** `query: str`
  - **Output:** Structured string with source summaries, confidence level, and metadata.
  - **Side effect:** Emits `ToolProgress` messages ("Searching knowledge base...", "Ranking results...") and forwards sources to the frontend via the progress emitter.
- **Progress emitter injection:** The `rag_query` tool function is constructed as a closure in `orchestrator_tools.py` that captures the progress emitter at tool-creation time. Same pattern for all tool wrappers — the emitter is bound when the orchestrator builds its tool set, not passed per-call.
- RAGService internals stay unchanged. The new `retrieve()` method reuses existing retrieval/reranking code, just skips the LLM generation phase.

**Files:** `services/rag_service.py` (add `retrieve()` method), new `services/orchestrator_tools.py`

---

### Story 4: Wrap Existing Tools for Orchestrator

**Goal:** Expose web_search, fetch_page, and MCP tools to the orchestrator.

**Details:**
- In `orchestrator_tools.py`, create thin `FunctionTool` wrappers:
  - `web_search(query: str) -> str` — calls `ToolService.execute_tool("search_web", ...)`. Emits "Searching the web...".
  - `fetch_page(url: str) -> str` — calls `ToolService.execute_tool("fetch_page", ...)`. Emits "Fetching page...".
  - `fetch_pages_batch(urls: list[str]) -> str` — parallel URL fetching.
- MCP tools: Pass through directly from `ToolService.tools`. These already have FunctionTool schemas. No wrapping needed — include them in the orchestrator's tool set as-is.
- **Progress emitter injection:** Same closure pattern as Story 3 — each wrapper captures the emitter at construction time.

**No `run_agent` tool.** The orchestrator only calls tools directly. Users invoke agents via `/commands`. This prevents agent nesting and iteration explosion.

**Files:** `services/orchestrator_tools.py`, `services/tool_service.py`

---

### Story 5: Orchestrator Service

**Goal:** The core orchestrator built on LlamaIndex FunctionAgent.

**Details:**
- New `services/orchestrator_service.py` with class `OrchestratorService`.
- **Initialization:**
  - Receives `ToolService`, `RAGService`, session config.
  - Builds tool set dynamically: RAG tool + built-in tool wrappers + MCP tools.
  - Gets (or creates) the hot singleton orchestrator LLM via `get_orchestrator_llm()`.
- **Uses `FunctionAgent` (not a custom loop):**
  - Reuses the pattern from `agents/function/factory.py` and `FunctionAgentWrapper`.
  - FunctionAgent receives the orchestrator LLM + tool set + system prompt.
  - FunctionAgent handles the tool-calling loop internally (call → result → decide → repeat).
  - Events (`ToolCall`, `ToolCallResult`, `AgentStream`) are streamed via the existing wrapper pattern.
- **Module metadata plumbing:**
  - `ChatContext.modules` is currently `List[str]` (names only). The orchestrator needs module descriptions for the system prompt.
  - `ModuleInfo` currently has `name`, `display_name`, `doc_type`, `sort_order` — but no `description` field. Rather than requiring a catalog migration, compose descriptions from existing fields: `"{display_name} ({doc_type})"` e.g., "PyTorch Docs (library_doc)", "Attention Is All You Need (paper)". This is pragmatic and sufficient for LLM routing.
  - Add module metadata loading: during `ChatContext` construction (in `chat.py`), fetch `ModuleInfo` from the catalog/modules service for each active module. Store as `ChatContext.module_metadata: List[ModuleInfo]`.
  - This is a small addition to `ChatContext.from_session()` — the module info already exists in the catalog, it just isn't wired into ChatContext yet.
- **System prompt composition:**
  - Describes the assistant's role and available tools.
  - **Enumerates all loaded/active vector index modules with names and descriptions**, composed from the newly-plumbed `ChatContext.module_metadata`. Example:
    ```
    You have access to a knowledge base with the following indexed modules:
    - pytorch_docs: PyTorch official documentation
    - linear_algebra_textbook: "Linear Algebra Done Right" by Sheldon Axler
    - attention_paper: "Attention Is All You Need" (Vaswani et al.)

    Use rag_query when the user's question likely relates to these topics.
    For current/live information not in these modules, use web_search + fetch_page.
    For simple conversational messages, respond directly without tools.
    ```
  - Includes session custom instructions and project metadata.
  - This is the primary mechanism for RAG routing — no heuristic needed. If the LLM misjudges, the user can nudge with "consult the RAG modules" or "search my documents".
- **Context window budgeting** (adopted from BrowseAgent pattern in `core/synthesis.py`):
  - Percentage-based allocation of the context window:
    - System prompt (modules + tool schemas): ~12%
    - Chat history: ~18% (turn-based truncation, oldest turns dropped first)
    - User prompt: ~18%
    - Response buffer: ~50%
  - Token-to-char conversion: 4 chars ≈ 1 token (same constant used in BrowseAgent).
  - History truncated via `ChatHistoryService` turn-based limiting (already exists).
  - If budget is tight: system prompt is non-negotiable, user prompt is essential, history is trimmed first.
- **Error handling:**
  - Tool failure → error message fed back to the LLM (FunctionAgent already does this via `is_error` flag on `ToolCallResult`). The LLM analyzes the cause:
    - If input issue → corrects and retries.
    - If internal tool error → continues with other tools if possible, or reports the failure in its response.
  - Tool errors reported to frontend as `StreamToolPhase` messages with error phase, and via `StreamToolProgress` with `action: "failed"`. Errors can also appear in the sources/steps panel (same UX style as source cards — consistent error reporting).
  - Max iterations: configurable, default 10. If exhausted, the LLM's last response is used as-is.
- **Conversation history:** Passed from `ChatHistoryService`, truncated to context budget.

**Files:** New `services/orchestrator_service.py`

---

### Story 6: Synthesizer Integration & Streaming

**Goal:** Stream the orchestrator's final response to the frontend.

**Details:**
- The orchestrator (FunctionAgent) handles synthesis naturally: the LLM's final text response (after all tool calls) IS the synthesized answer — it has all tool results in context.
- **Streaming approach:** FunctionAgent streams events via `handler.stream_events()`. When the final response generates, `AgentStream` events carry text deltas (forwarded as `{"type": "token"}` messages). This is real streaming, same pattern as existing `FunctionAgentWrapper`.
- **Event translation layer:** FunctionAgent emits LlamaIndex event types (`ToolCall`, `ToolCallResult`, `AgentStream`). An adapter in `OrchestratorService` translates these to WebSocket messages:
  - `ToolCall` → `StreamToolProgress` (existing schema: `action: "calling"`, tool name, params) + `StreamToolPhase` (new schema: phase message like "Searching the web...")
  - `ToolCallResult` → `StreamToolProgress` (`action: "completed"` or `"failed"`, output)
  - `AgentStream` → `{"type": "token", "content": delta}`
  - This adapter reuses the same mapping logic from `FunctionAgentWrapper.run()` (lines 59-86 of `wrapper.py`).
- **Sources:** Collected from RAG and web tools during execution, forwarded to frontend via a single batched `{"type": "sources"}` message after all tools complete. Backend accumulates sources from all tool calls and sends them together.
- **Confidence level:** Derived from RAG tool result if called, otherwise "high" for tool-free or web-only responses.
- **Metrics:** RAG retrieval metrics forwarded alongside sources when RAG was called.

**Files:** `services/orchestrator_service.py`, `api/routes/chat.py`

---

### Story 7: WebSocket Handler Refactor

**Goal:** Route messages through the orchestrator in the WebSocket handler.

**Details:**
- In `api/routes/chat.py`, the WebSocket handler currently:
  1. Receives prompt.
  2. Checks for `/command` prefix → dispatch to CommandRegistry.
  3. Otherwise → `ChatService.query()` → stream RAG response.
- New flow:
  1. Receives prompt.
  2. `/command` prefix → dispatch to CommandRegistry (unchanged, explicit override).
  3. Check if orchestrator is enabled (config + model capability via `check_tool_call_support()`).
  4. If orchestrator enabled → `OrchestratorService.execute()` → stream response.
  5. If orchestrator disabled → existing `ChatService.query()` path (unchanged fallback).
- The orchestrator receives a `WebSocketProgressEmitter` for forwarding tool progress and sources to the frontend.
- Final response tokens streamed via the same `{"type": "token"}` messages.
- Tool steps saved in the assistant message's `tool_steps` field (already optional in message schema).

**Files:** `api/routes/chat.py`

---

### Story 8: Configuration & Settings

**Goal:** Add orchestrator toggle to config and UI settings.

**Details:**
- **Backend config:**
  - Add `orchestrator_enabled: bool = True` to session params schema.
  - Add to global config (`config.yaml` / `AppConfig`).
  - Session-level overrides global. If model lacks `"tools"` capability → hard-disabled regardless.
- **API:**
  - Expose in session settings endpoints (already exist for other params).
  - Extend `GET /api/config/capabilities` (or existing model info endpoint) to return `{ orchestrator_available: bool }` based on current model's tool-call support.
- **Frontend:**
  - Add switch toggle in session settings panel: "Agentic mode" (on/off). Place in existing "Agent" section.
  - Disable with tooltip if model doesn't support tool-calling.
  - Store in session params (same pattern as temperature, model selection).

**Files:** Backend: `api/schemas/`, `services/session_service.py`, `config/`. Frontend: `SessionSettingsPanel.tsx`, hooks, API types.

---

### Story 9: Mixed Source Types — Frontend

**Goal:** Support displaying RAG sources and web sources together in a single response.

**Context:** Previously impossible — responses had either RAG sources or web sources, never both. The orchestrator can call both `rag_query` and `web_search` in one turn.

**Backend changes:**
- Single `StreamSources` message with all sources mixed. Backend groups by `doc_type` and includes counts:
  ```python
  {
      "type": "sources",
      "data": [...all sources...],
      "metrics": {...},          # Only for RAG sources
      "source_types": ["rag", "web"],
      "rag_count": 5,
      "web_count": 3
  }
  ```
- Sources accumulated during orchestrator execution and sent as one batched message (not incremental — avoids the `setSources()` replace-not-append problem).

**Frontend changes — Grouped sections approach:**
- `SourcesList` component detects mixed source types and renders grouped sections:
  - **RAG Results** section: source cards + MetricsPanel (score distribution, diversity, coverage).
  - **Web Results** section: source cards with fetch status badges (fetched/failed/skipped).
- When only one type present, renders as today (no grouping UI).
- Grouping done by filtering on `metadata.doc_type === "web"` (already used in codebase).
- `StreamSources` type updated with optional `source_types`, `rag_count`, `web_count` fields.

**Files:**
- Backend: `api/routes/chat.py`, `api/schemas/chat.py`
- Frontend: `api/types.ts`, `components/chat/SourceCard.tsx` (SourcesList refactor), `hooks/useWebSocketChat.ts`

---

### Story 10: Orchestrator Tool Schema Auto-Discovery

**Goal:** Dynamically build the orchestrator's tool set from all registered tools.

**Details:**
- On orchestrator initialization (per-session, or on config change):
  1. Create `rag_query` FunctionTool from `orchestrator_tools.py`.
  2. Create built-in tool wrappers (web_search, fetch_page, fetch_pages_batch).
  3. Collect MCP tools from `ToolService.tools` — pass through as-is (already FunctionTools with schemas).
  4. Build complete tool list for FunctionAgent.
- Tool descriptions must be clear and concise — the LLM uses them to decide which tool to call.
- MCP tools that become unavailable mid-session: if the orchestrator calls one and it fails, the error is fed back to the LLM (Story 5 error handling). Acceptable degradation.

**Files:** `services/orchestrator_service.py`, `services/orchestrator_tools.py`

---

### Story 11: End-to-End Testing

**Goal:** Verify the orchestrator works across all paths.

**Details:**
- **Unit tests:**
  - Orchestrator routes a knowledge question to `rag_query`.
  - Orchestrator routes a web search request to `web_search`.
  - Orchestrator handles direct conversational response (no tools).
  - Orchestrator calls multiple tools in sequence (web_search → fetch_page).
  - Orchestrator disables gracefully when model lacks `"tools"` capability.
  - Singleton LLM instance is reused across calls, replaced on model change.
  - Config toggle works (enabled/disabled).
  - Tool failure → LLM receives error → adapts or reports.
  - Context window budget respects limits.
- **Integration tests:**
  - WebSocket flow with orchestrator enabled.
  - WebSocket flow with orchestrator disabled (fallback to direct RAG).
  - `/command` bypass still works with orchestrator enabled.
  - Progress messages arrive in correct order.
  - Mixed sources (RAG + web) render correctly.

**Files:** `tests/unit/test_orchestrator_service.py`, `tests/integration/test_orchestrator_flow.py`

---

## Implementation Order

### Dependency Graph

```
     ┌─────────┐   ┌─────────┐
     │ Story 0 │   │ Story 1 │     ← Phase 1 (parallel, no deps)
     └────┬────┘   └────┬────┘
          │              │
          └──────┬───────┘
                 ▼
          ┌─────────────┐
          │   Story 2   │             ← Phase 2 (backend then frontend)
          └──────┬──────┘
          ┌──────┴──────┐
          ▼             ▼
     ┌─────────┐   ┌─────────┐
     │ Story 3 │   │ Story 4 │     ← Phase 3 (parallel)
     └────┬────┘   └────┬────┘
          └──────┬───────┘
                 ▼
          ┌─────────────┐
          │   Story 5   │             ← Phase 4 (largest, sequential)
          └──────┬──────┘
                 ▼
          ┌─────────────┐
          │   Story 6   │             ← Phase 5 (sequential)
          └──────┬──────┘
     ┌───────────┼───────────┐
     ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌──────────┐
│ Story 7 │ │ Story 8 │ │ Story 10 │  ← Phase 6 (all three parallel)
└────┬────┘ └────┬────┘ └──────────┘
     └──────┬────┘
            ▼
     ┌─────────────┐
     │   Story 9   │                  ← Phase 7 (needs 7 & 8)
     └──────┬──────┘
            ▼
     ┌─────────────┐
     │  Story 11   │                  ← Phase 8 (needs all)
     └─────────────┘
```

### Phase Breakdown

**Phase 1: Foundation** — Stories 0, 1 (parallel)
- No dependencies. Both are small additions to `core/ollama.py`.
- Exit: singleton helper and capability check available.

**Phase 2: Progress API** — Story 2
- Depends on: Phase 1 (needs ollama utilities for testing).
- Backend first (schemas + `ToolProgress` dataclass + RAGService refactor), then frontend (`ToolPhaseIndicator` component, store changes, WebSocket handler).
- Story 2 backend and frontend can overlap once schemas are defined.
- Exit: extensible progress system working end-to-end, old `status` messages still functional.

**Phase 3: Tools** — Stories 3, 4 (parallel)
- Depends on: Phase 2 (tool wrappers emit `ToolProgress`).
- Story 3 (RAG retrieval extraction) and Story 4 (tool wrappers) are independent of each other.
- Exit: all orchestrator tools unit-tested in isolation.

**Phase 4: Core Orchestrator** — Story 5 (largest story)
- Depends on: Phases 1-3 (needs LLM singleton, capability check, all tools).
- This is the critical path bottleneck. Includes: FunctionAgent integration, system prompt composition, module metadata plumbing, context window budgeting, error handling.
- Exit: orchestrator service accepts a query with tools and returns a structured response.

**Phase 5: Streaming Integration** — Story 6
- Depends on: Phase 4.
- Adds the event translation adapter (FunctionAgent events → WebSocket messages) and source accumulation logic.
- Exit: orchestrator streams tokens, tool progress, and sources over WebSocket.

**Phase 6: Integration & Config** — Stories 7, 8, 10 (all three parallel)
- All depend on Phase 5 (Story 10 technically only needs Phase 4, so it can start even earlier).
- Story 7: WebSocket handler routing (backend only).
- Story 8: Config schema + settings UI (backend + frontend).
- Story 10: Tool auto-discovery formalization (backend only).
- Exit: orchestrator wired into production WebSocket path, configurable via UI.

**Phase 7: Mixed Sources** — Story 9
- Depends on: Stories 7, 8 (needs orchestrator active and configurable to test mixed sources end-to-end).
- Backend schema updates + frontend grouped source rendering.
- Exit: RAG + web sources display correctly in the same response.

**Phase 8: Testing** — Story 11
- Depends on: all stories complete.
- Unit tests, integration tests, end-to-end verification.

### Notes on Parallelism

- **Frontend work** in Stories 2, 8, 9 can proceed as soon as backend schemas are defined (interface-first), without waiting for full backend implementation.
- **Story 10** has the weakest dependency chain — only needs Story 5. Can start as early as Phase 4 completion, running alongside Phase 5 or 6.
- **Story 5 is the bottleneck.** Everything after Phase 3 flows through it. No way to parallelize around it — it's the core service that everything integrates with.

---

## Resolved Decisions

1. **No custom agentic loop.** Reuse LlamaIndex FunctionAgent and existing `FunctionAgentWrapper` pattern. No wheel reinvention.
2. **Tools only, no agent delegation.** Orchestrator calls tools (RAG, web_search, fetch_page, MCP). Complex agents (browse, doc_researcher) invoked via `/commands` by the user. No agent nesting.
3. **RAG returns sources only.** The RAG tool does retrieval + reranking + confidence — stops before LLM synthesis. Orchestrator synthesizes the final response with all tool context.
4. **Tool-call detection:** `"tools" in capabilities` from Ollama `/api/show`. Confirmed working. Same pattern as thinking detection.
5. **Error handling:** Errors fed to LLM for analysis. LLM retries on input issues, reports on internal errors. Errors displayed in UI same style as sources.
6. **IntentService deprecated.** Not plugged into production flow. Orchestrator replaces all routing.
7. **Concurrency:** Not an issue — frontend locks input during streaming.
8. **Context budget:** Percentage-based allocation from BrowseAgent pattern (12% system, 18% history, 18% prompt, 50% response).
9. **Source merging:** Backend accumulates all sources and sends one batched message. Frontend groups by `doc_type` into sections.
10. **Progress API:** New `StreamToolPhase` message type for phase-level progress (separate from existing `StreamToolProgress` for tool step tracking). Each tool defines its own phases and messages. Replaces hardcoded `PipelineStatus`.
11. **Module metadata:** `ChatContext` extended to load module descriptions from catalog — required for orchestrator system prompt composition.
12. **Event translation:** Adapter in OrchestratorService maps FunctionAgent events (`ToolCall`, `ToolCallResult`, `AgentStream`) to WebSocket message types (`StreamToolProgress`, `StreamToolPhase`, `token`).
13. **`/command` stays as user-explicit bypass.** Orchestrator doesn't detect or route commands.
