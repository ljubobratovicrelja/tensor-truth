# Projects Feature — Master Plan

> **FILE PURPOSE:** This is a version-controlled planning document that carries
> context across Claude Code sessions for the Projects epic. It is the single
> source of truth for the feature design and story breakdown.

> **SCOPE NOTE FOR AGENTS:** This document is an **epic-level plan** containing
> many stories. Each story in the breakdown becomes its own Claude Code
> session/plan. Nothing here is "deferred" or "out of scope" — it is all in
> scope, just sequenced across separate tasks. Do not suggest cutting features;
> suggest ordering and dependencies instead.

> **AGENT INSTRUCTIONS:** Before starting any story, read `.agent/context.md`
> in the project root. It contains mandatory code quality workflow, testing
> requirements, and style guidelines that apply to every story.

> **MEMORY:** As you implement stories from this plan, record lessons learned,
> pitfalls, and patterns in the Claude Code project memory at
> `~/.claude/projects/-Users-relja-Projects-tensor-truth/memory/`. Update
> `MEMORY.md` with links to topic files (e.g., `projects-implementation.md`).

## Vision

Introduce **Projects** — persistent knowledge workspaces that sit between
presets (config templates) and sessions (individual chats). A project owns
built vector indexes and spawns sessions that reuse them.

```
Project
├── Config (model, params, system prompt — like a preset)
├── Knowledge Sources (fetched docs + built indexes — done once)
│   ├── Library modules (from sources.json catalog)
│   ├── Papers / Books (from sources.json catalog)
│   ├── Uploaded files (PDFs, plain text, markdown)
│   └── Hyperlink-based docs (URL + context description)
├── Session A (chat)
├── Session B (chat)
└── ...
```

## Core Principles

1. **No double implementation** — API endpoints and CLI commands MUST share the
   same core functions. The CLI parses args then calls core logic; the API
   receives requests then calls the **same** core logic. Specifically:
   - Doc fetching: both call `scrape_library()`, `fetch_arxiv_paper()`,
     `fetch_book()` etc. from `scrapers/*` — NOT `fetch_sources.main()`
     (which is a CLI entry point that parses `sys.argv`).
   - Index building: both call `build_module()` from `indexing/builder.py`
     — NOT `build_db.main()`.
2. **Shared indexes** — Two projects using `pytorch_2.9` reference the same
   global index in `~/.tensortruth/indexes/`. Building is idempotent.
3. **Incremental builds** — Adding a source to a project only builds what's new.
4. **Sessions belong to a project** (or to "Unorganized" for quick chats).
5. **Build-once, chat-many** — Indexes built at project level, not per-session.
6. **Projects replace presets** — Presets are removed. Projects subsume their
   role (config template + knowledge sources). Simplifies the mental model.

## UI Direction — Claude-inspired, adapted to RAG

**Inspiration: Claude Projects.** The UX should feel familiar to Claude users.
The key difference: instead of flat file uploads, we have vector-indexed
knowledge sources (libraries, papers, books, PDFs) that need fetching + building.

### Sidebar navigation

```
┌──────────────────┬───────────────────────────────────────────┐
│ + New Chat       │                                           │
│                  │                                           │
│ Chats            │  (main content area — see below)          │
│ Projects  <--    │                                           │
│                  │                                           │
│ ─────────────    │                                           │
│ Starred          │                                           │
│  Session A       │                                           │
│  Session B       │                                           │
│ Recents          │                                           │
│  Session C       │                                           │
│  Session D       │                                           │
│  ...             │                                           │
└──────────────────┴───────────────────────────────────────────┘
```

- Top-level nav: **Chats** (flat session list, like today) and **Projects**.
- Clicking "Projects" switches main content to the projects view.
- Clicking "Chats" shows the current flat session list (backwards-compatible).

### Projects listing (main content when "Projects" selected)

```
┌─────────────────────────────────────────────────────────────┐
│  Projects                          [+ New Project]          │
│                                                             │
│  [Search projects...]                                       │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ DL Research     │  │ CV Pipeline     │                  │
│  │                 │  │                 │                  │
│  │ Updated 2d ago  │  │ Updated 1w ago  │                  │
│  └─────────────────┘  └─────────────────┘                  │
│  ┌─────────────────┐                                        │
│  │ NLP Basics      │                                        │
│  │                 │                                        │
│  │ Updated 3w ago  │                                        │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

### Create project (centered form, minimal)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Create a new project                                       │
│                                                             │
│  What are you working on?                                   │
│  ┌───────────────────────────────┐                          │
│  │ Name your project             │                          │
│  └───────────────────────────────┘                          │
│                                                             │
│  What are you trying to achieve?                            │
│  ┌───────────────────────────────┐                          │
│  │ Describe your project...      │                          │
│  │                               │                          │
│  └───────────────────────────────┘                          │
│                                                             │
│          [Cancel]  [Create project]                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

- Just name + description to start. Sources added after creation.

### Inside a project (chat-first, knowledge sidebar)

```
┌──────────────────┬──────────────────────────────┬──────────────────┐
│ <- All projects  │                              │ Config         ✎ │
│                  │  "DL Research"    ... ☆      │ System prompt... │
│ + New Chat       │                              │ Model: qwen3:8b  │
│                  │  ┌──────────────────────┐    │                  │
│ Sessions:        │  │ Ask about your docs..│    │ Knowledge      + │
│  Session A       │  │                      │    │ ┌──────────────┐ │
│  Session B       │  │ ⚙ 🗂 model v    ➤    │    │ │pytorch_2.9 ✓ │ │
│  Session C       │  └──────────────────────┘    │ │numpy_2.3   ✓ │ │
│                  │                              │ │3 papers    ✓ │ │
│                  │  Start a chat to keep        │ │              │ │
│                  │  conversations organized     │ └──────────────┘ │
│                  │  and re-use project          │                  │
│                  │  knowledge.                  │ Status: Ready    │
│                  │                              │ [Build indexes]  │
└──────────────────┴──────────────────────────────┴──────────────────┘
```

- **Center**: Familiar chat input box — identical to current welcome page.
  Starting a chat creates a session within the project.
- **Right sidebar**: Project config (system prompt, model) + knowledge sources.
  This is where sources are added/removed and build status is shown.
  Analogous to Claude's "Instructions" + "Files" panels.
- **Left sidebar**: Narrows to project-scoped sessions + back navigation.

### Key UX principles

- Creating a project is lightweight (name + description only).
- Knowledge sources are added *inside* the project, not during creation.
- The chat input is always front-and-center — projects don't feel "heavy".
- Right sidebar for knowledge is collapsible / hideable.
- Build status is visible but not blocking — you can chat without indexes.

## Knowledge Sources — types, naming, and ingestion

**Terminology:**
- **"Knowledge Sources"** — user-facing umbrella term for everything that feeds
  the RAG pipeline.
- **"Documents"** — uploaded files and URL-fetched content. Stored per-scope
  (project or session). Indexed by `DocumentIndexBuilder`.
- **"Catalog modules"** — pre-defined libraries/papers/books from
  `sources.json`. Stored in global shared indexes (`~/.tensortruth/indexes/`).
  Built by `build_module()` from `indexing/builder.py`. These are a
  tensor-truth-level concept: curated, versioned, reusable across projects.

Three source types, all available at both project and session scope:

| Type | Input | Ingestion | Index location |
|---|---|---|---|
| **File upload** | PDF, `.txt`, `.md` | Upload → convert → chunk → index | Scope-local (`{scope}/index/`) |
| **Hyperlink** | URL + user-provided context | Fetch URL → combine with context → chunk → index | Scope-local (`{scope}/index/`) |
| **Catalog module** | Pick from sources.json | `scrapers/*` → `build_module()` | Global shared (`indexes/{model}/`) |

URL fetching reuses the existing scraper implementations — same core functions
used by both CLI and API (no separate "simple fetch" path).

### Adding sources is always available

Sources can be added **at any time** — during project setup, mid-chat, whenever.
Adding new sources triggers an index build for only the new material (incremental).
The user can continue chatting while a build runs in the background; once complete,
new indexes are hot-loaded into subsequent queries.

### **BUILD PROGRESS UI IS A MUST**

Building vector indexes is slow (seconds to minutes). The user MUST see:
- What is being built (source name)
- Progress indication (progress bar or spinner with stage: fetching → converting → indexing)
- Completion notification
- Error state with retry option

Progress must be visible both in the right sidebar knowledge panel AND as a
non-blocking indicator in the chat area (e.g., subtle banner or toast).

## Unified Document Management — no double implementation

**Critical constraint:** The "Documents" UI used inside a chat (current per-session
PDFs) and the "Knowledge" panel in the project right sidebar MUST share the same
frontend components and the same backend services. The only difference is **scope**.

### Current state (investigation results)

The existing document stack is structurally close to scope-agnostic but has
several concrete couplings that must be addressed:

| Layer | Current (session-only) | Generalization needed |
|---|---|---|
| `pdf_handler.py` | Operates on `scope_dir/pdfs/` + `scope_dir/markdown/` | Mostly generic (takes a dir), but has session-specific metadata headers and logging. Update metadata source labels and docstrings. |
| `session_index.py` | `SessionIndexBuilder(session_id, ...)` | Rename → `DocumentIndexBuilder`. **Must decouple from `paths.py`** — currently hardcodes `get_session_index_dir()` / `get_session_markdown_dir()`. Accept explicit `index_dir` and `markdown_dir` as constructor args instead. |
| `pdf_service.py` | `PDFService(session_id, session_dir, ...)` | Add `scope_type` param ("session" \| "project"). |
| `rag_engine.py` | `load_engine_for_modules(session_index_path=...)` | **Not generic** — accepts a single `session_index_path`. Must change to `additional_index_paths: List[str]` to load both project + session indexes simultaneously. (Story 0 removes `is_llm_only_mode` gating first, simplifying this change.) |
| `rag_service.py` | Config hash uses `has_session_index = bool(...)` | **Bug for projects** — two different project indexes both truthy → no reload. Hash must include actual index paths, not just a boolean. |
| `app_utils/paths.py` | Only has `get_session_*` path functions | Add `get_project_dir()`, `get_project_index_dir()`, etc. or a generic `get_scope_dir(scope_id, scope_type)`. |
| `api/routes/pdfs.py` | Routes under `/sessions/{id}/pdfs` | Add parallel `/projects/{id}/documents` routes, or scope-generic routes. |
| Frontend `PdfDialog` | Takes `sessionId` prop | → `DocumentPanel(scopeId, scopeType)` |
| Frontend `usePdfs` hook | Hardcoded to session API | → `useDocuments(scopeId, scopeType)` |

### Target architecture

```
Frontend:  <DocumentPanel scopeId={id} scopeType="project"|"session" />
              │
              ├── useDocuments(scopeId, scopeType)  ← single hook
              ├── DocumentUploader                   ← handles PDF, text, markdown, URL
              └── DocumentList                       ← shows all sources + build status
                     │
API:       /api/{scopeType}s/{scopeId}/documents/*   ← unified routes
                     │
Service:   DocumentService(scope_id, scope_dir, scope_type)
                     │
Builder:   DocumentIndexBuilder(scope_id, scope_dir)
```

The existing chat header "Documents" button becomes an instance of `DocumentPanel`
with `scopeType="session"`. The project right sidebar knowledge section is the
same `DocumentPanel` with `scopeType="project"`.

When a session belongs to a project, both the project-level and session-level
indexes are loaded as retrievers (project knowledge + session-specific additions).

## Key Existing Code to Reuse

| Purpose | Module | Callable core function |
|---|---|---|
| Fetch docs | `scrapers/*` | `scrape_library()`, `fetch_arxiv_paper()`, `fetch_book()` |
| Build indexes | `indexing/builder.py` | `build_module()` — NOT `build_db.main()` (CLI-only, uses argparse) |
| Sources catalog | `config/sources.json`, `utils/sources_config.py` | loaded at startup |
| Presets | `app_utils/presets.py`, `preset_defaults.py` | GET `/api/presets` |
| Sessions | `services/session_service.py` | CRUD via `/api/sessions` |
| PDF handling | `services/pdf_service.py`, `session_index.py` | `/api/sessions/{id}/pdfs` |

**Important:** `fetch_sources.main()` and `build_db.main()` are CLI entry points
that call `argparse`. They CANNOT be called programmatically. API endpoints must
call the underlying core functions directly. The CLI should be refactored to also
call these same core functions (CLI = argparse → core function; API = route → core function).

## Data Model

### Directory structure

```
~/.tensortruth/
├── sessions/                        (ALL sessions, regardless of project)
│   ├── sessions_index.json
│   └── {session_id}/
│       ├── session.json             (gains optional "project_id" field)
│       ├── documents/
│       ├── markdown/
│       └── index/
├── projects/
│   └── {project_id}/
│       ├── project.json             (includes session_ids list)
│       ├── documents/               (project-level uploaded files)
│       ├── markdown/                (converted content for indexing)
│       └── index/                   (ChromaDB — project-scoped vector index)
└── indexes/                         (global shared catalog module indexes)
    └── {embedding_model_id}/
        └── {module_name}/
```

**Key design decision:** ALL sessions stay in the flat `sessions/` directory,
even those belonging to a project. This means:
- `SessionService` is unchanged — same directory, same lookup, same `get_session_dir()`.
- `get_pdf_service()`, all path functions, and all 6+ call sites work as-is.
- WebSocket handler needs zero changes for session resolution.
- Projects track their sessions via `session_ids` list in `project.json`.
- Session data includes an optional `project_id` field for reverse lookup.
- Deletion cascade: read `session_ids` from `project.json` → delete each session
  from `sessions/` → `rmtree` the project directory. User must be warned
  ("This will delete X sessions").

### project.json schema

```json
{
  "project_id": "uuid",
  "name": "DL Research",
  "description": "Explore deep learning theory and research papers",
  "created_at": "2026-02-14T...",
  "updated_at": "2026-02-14T...",
  "catalog_modules": {
    "pytorch_2.9":  { "status": "indexed" },
    "numpy_2.3":    { "status": "indexed" },
    "papers_dl_architectures_optimization": { "status": "building", "task_id": "uuid" }
  },
  "documents": [
    { "doc_id": "uuid", "type": "pdf",  "filename": "paper.pdf", "status": "indexed" },
    { "doc_id": "uuid", "type": "text", "filename": "notes.md",  "status": "indexed" },
    { "doc_id": "uuid", "type": "url",  "url": "https://...",    "context": "PyTorch CUDA docs", "status": "indexed" }
  ],
  "session_ids": ["uuid1", "uuid2", "uuid3"],
  "config": {
    "model": "qwen3:8b-q8_0",
    "system_prompt": "You are a deep learning research assistant...",
    "embedding_model": "BAAI/bge-m3",
    "temperature": 0.6,
    "context_window": 8192,
    "max_tokens": 4096,
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "reranker_top_n": 5,
    "confidence_cutoff": 0.15,
    "confidence_cutoff_hard": 0.08,
    "balance_strategy": "fixed_top_n",
    "rag_device": "auto",
    "llm_device": "auto"
  }
}
```

Note: `catalog_modules` is an object (not array) with per-module build status.
`updated_at` is bumped on: document add/remove, catalog module change, config
change, or session creation within the project.

## Architectural Concerns (must be addressed in relevant stories)

### Frontend routing

Current routes: `/` (WelcomePage) and `/chat/:sessionId` (ChatContainer).
New routes needed:
- `/projects` — project listing
- `/projects/new` — create project form
- `/projects/:projectId` — project view (three-panel layout)
- `/projects/:projectId/chat/:sessionId` — chat within a project

The sidebar content changes based on route context: global session list (Chats)
vs project-scoped session list (inside a project). `AppLayout` must support this.

### Frontend project data flow

The connection between a session and its project on the frontend:

1. `SessionResponse` gains a `project_id` field (Story 1). Currently defined
   in `api/schemas/session.py` (backend) and `api/types.ts` (frontend) — both
   need updating. The `_session_to_response()` helper in `sessions.py` maps it.
2. When `ChatContainer` loads a session and sees `project_id`, it fetches
   `GET /api/projects/{project_id}` to get catalog modules, documents, config.
3. Both `/chat/:sessionId` and `/projects/:projectId/chat/:sessionId` render
   the same `ChatContainer`. The component detects project context from the
   session's `project_id` (not from the URL) and loads project data accordingly.
   The `/projects/...` route exists for natural navigation from within a
   project view — not as the only way to reach project context.
4. Components that need project context (ModuleSelector, DocumentPanel, config
   panel) receive the resolved project data via a hook (e.g.,
   `useProject(projectId)`) that caches the response.

This means the `GET /api/projects/{id}` endpoint (Story 1) must return enough
data for the frontend: catalog modules with status, documents with status,
config, and session_ids.

### Welcome page

The current welcome page shows presets and creates sessions. Since presets are
being removed (Principle #6), the welcome page needs redesigning:
- Welcome page remains at `/` as the default landing.
- Preset section is removed entirely (WelcomePage.tsx lines 268-298, preset
  click handlers, `activePreset`/`isAnimating` state, `.preset-glow` CSS).
- Replaced with a simpler landing: quick-start chat input + link to Projects.
- Sessions created from the welcome page are "unorganized" (no project).

### WebSocket and project context flow

The chat WebSocket (`/ws/chat/{session_id}`) only receives `session_id`.
For a project session, the backend must:
1. Look up the session's `project_id` from session data
2. Load the project's index path + catalog modules
3. Pass both project and session index paths to the RAG engine

No WebSocket protocol changes needed — project context is resolved server-side.
Specifically, `ChatContext.from_session()` in `chat.py` must be updated to also
resolve project data (catalog modules, project index path) from the session's
`project_id`. Project data should be loaded per-connection (not per-message)
and cached for the duration of the WebSocket session.

### Config inheritance

Priority chain: **session params > project config > global config defaults**.
- Session created within a project inherits project config as initial params.
- Session can override individual params after creation.
- Changes to project config do NOT retroactively update existing sessions.

### Session-project binding

Sessions cannot be moved between projects. A session is created within a project
(or as unorganized) and stays there.

### Deletion cascade

Deleting a project: read `session_ids` from `project.json` → delete each
referenced session from `sessions/` → `rmtree` the project directory (config,
documents, indexes). The user must be warned before deletion ("This will
delete X sessions and all project knowledge").

### ModuleSelector behavior inside a project session

The existing `ModuleSelector` (catalog module picker in chat input) stays, but
adapts its display when inside a project session:

```
┌─────────────────────────────┐
│ Project Knowledge            │  (fixed header, not a checklist)
│   📄 paper.pdf               │  ← project-level uploaded docs
│   📄 notes.md                │  ← shown as a flat, non-editable list
│ ─────────────────────────── │
│ Project Modules              │  (pre-selected, greyed out)
│   ☑ pytorch_2.9  (locked)   │  ← defined at project level
│   ☑ numpy_2.3    (locked)   │  ← cannot be unchecked here
│ ─────────────────────────── │
│ Additional Modules           │  (normal checklist — user can add)
│   ☐ opencv_4.11             │  ← can be toggled per-session
│   ☐ papers_nlp_transformers │
│   ...                        │
└─────────────────────────────┘
```

- **Top group**: Non-editable list of project-level uploaded documents/URLs.
  Just informational — shows what custom knowledge the project has.
- **Middle group**: Catalog modules defined at the project level. Shown as
  selected and greyed out (locked). User cannot uncheck them from a session.
- **Bottom group**: All other available catalog modules (from global
  `~/.tensortruth/indexes/`). Normal checklist — user can freely add these
  as session-level additions.

Session-level module additions are stored in the session's own data and are
additive to the project's modules. They do NOT modify the project config.

### RAG engine multi-index loading

When a session belongs to a project, the engine must load:
1. Global catalog module indexes (from project's `catalog_modules` list)
2. Project-scoped document index (`~/.tensortruth/projects/{id}/index/`)
3. Session-scoped document index (`~/.tensortruth/sessions/{id}/index/`)

All three are passed as retrievers to `MultiIndexRetriever`.

---

## Story Breakdown (high-level — to be expanded)

*Each item becomes its own Claude Code session/plan. Stories are grouped into
phases. Within a phase, stories with no dependency arrow between them can be
run in parallel.*

### Phase 0: Pre-epic cleanup

**Story 0. Remove `is_llm_only_mode` branching**
- `is_llm_only_mode` is a code smell. RAG is not a "mode" — it's just a
  pipeline step that enriches the context prompt. If no indexes/modules are
  present, retrieval returns nothing and the synthesizer works off the base
  prompt. No special branching needed.
- Currently checked in `chat_service.py` (`is_llm_only_mode()` line 177) and
  `chat.py` (`ChatContext.is_llm_only_mode` line 54). Both gate whether the
  RAG pipeline runs at all, creating fragile branching that breaks every time
  the index configuration changes (e.g., adding project-level sources).
- Remove the mode check entirely and all code paths that exist solely to
  support it. Always run through the same code path. If there are no
  retrievers, retrieval is a no-op and the synthesizer proceeds with the
  base prompt alone. The implementer should trace the full call path from
  the mode check through to the engine to ensure zero-retriever scenarios
  are handled gracefully end-to-end (no crashes, sensible defaults).
- This simplifies Story 2 significantly — the signature cascade no longer
  needs to worry about mode checks reacting to the new `additional_index_paths`.
- Must land before Phase 1 starts.
- Tests: verify that chat works with zero modules/indexes (no regression),
  and that adding indexes mid-session still picks them up.

### Phase 1: Foundation

**Story 1. Backend: Project data model + CRUD API**
- `project_service.py` following `session_service.py` pattern.
- Storage at `~/.tensortruth/projects/{id}/`.
- Path functions in `paths.py` (`get_project_dir`, `get_project_index_dir`, etc.)
- API routes:
  - `POST /api/projects` — create project. Schema: `ProjectCreate` (name,
    description) → `ProjectResponse`.
  - `GET /api/projects` — list projects → `ProjectListResponse`.
  - `GET /api/projects/{id}` — get project → `ProjectResponse` (includes
    catalog_modules with status, documents with status, config, session_ids).
  - `PATCH /api/projects/{id}` — update name, description, config →
    `ProjectResponse`. Schema: `ProjectUpdate`.
  - `DELETE /api/projects/{id}` — delete project + cascade delete sessions.
  - `POST /api/projects/{id}/sessions` — create session within a project.
    Creates the session (with project config as defaults), appends session_id
    to project's `session_ids`, sets `project_id` on the session.
  - `GET /api/projects/{id}/sessions` — list project's sessions.
- Pydantic schemas (in `api/schemas/project.py`):
  - `ProjectCreate`: `name: str`, `description: str`.
  - `ProjectUpdate`: all optional — `name`, `description`, `config`.
  - `CatalogModuleStatus`: `status: str`, `task_id: Optional[str]`.
  - `DocumentInfo`: `doc_id: str`, `type: str`, `filename: Optional[str]`,
    `url: Optional[str]`, `context: Optional[str]`, `status: str`.
  - `ProjectResponse`: `project_id`, `name`, `description`, `created_at`,
    `updated_at`, `catalog_modules: Dict[str, CatalogModuleStatus]`,
    `documents: List[DocumentInfo]`, `session_ids: List[str]`,
    `config: Dict[str, Any]`.
  - `ProjectListResponse`: `projects: List[ProjectResponse]`.
- Add `project_id: Optional[str]` field to session data model, `SessionResponse`
  schema (`api/schemas/session.py`), and frontend `types.ts`.
- Session deletion bookkeeping: the route handler in `api/routes/sessions.py`
  (not `SessionService` itself) checks the session's `project_id` before
  deletion. If present, loads the project via `ProjectService` and removes
  the session_id from the project's `session_ids` list. This keeps
  `SessionService` decoupled from projects. Add `get_project_service()` as
  a dependency to the delete endpoint.
- All `ProjectService` mutation methods must bump `updated_at`.
- Config inheritance in session creation: session params > project config >
  global config defaults. Study `SessionService._apply_config_defaults()`
  to understand how global defaults are currently merged — the project
  session creation endpoint must insert project config into the merge chain
  without modifying `SessionService` itself.
- **Module naming convention:** Before designing the `catalog_modules`
  schema, read `config/sources.json`, `api/routes/modules.py` (the
  `list_modules` endpoint), and `rag_engine.py` (module → index path
  resolution) to understand how module names flow through the system. The
  schema must use the same naming convention.
- Unit tests for project CRUD, session-project bookkeeping, and config
  inheritance.

**Story 2. Backend: Session-project association + RAG engine**
- Change `session_index_path` → `additional_index_paths: List[str]` across
  the full call chain (signature cascades through 4 files):
  - `rag_engine.py`: `load_engine_for_modules()`
  - `rag_service.py`: `_compute_config_hash()`, `load_engine()`, `needs_reload()`
  - `chat_service.py`: `execute()`, `query()`
  - `chat.py`: `ChatContext` dataclass + `from_session()` + WebSocket handler
- Fix `RAGService` config hash to include actual index paths (not `bool`).
- Dependency injection: add `get_project_service()` to `api/deps.py`
  (singleton, same pattern as `get_session_service()`). Inject via
  `Depends(get_project_service)` in the WebSocket handler and REST chat
  endpoint. Note that `get_pdf_service()` is NOT a Depends — it's called
  directly with a runtime `session_id`. `ProjectService` is a singleton
  (like `SessionService`), so it CAN be a proper Depends.
- Update `ChatContext.from_session()` to accept `ProjectService`, resolve
  project data from session's `project_id` (catalog modules, project index
  path), and populate `additional_index_paths` with both project and
  session index paths. Cache per-connection.
- **Critical:** The REST chat endpoint uses `ChatContext.from_session()`,
  but the WebSocket handler builds context **inline** (it extracts modules,
  params, and session_index_path directly without calling `from_session()`).
  Both code paths must be updated to resolve project context. Consider
  refactoring the WebSocket handler to also use `from_session()` to avoid
  divergence.
- Merge project catalog modules (where built/indexed) with session modules
  at query time. This is a set union computed on-the-fly — do not persist
  the merged list back to session data.
- Note: this story uses the existing `PDFService` / `get_pdf_service()` for
  session index paths. Project index paths come from `paths.py` functions
  directly. The rename to `DocumentService` happens in Story 4.
- Regression test for config hash fix.
- Depends on: Stories 0, 1.

**Story 3. Backend: Async task runner infrastructure**
- Background task execution framework for long-running operations (doc
  fetching, index building). Needed before any API endpoint can trigger
  `scrape_library()` or `build_module()` without blocking.
- Task state: in-memory dict keyed by task ID (we're single-server; tasks
  are lost on restart, which is acceptable — builds can be re-triggered).
- State machine per task: `pending → running → completed | error`, with
  optional progress percentage and stage label.
- Polling endpoint: `GET /api/tasks/{task_id}` returns current state.
  SSE can be added later if polling proves insufficient.
- Response contract: all endpoints that create a task (Story 6 and beyond)
  MUST return `{ task_id: str }` in their response body so the frontend
  can poll progress.
- Concurrency: one build at a time (queue additional requests).
- Does NOT include the frontend progress UI (that's Story 10).
- Depends on: nothing (can start in parallel with Story 1).

### Phase 2: Backend Documents + Frontend Shell (parallelizable)

**Backend track:**

**Story 4. Generalize document backend**
- Rename `session_index.py` → `document_index.py`, class
  `SessionIndexBuilder` → `DocumentIndexBuilder`. Decouple from `paths.py` —
  accept explicit `index_dir`/`markdown_dir` constructor args. Update all
  imports (`pdf_service.py`, etc.).
- Rename `PDFService` → `DocumentService`. Add `scope_type` param.
- On-disk directory stays `documents/` for new sessions/projects. Existing
  sessions with `pdfs/` are supported via fallback (check both paths).
- Update `pdf_handler.py` metadata headers (`# Source: Session Upload` →
  scope-aware label) and logging.
- Add text/markdown file ingestion (simpler than PDF — no conversion needed,
  just chunk and index).
- Unit tests.
- Depends on: Story 1 (for project path functions).

**Story 5. Backend: URL/hyperlink document ingestion**
- New document type: user provides URL + context description.
- Fetch URL content (reuse scraper infrastructure for HTML extraction).
- Combine fetched content with user-provided context description.
- Chunk and index via `DocumentIndexBuilder`.
- Wire into `DocumentService` alongside PDF and text types.
- Depends on: Story 4.

**Story 6. Backend: Document management API for both scopes**
- Wire `DocumentService` to both `/sessions/{id}/documents` and
  `/projects/{id}/documents` routes (parallel route registrations in FastAPI).
- Catalog module add/remove endpoints for projects (validates against
  `sources_config.py`, triggers async build via Story 3's task runner).
  The add response MUST include `task_id` so the frontend can poll build
  progress. The `task_id` is also stored in the module's `project.json`
  entry during the build (see schema above).
- Rename `get_pdf_service()` in `deps.py` to
  `get_document_service(scope_id, scope_type)`. Grep for all call sites
  (`get_pdf_service`, `PDFService`) across both `api/routes/` and
  `api/deps.py` to ensure complete coverage — there are usages in
  `chat.py`, `pdfs.py`, and `deps.py`.
- Depends on: Stories 3, 4.

**Frontend track (can run in parallel with Stories 4-6):**

**Story 7. Frontend: Routing + sidebar navigation + projects listing**
- Add React Router routes: `/projects`, `/projects/new`,
  `/projects/:projectId`, `/projects/:projectId/chat/:sessionId`.
- Dynamic sidebar: global session list (Chats view) vs project-scoped
  session list (inside a project). `AppLayout` supports both via route context.
- "Projects" nav item in sidebar, card grid listing page with search.
- Depends on: Story 1 (needs project CRUD API).

**Story 8. Frontend: Project creation + project view**
- Create project form (name + description, centered).
- Inside-project layout: left sidebar (project sessions + back nav), center
  (chat input), right sidebar (config + knowledge panel).
- Config editing in right sidebar (system prompt, model, params).
- `ModuleSelector` adaptation: locked project modules + free session additions
  (see "ModuleSelector behavior" section above).
- Depends on: Story 7.

### Phase 3: Unified Documents + Progress

**Story 9. Frontend: Unified DocumentPanel component**
- Replace `PdfDialog` with scope-generic
  `DocumentPanel(scopeId, scopeType)`. Supports PDF, text/md, and URL
  uploads. Shared between project right sidebar and session header
  "Documents" button. Single implementation, two scopes.
- Rename `usePdfs` → `useDocuments(scopeId, scopeType)`.
- Rename frontend API client functions accordingly.
- Depends on: Stories 6 (backend API), 8 (project view).

**Story 10. Frontend: Build progress UI**
- Progress bar in DocumentPanel for active builds.
- Non-blocking indicator in chat area (banner/toast) when builds complete.
- Build status polling or SSE streaming (connects to Story 3's progress API).
- Completion notification, error state with retry.
- **This is the critical UX path — indexes take seconds to minutes.**
- Depends on: Stories 3 (task runner), 9 (DocumentPanel).

### Phase 4: Cleanup

**Story 11. Preset removal + welcome page redesign**
- Remove preset system entirely. Files to delete/modify:
  - Backend: `app_utils/presets.py`, `preset_defaults.py`,
    `api/routes/modules.py` (remove `/api/presets` and
    `/api/presets/favorites` endpoints), `app_utils/paths.py`
    (`get_presets_file()`), `app_utils/__init__.py` (remove preset
    re-exports from `__getattr__` and `__all__`).
  - Frontend: `api/modules.ts` (`listPresets`, `listFavoritePresets`),
    `api/types.ts` (`PresetInfo`, `PresetsResponse`),
    `hooks/useModules.ts` (`usePresets`, `useFavoritePresets`),
    `components/welcome/WelcomePage.tsx` (preset buttons, click handler,
    `activePreset`/`isAnimating` state), `index.css` (`.preset-glow`),
    `lib/constants.ts` (preset query keys).
- Redesign welcome page: remove presets, add Projects quick-access.
- Existing unorganized sessions remain in flat "Chats" list.
- Depends on: Stories 7-8 (projects UI must exist first).

## Testing Strategy

Each story includes unit tests for new/modified backend services. Key areas:
- Chat works with zero modules/indexes after `is_llm_only_mode` removal (Story 0)
- Project CRUD service (Story 1)
- Config inheritance chain: session > project > global (Story 1)
- RAG engine multi-index loading (Story 2)
- Config hash regression — switching projects triggers reload (Story 2)
- DocumentIndexBuilder decoupled from paths.py (Story 4)
- Document type ingestion: PDF, text/md, URL (Stories 4-5)
- Scope-generic document API routes (Story 6)

Manual testing is done collaboratively after each story lands.

---

*Last updated: 2026-02-14*
