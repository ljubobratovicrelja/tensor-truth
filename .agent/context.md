# Tensor-Truth: AI Agent Context Guide

**Purpose**: Guide AI coding agents working on this local RAG pipeline project.

---

## CRITICAL INSTRUCTIONS FOR AI AGENTS

### Before Implementing Anything (MANDATORY)

**ALWAYS check for existing implementations first:**

1. **Understand project structure**: Use Glob tool to explore module organization
2. **Search for similar functions**: Use Grep tool to search for patterns across the codebase
3. **Read existing utilities**: Check `utils/`, `core/`, and `app_utils/` for reusable functions
4. **Avoid duplication**: If similar functionality exists, reuse or extend it rather than reimplementing

**Why this matters**: This codebase has many utilities already implemented. Creating duplicate functions wastes effort and creates maintenance burden.

### Code Quality Workflow (MANDATORY)
**After editing any Python code, you MUST:**

1. **Format**: Run `python scripts/format.py <files>` (isort + black)
2. **Lint**: Run `python -m flake8 <files>` and fix ALL issues
3. **Test**: Run relevant pytest tests and ensure they pass

**Never skip these steps.** This applies to all `.py` files in `src/`, `tests/`, and `scripts/`.

### Testing Requirements
**When creating new code, write tests:**

- **Unit tests** in `tests/unit/` for individual functions/classes
- **Integration tests** in `tests/integration/` for multi-component workflows (use `@pytest.mark.integration`)
- Mock external dependencies (file I/O, APIs, databases)
- Test both success and error paths
- Aim for >80% coverage on new code

**Then run tests as part of the Code Quality Workflow above.**

### Code Style Guidelines

**Emoji usage:**
- Use ONLY for functional UI purposes (status indicators, interactive buttons)
- NEVER in logs, comments, docstrings, CLI output, or error messages
- Rationale: Prevents encoding issues, improves clarity

**Documentation:**
- Do NOT create/modify README.md or other docs unless explicitly requested
- Keep inline comments minimal and focused on "why", not "what"

---

## Project Overview

Tensor-Truth is a **local RAG (Retrieval-Augmented Generation) pipeline** for reducing LLM hallucinations by grounding responses in indexed technical documentation and research papers.

**Core approach**: Hierarchical node parsing with auto-merging retrieval and cross-encoder reranking.

**Stack**: LlamaIndex orchestration, ChromaDB vectors, HuggingFace embeddings (BAAI/bge-m3), BGE rerankers, Ollama LLM inference, Streamlit UI.

## Architecture Overview

**Key directories:**
- `src/tensortruth/` - Main application code
  - `app.py` - Streamlit UI entry point
  - `rag_engine.py` - RAG retrieval and engine orchestration
  - `cli.py` - CLI command routing
  - `build_db.py` - Vector index builder (ChromaDB)
  - `fetch_sources.py` - Documentation and paper fetching
  - `pdf_handler.py` - Session PDF upload/conversion
  - `session_index.py` - Per-session vector indexing
  - `utils/` - **Shared utilities (CHECK HERE FIRST)** - chat, PDF, metadata, web content handlers
  - `core/` - **System utilities** - VRAM, Ollama, device detection
  - `app_utils/` - **Streamlit helpers** - commands, presets, sessions
- `config/sources.json` - Source configurations (libraries + papers)
- `tests/` - Test suite (unit, integration)
- `~/.tensortruth/` - User data (sessions, presets, indexes)

**Before implementing new utilities:**
- `utils/` likely has what you need: PDF ops, chat formatting, metadata extraction, web fetching
- `core/` for hardware/system operations
- `app_utils/` for Streamlit-specific helpers
- Run `tree src/tensortruth/utils/ src/tensortruth/core/ src/tensortruth/app_utils/` to see what's available

**Dependency organization:**
- Core install: Full RAG pipeline, PDF processing, web content fetching, Streamlit UI
- Dev install (`[dev]`): Adds pytest, black/isort/flake8, mypy, hypothesis, coverage

## Critical Architectural Patterns

### Background Engine Loading
**Pattern**: Thread-based async loading to keep UI responsive
- Load ChromaDB indexes, embedder, reranker, LLM in background thread
- Use `threading.Event` for synchronization
- First query waits for engine with timeout
- Preserve recent chat history for context continuity

**Key locations**: `app.py` engine loading, session state management

### Two-Phase RAG Query Flow
**Pattern**: Separate retrieval from generation
1. **Retrieval phase** (with spinner) - Query → MultiIndexRetriever → Auto-merge → Rerank
2. **Generation phase** (streaming) - Prompt + context → LLM → Token stream

**Rationale**: Clear user feedback, optimal UX for local LLM latency

### Multi-Index Parallel Retrieval
**Pattern**: ThreadPoolExecutor for concurrent index queries
- Retrieve from permanent knowledge bases + session PDFs in parallel
- Aggregate and deduplicate nodes
- Use LRU caching for repeated queries
- Adaptive `similarity_top_k` (typically 2x reranker output count)

**Key locations**: `rag_engine.py::MultiIndexRetriever`

### Per-Session PDF Isolation
**Pattern**: Session-scoped vector indexes
- Each session has own PDF collection and ChromaDB index
- Session cleanup removes all associated files (PDFs, indexes, markdown)
- Metadata caching to avoid re-extraction on rebuild
- CPU embedding for session PDFs to preserve VRAM

**Key locations**: `pdf_handler.py`, `session_index.py`, `app_utils/paths.py`

### Metadata Extraction Strategy
**Fallback chain** (tried in order):
1. **Explicit metadata** - YAML headers, PDF metadata dict
2. **LLM extraction** - Use local LLM to extract from content (~500-1000ms)
3. **Filename fallback** - Sanitized filename as last resort

**Always cache extracted metadata** to avoid duplicate extractions.

**Key locations**: `utils/metadata.py`

### Command System Architecture
**Pattern**: Abstract `Command` base class + registry
- Extensible slash command system in chat interface
- Commands return optional state modifier functions
- State mutations executed after display

**When adding commands**: Create subclass, register, update help text

**Key locations**: `app_utils/commands.py`

### No Hard Confidence Cutoffs
**Design principle**: Always return retrieved nodes, show soft warnings when confidence is low
- Inject low-confidence prompt template instead of filtering
- Let LLM acknowledge uncertainty rather than blocking response
- User decides whether to trust the response

**Key locations**: `rag_engine.py` prompt templates and confidence logic

## Platform-Aware Device Placement

**Default strategy:**
- **Apple Silicon (MPS)**: MPS for RAG components, GPU for LLM
- **CUDA available**: CPU for RAG (preserve VRAM), GPU for LLM
- **CPU-only**: CPU for all components

**Rationale**: LLM is largest VRAM consumer; keep embedder/reranker on CPU when GPU memory is limited.

**Key locations**: `core/device.py`, `rag_engine.py` factory functions

## Session State Management

**Critical session_state variables** (`app.py`):
- `chat_data` - Sessions dictionary and current session ID
- `mode` - Current UI mode ("setup" or "chat")
- `engine` - Loaded RAG engine instance
- `loaded_config` - Configuration hash for cache invalidation
- `engine_loading` - Flag for background loading state
- `simple_llm` - Ollama instance for no-RAG mode

**Session data structure** (persisted in `~/.tensortruth/`):
- Each session: title, selected modules, model params, messages
- PDF documents: filename, status, display_name, metadata
- Per-session indexes: isolated ChromaDB for uploaded PDFs

## Hierarchical Chunking Strategy

**Default chunk sizes**: 2048 (parent) / 512 (medium) / 128 (leaf)
- Embed only leaf nodes for efficiency
- Auto-merging retriever reconstructs parent chunks when child chunks match
- Sizes must fit embedding model context window (bge-m3 = 8192 tokens)

**When modifying**: Consider document type (code docs vs papers), test retrieval quality after changes.

**Key locations**: `build_db.py`, `session_index.py`

## CLI Lazy Import Pattern

**Pattern**: Import heavy dependencies only when needed
- Keep startup time fast for simple commands
- Avoid loading LlamaIndex/ChromaDB for info commands

**Example**: `cli.py` imports `fetch_sources` only when running fetch command

## Testing Strategy

**Test markers** (defined in `pyproject.toml`):
- `unit` - Fast, isolated tests
- `integration` - Local resources (Ollama, filesystem)
- `e2e` - Full system tests
- `slow`, `requires_gpu`, `requires_ollama`, `requires_network` - Conditional execution

**Key patterns:**
- Mock external services (Ollama API, web requests)
- Use `tmp_path` fixtures for isolated filesystem ops
- Property-based testing with Hypothesis for edge cases

## Key Design Decisions

**Architectural choices:**
- **ChromaDB PersistentClient** - Single-process, single-user local deployment
- **Ollama backend only** - No cloud API integration
- **Streamlit stateful UI** - Threading for background tasks, session_state for persistence
- **No authentication** - Assumes trusted local network, sessions in plaintext

**Operational assumptions:**
- Ollama server must be running and accessible
- GPU optional but recommended (CPU fallback for all components)
- Document sources converted to markdown before indexing
- Users can build indexes from `config/sources.json`

## Performance Considerations

**Query latency factors:**
- Retrieval speed scales linearly with index count (parallel retrieval helps)
- Reranking overhead depends on node count (similarity_top_k × num_indexes)
- LLM first token time dominates user-perceived latency

**Memory usage:**
- LLM is largest consumer (especially for 7B+ models)
- Embedder and reranker models in VRAM/RAM
- Multiple indexes increase footprint linearly

**Scaling tips:**
- Reduce active indexes when possible
- Preload models on startup to avoid first-query delays
- Use CPU for RAG components when VRAM limited

## Common Pitfalls

**ChromaDB on Windows:**
- Explicitly close PersistentClient connections
- Windows file locking can prevent index deletion

**Session index rebuilds:**
- Full rebuild required on PDF deletion
- Preserve metadata cache to avoid re-extraction

**Engine loading race conditions:**
- Always check `engine_loading` flag before querying
- Use event synchronization for thread safety

**Chunking configuration:**
- Chunk sizes must be hierarchical (large > medium > small)
- Total chunk size must fit in embedding model context window

## Development Setup

```bash
# Setup environment
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest -v

# Format and lint
python scripts/format.py <files>
python -m flake8 <files>

# Start app (requires Ollama running)
tensor-truth
```

---

**For implementation details, always read the code first:**
- Component behavior → Read the module
- Configuration schema → Check `pyproject.toml` and `config/sources.json`
- Test patterns → Examine existing tests in `tests/`
- User features → See `README.md`
