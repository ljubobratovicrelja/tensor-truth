# Tensor-Truth: Claude Context Document

**Last Updated**: 2025-12-17
**Version**: 0.1.9 (PDF Ingestion Feature)
**Python**: 3.11+

---

## ⚠️ CRITICAL INSTRUCTIONS FOR CLAUDE ⚠️

### Document Reading Protocol
**WHEN INSTRUCTED TO "READ CLAUDE.MD" WITH NO FURTHER CONTEXT:**
- Assume you've been initialized to perform a specific task within this project
- Parse the document silently to understand the architecture and context
- **Do NOT provide a comprehensive response** explaining your understanding
- **Simply respond in one sentence** that you are ready to begin work

### Document Maintenance Protocol
**WHENEVER YOU COMPLETE A CODING TASK IN THIS PROJECT:**
1. **Before ending the conversation**, review this CLAUDE.md document
2. **Identify any sections** that are now outdated or deprecated due to your changes
3. **Update this document** to reflect:
   - New file structures or architectures
   - Modified functions/classes (line numbers, signatures, behavior)
   - Changed constants, defaults, or configuration patterns
   - New commands, features, or workflows
   - Deprecated approaches that should no longer be used
4. **Update the "Last Updated" timestamp** at the top of this document
5. **Add a brief changelog entry** if significant changes were made

### Documentation Restrictions
**DO NOT create, write, or modify ANY of the following unless explicitly instructed:**
- README.md files
- Technical documentation in markdown format
- API documentation
- User guides or tutorials
- Any other .md files besides CLAUDE.md

**ONLY update CLAUDE.md** as part of your regular workflow to keep it accurate.

### Code Quality Workflow (MANDATORY END-OF-RESPONSE CHECKLIST)
**⚠️ CRITICAL: AT THE END OF EVERY RESPONSE WHERE YOU EDIT CODE ⚠️**

This is a **mandatory 3-step process** that MUST be completed before you finish your response:

#### Step 1: Format Code
```bash
python scripts/format.py <file1> <file2> ...
```
- Runs `isort` (import sorting) and `black` (code formatting)
- Run on ALL files you created or modified in this response

#### Step 2: Run flake8
```bash
python -m flake8 <file1> <file2> ...
```
- **Fix ALL flake8 issues** that appear:
  - Unused imports, undefined names, syntax errors
  - Line length issues (max 100 chars, some exceptions allowed)
  - Complexity warnings where reasonable
- Re-run flake8 until there are **zero issues**

#### Step 3: Run Tests
```bash
pytest tests/unit/test_<your_module>.py -v
pytest tests/integration/test_<your_feature>.py -v
```
- **Fix ALL failing tests** before completing your response
- If tests fail, debug and fix the code or the tests
- Do NOT leave failing tests in the codebase
- Run tests multiple times to ensure they pass consistently

**YOU MUST COMPLETE ALL 3 STEPS IN ORDER BEFORE ENDING YOUR RESPONSE.**

If you skip any step, you have not followed the protocol correctly. This applies to all `.py` files in `src/`, `tests/`, and `scripts/`.

---

### Testing Protocol (When Writing New Code)
**WHENEVER YOU CREATE NEW CODE FILES:**

1. **Unit Tests**: Create tests for individual functions/classes
   - Location: `tests/unit/test_<module_name>.py`
   - Test all public methods, edge cases, error handling
   - Use mocks for external dependencies (file I/O, API calls, databases)
   - Example: `test_pdf_handler.py` for `pdf_handler.py`

2. **Integration Tests**: Create tests for complex systems with multiple components
   - Location: `tests/integration/test_<feature_name>.py`
   - Test end-to-end workflows
   - Use `@pytest.mark.integration` marker
   - Example: `test_pdf_ingestion_pipeline.py` for PDF upload → convert → index flow

3. **Test Coverage**:
   - Aim for >80% coverage on new code
   - Test both success and failure paths
   - Include edge cases (empty inputs, corrupted data, etc.)

**When to write tests:**
- Always for new modules/classes
- Always for complex logic or algorithms
- Always for data processing pipelines
- Always for user-facing features
- Skip only for trivial helpers or one-off scripts

---

## Project Overview

Tensor-Truth is a **local RAG (Retrieval-Augmented Generation) pipeline** for reducing LLM hallucinations by indexing technical documentation and research papers. It's a personal research tool built for local hardware (M1 Max, RTX 3090 Ti), distributed via PyPI and Docker Hub.

**Core Value Proposition**: Ground LLM responses in source material using hierarchical node parsing with auto-merging retrieval and cross-encoder reranking.

## Architecture at a Glance

```
tensor-truth/
├── src/tensortruth/
│   ├── app.py                 # Streamlit UI (main entry point)
│   ├── rag_engine.py          # Core RAG logic (retriever, prompts, engine loading)
│   ├── cli.py                 # CLI routing (tensor-truth, tensor-truth-docs, tensor-truth-build)
│   ├── build_db.py            # Vector index builder (ChromaDB + hierarchical nodes)
│   ├── fetch_sources.py       # Unified source fetching (libraries, papers) (NEW v0.1.10)
│   ├── pdf_handler.py         # Session PDF upload/conversion handler (v0.1.9)
│   ├── session_index.py       # Session-scoped vector index builder (v0.1.9)
│   ├── utils/                 # Utility modules (NEW v0.1.10)
│   │   ├── __init__.py        # Re-exports for backward compatibility
│   │   ├── chat.py            # LaTeX conversion, markdown export, thinking parser
│   │   └── pdf.py             # PDF processing utilities
│   ├── core/                  # System utilities (VRAM, Ollama API, device detection)
│   └── app_utils/             # Streamlit helpers (commands, presets, sessions, config)
├── config/
│   └── sources.json           # Unified config: libraries + papers (NEW v0.1.10)
├── tests/                     # Pytest suite (unit, integration)
├── pyproject.toml             # Package definition
└── ~/.tensortruth/            # User data dir (sessions, presets, indexes, config)
```

## Technology Stack

### RAG Pipeline (Core)
- **LlamaIndex**: Orchestration framework (query engines, chat engines, retrievers)
- **ChromaDB**: Vector database (persistent, single-process)
- **HuggingFace Embeddings**: `BAAI/bge-m3` (1024-dim, supports code+text)
- **Rerankers**: BGE cross-encoder models (`bge-reranker-v2-m3`, `bge-reranker-base`)
- **Hierarchical Chunking**: 2048/512/128 token chunks with auto-merging
- **Ollama**: Local LLM inference (DeepSeek-R1, DeepSeek-Coder-V2)

### Web UI
- **Streamlit**: Chat interface, settings, session management
- **Threading**: Async title generation, background engine loading, token streaming

### Data Ingestion
- **BeautifulSoup4 + Markdownify**: Sphinx doc conversion
- **sphobjinv**: Sphinx inventory parsing
- **pymupdf4llm, marker-pdf**: PDF processing
- **arxiv**: ArXiv API client

### Dev/Testing
- **pytest**: Test framework (unit, integration, e2e markers)
- **black, isort, flake8, mypy**: Code quality
- **hypothesis**: Property-based testing

## Dependency Organization

As of v0.1.10, dependencies are organized to support both the main app and unified CLI tools:

### Core Dependencies (pip install tensor-truth)
Includes everything needed to run the main Streamlit app with full functionality:
- RAG pipeline (LlamaIndex, ChromaDB, embeddings, rerankers)
- Web UI (Streamlit, utilities)
- **PDF processing** (pymupdf4llm, marker-pdf, tqdm) - Required for session PDF uploads
- Torch, sentence-transformers, and other ML libraries

### Optional Dependencies
**`[docs]` extra** - For unified source fetching (libraries + papers):
```bash
pip install tensor-truth[docs]
```
Adds:
- `beautifulsoup4`, `markdownify`, `sphobjinv` (Sphinx/Doxygen doc scraping)
- `arxiv` (ArXiv paper fetching)

Enables:
- `tensor-truth-docs --type library <name>` (scrape library documentation)
- `tensor-truth-docs --type papers --category <cat>` (fetch ArXiv papers)

**`[dev]` extra** - For development:
```bash
pip install tensor-truth[dev]
```
Includes `[docs]` plus testing and code quality tools (pytest, black, isort, flake8, mypy, hypothesis)

### Important Notes
- **PDF dependencies are NOT optional** - `pymupdf4llm` and `marker-pdf` are in core dependencies because the session PDF upload feature is part of the main app UI
- The `tensor-truth-build` command works without extras (uses core dependencies only)
- The unified `tensor-truth-docs` command requires `[docs]` extra for both library and paper fetching

## Key Files Deep Dive

### 1. `app.py` (~1208 lines) - Streamlit UI Controller
**Purpose**: Chat interface, session management, RAG integration

**Critical Sections**:
- **Lines 1-92**: Initialization (CSS, config, auto-download indexes from GDrive)
- **Lines 97-242**: Sidebar (session switcher, settings, delete/export dialogs)
- **Lines 247-604**: Setup mode (presets, favorites, manual config, connection settings)
- **Lines 606-1208**: Chat mode (message rendering, engine loading, streaming, commands)

**Key Patterns**:
- **Background Engine Loading** (Lines 653-730): Threading + event synchronization to avoid blocking UI
- **Two-Phase RAG** (Lines 910-1105): Upfront retrieval with spinner → streaming LLM response
- **Command Processing** (Lines 844-867): `/list`, `/load`, `/model`, `/device`, `/conf` handled via `commands.py`
- **No-RAG Fallback** (Lines 1108-1206): Direct Ollama chat when no modules selected
- **Auto-Title Generation** (Lines 890-903): Background async task using `qwen2.5:0.5b`

**Session State Variables**:
- `chat_data`: Sessions dict + current_id
- `mode`: "setup" | "chat"
- `engine`: Loaded RAG engine instance
- `loaded_config`: Tuple of (modules, params) hash for cache invalidation
- `engine_loading`: Boolean flag for background loading
- `simple_llm`: Ollama instance for no-RAG mode

### 2. `rag_engine.py` (~312 lines) - RAG Core
**Purpose**: Retrieval logic, prompts, engine factory

**Key Functions**:
- `load_engine_for_modules(selected_modules, engine_params, preserved_chat_history)` (Lines 233-311)
  - Loads multiple ChromaDB indexes
  - Creates `MultiIndexRetriever` (parallel retrieval across indexes)
  - Configures `CondensePlusContextChatEngine` with custom prompts
  - Adaptive `similarity_top_k` = `reranker_top_n * 2` (e.g., 6 candidates → top 3 after rerank)

- `MultiIndexRetriever` (Lines 193-230)
  - ThreadPoolExecutor for parallel retrieval
  - LRU cache (128 queries) for speed
  - Aggregates nodes from all active indexes

- `get_embed_model(device)`, `get_llm(params)`, `get_reranker(params, device)` (Lines 147-191)
  - Factory functions for RAG components
  - Device placement: `cpu`, `cuda`, `mps`

**Custom Prompts** (Lines 51-144):
- `CUSTOM_CONTEXT_PROMPT_TEMPLATE`: Normal RAG response
- `CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE`: Soft warning when similarity < threshold
- `CUSTOM_CONTEXT_PROMPT_NO_SOURCES`: Fallback to general knowledge
- `CUSTOM_CONDENSE_PROMPT_TEMPLATE`: Follow-up query reformulation

**Design Philosophy**:
- No hard filtering by confidence cutoff (removed in v0.1.6+)
- Soft warnings shown in UI when confidence is low
- Always return nodes to avoid "no data" errors

### 3. `app_utils/commands.py` (~474 lines) - Command System
**Purpose**: Slash command processing (class-based architecture)

**Commands**:
- `/list` | `/status`: Show knowledge base + hardware usage + Ollama runtime
- `/model [name]`: Show/switch models
- `/load <index>`: Mount knowledge base module
- `/unload <index>`: Unmount module
- `/reload`: Flush VRAM, restart engine
- `/device rag <cpu|cuda|mps>`: Move embedder/reranker
- `/device llm <cpu|gpu>`: Move LLM (Ollama)
- `/conf <0.0-1.0>`: Set confidence warning threshold
- `/help`: Command reference

**Pattern**: `Command` abstract base class → `CommandRegistry` → `process_command()`

**State Modifiers**: Functions returned by commands that mutate session state after display (Lines 199, 248, 327, 381)

### 4. `build_db.py` (~132 lines) - Index Builder
**Purpose**: Convert markdown/HTML docs → ChromaDB vector indexes

**Process**:
1. Clean old index (per-module isolation)
2. Load docs from `./library_docs/{module_name}/`
3. Hierarchical parsing (2048/512/128 token chunks)
4. Embed leaf nodes with `BAAI/bge-m3`
5. Persist to `~/.tensortruth/indexes/{module_name}/`

**Usage**:
```bash
tensor-truth-build --modules pytorch numpy
tensor-truth-build --all
tensor-truth-build --chunk-sizes 4096 1024 256
```

### 5. `cli.py` (~65 lines) - CLI Router
**Purpose**: Entry points for all CLI commands

**Commands**:
- `tensor-truth` → Launch Streamlit app
- `tensor-truth-docs` → Unified source fetching (libraries + papers) (requires `[docs]` extra)
- `tensor-truth-build` → Build vector indexes

**Pattern**: Lazy imports to avoid loading heavy dependencies when not needed

### 6. `fetch_sources.py` (~700 lines) - Unified Source Fetching
**Purpose**: Unified CLI for fetching both library documentation and ArXiv papers

**Key Functions**:
- `scrape_library(library_name, config, ...)` - Scrape Sphinx/Doxygen docs
- `fetch_arxiv_paper(arxiv_id, output_dir, converter)` - Fetch single ArXiv paper
- `fetch_paper_category(category_name, category_config, ...)` - Fetch entire paper category
- `list_sources(config)` - List all available libraries and paper categories

**Usage Examples**:
```bash
# List all sources
tensor-truth-docs --list

# Fetch library docs (backward compatible positional args)
tensor-truth-docs pytorch numpy

# Fetch library docs (explicit type)
tensor-truth-docs --type library pytorch

# Fetch papers in a category
tensor-truth-docs --type papers --category dl_foundations

# Fetch specific papers
tensor-truth-docs --type papers --category dl_foundations --ids 1706.03762 1810.04805

# Use marker converter for better math
tensor-truth-docs --type papers --category dl_foundations --converter marker
```

### 7. `utils/` Module - Utility Functions
**Purpose**: Organized utility functions for PDF processing and chat utilities

**Structure**:
- `utils/__init__.py` - Re-exports for backward compatibility
- `utils/chat.py` - Chat/thinking/LaTeX utilities
- `utils/pdf.py` - PDF processing (conversion, TOC extraction, splitting)

**Key Functions**:
- `utils.pdf.convert_pdf_to_markdown()` - PDF → Markdown with pymupdf4llm or marker
- `utils.pdf.convert_with_marker()` - GPU-accelerated PDF conversion (better math)
- `utils.pdf.post_process_math()` - Unicode → LaTeX symbol conversion
- `utils.pdf.clean_filename()` - Sanitize filenames for filesystem
- `utils.pdf.get_pdf_page_count()` - Extract page count from PDF
- `utils.chat.parse_thinking_response()` - Extract <thought> tags from LLM output
- `utils.chat.convert_latex_delimiters()` - \[...\] → $$...$$ for Streamlit
- `utils.chat.convert_chat_to_markdown()` - Export chat history to markdown file

### 8. `config/sources.json` - Unified Source Configs
**Purpose**: Unified configuration for all documentation sources (libraries + papers)

**Structure**:
```json
{
  "libraries": {
    "pytorch": {
      "type": "sphinx",
      "version": "2.9",
      "doc_root": "https://pytorch.org/docs/stable/",
      "inventory_url": "https://pytorch.org/docs/stable/objects.inv",
      "selector": "div[role='main']"
    }
  },
  "papers": {
    "dl_foundations": {
      "type": "arxiv",
      "description": "Deep Learning Foundations...",
      "items": [
        {"title": "...", "arxiv_id": "1512.03385", "url": "..."}
      ]
    }
  }
}
```

**Supported Libraries** (16 total): PyTorch, NumPy, SciPy, Matplotlib, Pandas, Scikit-learn, Seaborn, Requests, Flask, Django, TensorFlow, Transformers, Pillow, SQLAlchemy, NetworkX, OpenCV

**Paper Categories** (8 total): dl_foundations, vision_2d_generative, 3d_reconstruction_rendering, linear_algebra_books, calculus_books, numerical_optimization_books, machine_learning_books, deep_learning_books

### 7. User Data Directory (`~/.tensortruth/`)
**Structure**:
```
~/.tensortruth/
├── chat_sessions.json      # Chat history (messages, params, modules, PDFs)
├── presets.json            # Saved configurations (favorites, descriptions)
├── config.yaml             # Global settings (Ollama URL, etc.)
├── indexes/                # Permanent vector databases (knowledge bases)
│   ├── pytorch/            # ChromaDB instance
│   │   ├── chroma.sqlite3
│   │   ├── docstore.json
│   │   └── ...
│   └── numpy/
└── sessions/               # Per-session data (PDFs, temp indexes)
    └── sess_abc123/
        ├── pdfs/           # Uploaded PDF files
        │   ├── pdf_xyz_attention.pdf
        │   └── pdf_def_bert.pdf
        ├── markdown/       # Converted markdown (for inspection/reindex)
        │   ├── pdf_xyz.md
        │   └── pdf_def.md
        └── index/          # Session-specific ChromaDB vector index
            ├── chroma.sqlite3
            ├── docstore.json
            └── ...
```

**Sessions Structure** (example):
```json
{
  "current_id": "sess_123",
  "sessions": {
    "sess_123": {
      "title": "PyTorch Autograd Deep Dive",
      "modules": ["pytorch"],
      "params": {
        "model": "deepseek-r1:8b",
        "temperature": 0.3,
        "context_window": 4096,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_top_n": 3,
        "confidence_cutoff": 0.3,
        "rag_device": "mps",
        "llm_device": "gpu",
        "system_prompt": ""
      },
      "pdf_documents": [
        {
          "id": "pdf_xyz",
          "filename": "attention.pdf",
          "uploaded_at": "2025-12-17 12:34:56",
          "file_size": 524288,
          "page_count": 12,
          "status": "indexed",
          "error_message": null
        }
      ],
      "has_temp_index": true,
      "messages": [
        {"role": "user", "content": "Explain autograd.Function"},
        {"role": "assistant", "content": "...", "sources": [...], "time_taken": 2.3}
      ]
    }
  }
}
```

## Session-Scoped PDF Ingestion (NEW in v0.1.9)

**Feature**: Upload and index PDFs within a session alongside permanent knowledge bases.

### Architecture

**Per-Session Isolation**: Each session can have its own set of uploaded PDFs with a dedicated vector index stored in `~/.tensortruth/sessions/{session_id}/`.

**Workflow**:
1. User uploads PDF via sidebar in chat mode
2. PDF saved to `~/.tensortruth/sessions/{session_id}/pdfs/`
3. PDF converted to markdown using marker-pdf (fallback to pymupdf4llm)
4. Markdown saved to `~/.tensortruth/sessions/{session_id}/markdown/`
5. Vector index built/rebuilt with HierarchicalNodeParser (2048/512/128 chunks)
6. Index persisted to `~/.tensortruth/sessions/{session_id}/index/`
7. MultiIndexRetriever loads both permanent knowledge bases AND session index
8. Queries retrieve from all indexes in parallel (permanent + session)

### Key Components

**1. PDFHandler** ([pdf_handler.py](src/tensortruth/pdf_handler.py)):
- `upload_pdf(uploaded_file)` → saves PDF, extracts metadata (page count, size)
- `convert_pdf_to_markdown(pdf_path)` → marker-pdf first, pymupdf4llm fallback
- `delete_pdf(pdf_id)` → removes PDF + markdown
- `get_all_markdown_files()` → lists all session markdowns for indexing

**2. SessionIndexBuilder** ([session_index.py](src/tensortruth/session_index.py)):
- `build_index(markdown_files)` → creates ChromaDB index with same logic as build_db.py
- `rebuild_index()` → reindexes after PDF deletion
- `delete_index()` → cleanup on session delete or last PDF removal
- `index_exists()` → checks for valid ChromaDB

**3. RAG Engine Integration** ([rag_engine.py](src/tensortruth/rag_engine.py:233-303)):
- `load_engine_for_modules(..., session_index_path=None)` → new optional parameter
- Lines 282-300: Loads session index if path provided, adds to `active_retrievers`
- Line 236: Updated validation to allow session index without permanent modules

**4. App UI Integration** ([app.py](src/tensortruth/app.py)):
- Lines 124-166: Sidebar "Session Documents" section with upload widget
- Lines 48-110: `process_pdf_upload()` → orchestrates upload → convert → index
- Lines 112-150: `delete_pdf_from_session()` → remove PDF, rebuild/delete index
- Lines 791-802: Config hash includes `has_temp_index` flag for cache invalidation
- Lines 865-878: Engine loading includes session index path

**5. Path Management** ([app_utils/paths.py](src/tensortruth/app_utils/paths.py:43-75)):
- `get_sessions_data_dir()` → `~/.tensortruth/sessions/`
- `get_session_dir(session_id)` → `~/.tensortruth/sessions/{session_id}/`
- `get_session_pdfs_dir(session_id)` → PDF storage
- `get_session_markdown_dir(session_id)` → Converted markdown
- `get_session_index_dir(session_id)` → Vector index

**6. Session Cleanup** ([app_utils/session.py](src/tensortruth/app_utils/session.py:77-91)):
- `delete_session(session_id, sessions_file)` → removes session JSON + entire session directory (PDFs, markdown, index)

### User Experience

**Sidebar UI**:
- Status icons: ⏳ uploading, 🔄 processing, ✅ indexed, ❌ error
- Delete button per PDF (rebuilds index or removes if last PDF)
- Upload widget: accepts .pdf files
- Progress spinners: "Converting...", "Indexing..."

**PDF Status States**:
- `uploading` → PDF being saved
- `processing` → Converting to markdown + building index
- `indexed` → Ready for queries
- `error` → Conversion/indexing failed (error_message field populated)

**Persistence**:
- PDFs and indexes persist across app restarts
- Stored in session JSON as `pdf_documents` array + `has_temp_index` boolean
- Index cached on disk → no rebuild on restart

**Retrieval**:
- Session PDFs appear in sources with 📄 icon (vs 📚 for knowledge bases)
- Parallel retrieval across permanent + session indexes
- Same reranking and confidence logic applies

### Limitations & Design Decisions

**Session-Scoped Only**: PDFs cannot be shared across sessions (intentional isolation).

**Marker-PDF First**: Better for academic papers with math notation, but slower (~10-30s per paper).

**No Inline PDF Display**: Only filename + status shown, no preview/thumbnails.

**Rebuild on Delete**: Removing a PDF triggers full index rebuild (acceptable for 3-5 papers).

**No Page-Level Citations**: Sources show filename, not specific page numbers.

## RAG Pipeline Flow

### Query Execution (Normal Mode with RAG)
```
User Input
    ↓
Condense Query (if follow-up) using Chat History
    ↓
MultiIndexRetriever (parallel across all loaded indexes)
    ↓ [similarity_top_k per index, e.g., 6 nodes]
Auto-Merging Retrieval (parent chunk reconstruction)
    ↓
Cross-Encoder Reranking (top_n=3)
    ↓ [confidence check: best_score < threshold?]
    ├─ YES → Inject LOW_CONFIDENCE prompt
    └─ NO  → Use NORMAL prompt
    ↓
Ollama Streaming (token-by-token)
    ↓
Response + Sources + Timing
```

### Engine Loading Strategy
```
Session Start/Config Change
    ↓
Background Thread: load_engine_background()
    ├─ Preserve last 4 messages from chat history
    ├─ Load embedder (BAAI/bge-m3) on rag_device
    ├─ Load indexes from ChromaDB (per module)
    ├─ Load reranker (BGE-v2-m3) on rag_device
    ├─ Load LLM (Ollama) with llm_device param
    └─ Signal completion via threading.Event
    ↓
UI remains responsive (user can type, messages render)
    ↓
On first query: Wait for engine (timeout=60s)
    ↓
Execute query
```

## Configuration System

### Smart Defaults (Platform-Aware)
- **Apple Silicon (MPS detected)**:
  - `rag_device = "mps"` (embedder + reranker)
  - `llm_device = "gpu"` (Ollama uses MPS automatically)
- **CUDA (nvidia-smi available)**:
  - `rag_device = "cpu"` (save VRAM for LLM)
  - `llm_device = "gpu"`
- **CPU-only**:
  - `rag_device = "cpu"`
  - `llm_device = "cpu"`

### Configurable Parameters
| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `model` | `deepseek-r1:8b` | Any Ollama model | Tested: DeepSeek-R1 (8b/14b/32b), Coder-V2 |
| `temperature` | `0.3` | 0.0-1.0 | Lower = more deterministic |
| `context_window` | `4096` | 2048-32768 | Must match model's context limit |
| `reranker_model` | `BAAI/bge-reranker-v2-m3` | BGE models | v2-m3 (best), base, MiniLM-L6 |
| `reranker_top_n` | `3` | 1-20 | Final nodes sent to LLM |
| `confidence_cutoff` | `0.3` | 0.0-1.0 | Soft warning threshold (not a filter) |
| `rag_device` | `"mps"` or `"cpu"` | cpu/cuda/mps | Hardware for embedder/reranker |
| `llm_device` | `"gpu"` | cpu/gpu | Ollama execution device |
| `system_prompt` | `""` | Text | Custom system instructions |

## Testing Strategy

### Markers (pytest.ini_options)
- `unit`: Fast, isolated (no external deps)
- `integration`: Local resources (Ollama, filesystem)
- `e2e`: Full system tests
- `slow`: Long-running tests
- `requires_gpu`, `requires_ollama`, `requires_network`: Conditional execution

### Test Structure
```
tests/
├── unit/
│   ├── test_core.py           # System utils (VRAM, device detection)
│   ├── test_rag_engine.py     # Retriever, engine loading
│   ├── test_commands.py       # Command processing
│   ├── test_app_utils.py      # Presets, sessions
│   └── test_utils.py          # LaTeX, markdown conversion
├── integration/
│   └── test_ingestion_pipeline.py  # End-to-end indexing
└── conftest.py                # Fixtures (temp dirs, mock indexes)
```

### Key Test Patterns
- **Mocking Ollama**: `requests.post` patches for API calls
- **Temp Indexes**: `tmp_path` fixtures for isolated ChromaDB instances
- **Property-Based**: Hypothesis for LaTeX conversion edge cases

## Common Development Tasks

### Adding a New Library to Index
1. Add entry to `config/sources.json` under `"libraries"` (Sphinx/Doxygen) or implement custom scraper
2. Run `tensor-truth-docs --type library <library_name>` (or just `tensor-truth-docs <library_name>`)
3. Run `tensor-truth-build --modules <library_name>`
4. Restart app, module appears in multiselect

### Adding a New Paper Category
1. Add entry to `config/sources.json` under `"papers"` with `"type": "arxiv"`
2. Run `tensor-truth-docs --type papers --category <category_name>`
3. Run `tensor-truth-build --modules <category_name>`
4. Restart app, category appears in multiselect

### Implementing a New Command
1. Create `Command` subclass in `app_utils/commands.py`
2. Register in `_registry` (bottom of file)
3. Add to `CommandRegistry.get_help_text()`
4. Test via `/help` in chat

### Changing Chunking Strategy
1. Edit `build_db.py:23` (default `chunk_sizes=[2048, 512, 128]`)
2. Rebuild affected indexes with `--chunk-sizes` flag
3. For custom logic: Modify `HierarchicalNodeParser` call

### Debugging RAG Retrieval
1. Enable `verbose=True` in `rag_engine.py:308` (`CondensePlusContextChatEngine`)
2. Check node metadata: `node.metadata['file_name']`, `node.score`
3. Inspect `context_nodes` in `app.py:916` (pre-rerank) and `app.py:1064` (post-rerank)

### Custom Prompts
- Edit `CUSTOM_CONTEXT_PROMPT_TEMPLATE` in `rag_engine.py:51-74`
- For low-confidence variant: `CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE` (Lines 76-97)
- For no-sources fallback: `CUSTOM_CONTEXT_PROMPT_NO_SOURCES` (Lines 100-118)

## Performance Characteristics

### Latency Breakdown (Typical Query on M1 Max)
- **Retrieval** (MultiIndexRetriever): 0.5-1.5s (depends on # of indexes)
- **Reranking** (BGE-v2-m3): 0.2-0.5s
- **LLM First Token** (DeepSeek-R1:8b): 1-2s
- **Total to First Token**: ~2-4s
- **Streaming**: 20-40 tokens/sec

### Memory Usage
- **Embedder (BAAI/bge-m3)**: ~1.2GB VRAM
- **Reranker (BGE-v2-m3)**: ~800MB VRAM
- **LLM (DeepSeek-R1:8b)**: ~5-6GB VRAM/RAM
- **ChromaDB Index**: ~100-500MB per library (in-memory)

### Scaling Limits
- **Max Indexes**: No hard limit, but retrieval latency scales linearly
- **Max Messages**: ChatMemoryBuffer truncates to 3000 tokens (~2-4 turns)
- **Max Nodes**: `similarity_top_k` * num_indexes (e.g., 6 * 5 = 30 pre-rerank)

## Edge Cases & Gotchas

### 1. Engine Loading Race Condition
**Issue**: User sends query before engine finishes loading
**Fix**: `app.py:807-827` waits for `engine_load_event` with 60s timeout

### 2. Empty Retrieval Results
**Issue**: Query matches nothing (rare with reranker's soft scoring)
**Fix**: `app.py:955-981` injects synthetic warning node with score=0.0

### 3. Ollama Connection Failure
**Issue**: Ollama not running or wrong URL
**Symptoms**: Engine load error, LLM timeout
**Fix**: Check `get_ollama_url()` via `/list`, update in Connection Settings

### 4. VRAM OOM on GPU
**Issue**: Embedder + Reranker + LLM exceed VRAM
**Fix**: `/device rag cpu` to offload pipeline, or `/device llm cpu` for LLM

### 5. Index Corruption
**Issue**: ChromaDB errors, missing files
**Fix**: Delete `~/.tensortruth/indexes/{module}/`, rebuild with `tensor-truth-build`

### 6. Streamlit Session State Desync
**Issue**: `st.rerun()` races, widget state mismatch
**Fix**: Use `skip_last_message_render` flag (app.py:743-746) to prevent double-render

## Docker Deployment

### Dockerfile Strategy
- **Base**: `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime` (Python 3.11.4)
- **Install**: `pip install tensor-truth` (no dev/docs extras needed for basic usage)
- **Volume**: `/root/.tensortruth` (sessions, presets, indexes)
- **Port**: 8501 (Streamlit default)
- **Env**: `OLLAMA_HOST=http://host.docker.internal:11434`

### Running
```bash
docker run -d \
  --name tensor-truth \
  --gpus all \
  -p 8501:8501 \
  -v ~/.tensortruth:/root/.tensortruth \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  ljubobratovicrelja/tensor-truth:latest
```

### Building Locally
```bash
docker build -t tensor-truth .
docker run -d --name tensor-truth --gpus all -p 8501:8501 -v ~/.tensortruth:/root/.tensortruth tensor-truth
```

## Version History (Key Changes)

### 0.1.10 (Current)
- **CLI Unification**: Merged `tensor-truth-papers` into `tensor-truth-docs` command
- **Module Reorganization**: Created `utils/` package (chat.py, pdf.py modules)
- **Unified Config**: `sources.json` replaces `api.json` + `papers.json`
- **Breaking Changes**:
  - Removed `tensor-truth-papers` CLI command (use `tensor-truth-docs --type papers` instead)
  - Removed `fetch_paper.py` (logic moved to `fetch_sources.py` + `utils/pdf.py`)
  - Removed `scrape_docs.py` (replaced by `fetch_sources.py`)
  - Removed `utils.py` (replaced by `utils/` module)
- Simplified dependency structure (single `[docs]` extra for both libraries and papers)

### 0.1.9
- **Session-scoped PDF ingestion**: Upload PDFs in chat mode, converted with marker-pdf
- **Temporary vector indexes**: Per-session ChromaDB indexes for uploaded PDFs
- **Parallel retrieval**: Queries fetch from both permanent knowledge bases + session PDFs
- New files: `pdf_handler.py`, `session_index.py`
- Extended `load_engine_for_modules()` with `session_index_path` parameter
- Session cleanup now removes PDFs, markdown, and indexes

### 0.1.8
- Fixed gdown paths for index auto-download
- Docker Hub README improvements

### 0.1.7
- Removed hard confidence cutoff filtering (soft warnings only)
- Improved low-confidence UX

### 0.1.6
- Background engine loading (non-blocking UI)
- Two-phase RAG (upfront retrieval + streaming)
- Command system refactor (class-based)

### 0.1.5
- Auto-title generation with `qwen2.5:0.5b`
- Favorite presets (quick launch)
- `/model` command

## File Reference Quick Lookup

### When Modifying...
| Task | Primary File(s) | Secondary Files |
|------|-----------------|-----------------|
| UI Layout | `app.py` (lines 97-242, 247-604) | `media/app_styles.css` |
| Chat Logic | `app.py` (lines 606-1208) | - |
| RAG Retrieval | `rag_engine.py` (lines 193-230) | - |
| Prompts | `rag_engine.py` (lines 51-144) | - |
| Commands | `app_utils/commands.py` | - |
| Presets | `app_utils/presets.py` | - |
| Sessions | `app_utils/session.py` | - |
| Config | `app_utils/config.py`, `config_schema.py` | - |
| Ollama API | `core/ollama.py` | - |
| System Utils | `core/system.py` | - |
| Indexing | `build_db.py` | - |
| Source Fetching (libraries + papers) | `fetch_sources.py` | `config/sources.json` |
| CLI Routing | `cli.py` | - |
| PDF Processing | `utils/pdf.py` | - |
| Chat Utilities | `utils/chat.py` | - |

## Critical Constants & Globals

```python
# User data directory
USER_DIR = ~/.tensortruth  # Platform-specific (macOS/Linux/Windows)

# Index directory
INDEX_DIR = ~/.tensortruth/indexes

# Session file
SESSIONS_FILE = ~/.tensortruth/sessions.json

# Presets file
PRESETS_FILE = ~/.tensortruth/presets.json

# Config file
CONFIG_FILE = ~/.tensortruth/config.json

# GDrive index download
GDRIVE_LINK = https://drive.google.com/file/d/1jILgN1ADgDgUt5EzkUnFMI8xwY2M_XTu/view

# Default embedder
EMBED_MODEL = "BAAI/bge-m3"

# Default reranker
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Default LLM
DEFAULT_MODEL = "deepseek-r1:8b"

# Chunk sizes (hierarchical)
DEFAULT_CHUNK_SIZES = [2048, 512, 128]

# Chat memory limit
MEMORY_TOKEN_LIMIT = 3000

# Preserved history on reload
MAX_PRESERVED_MESSAGES = 4

# Engine load timeout
ENGINE_LOAD_TIMEOUT_MS = 60000

# Token streaming poll interval
TOKEN_POLL_INTERVAL_MS = 50
```

## Important Assumptions & Design Decisions

### 1. Single-Process Isolation
- ChromaDB uses `PersistentClient` (not server mode)
- No concurrent writes to same index
- Safe for single-user local deployment

### 2. No Authentication/Multi-User
- Streamlit runs without auth
- Docker deployment assumes trusted local network
- Sessions stored in plaintext JSON

### 3. Ollama as LLM Backend
- No OpenAI/Anthropic API integration
- Assumes Ollama server is running and accessible
- Model availability checked via `/api/tags` endpoint

### 4. GPU Optional But Recommended
- CPU fallback for all components
- Embedder/Reranker can run on CPU (slower but functional)
- LLM on CPU is very slow (not recommended for >3B models)

### 5. Pre-Built Indexes from GDrive
- First-run auto-download (~500MB tar.gz)
- Manual build requires `[docs]` extra for scraping tools
- Index format tied to LlamaIndex version (breaking changes possible)

### 6. Streamlit as UI Framework
- Stateful session management via `st.session_state`
- Reruns entire script on interaction (optimized with caching)
- Threading for background tasks (not asyncio)

### 7. Markdown/HTML as Index Source
- Docs converted to Markdown before indexing
- HTML preserved for Doxygen (OpenCV)
- PDFs supported via `pymupdf4llm` or `marker-pdf`

## Troubleshooting Guide

### Symptom: "No models found in Ollama"
**Cause**: Ollama not running or wrong URL
**Fix**:
1. `ollama serve` (start server)
2. Check URL in Connection Settings (app.py:576-604)
3. Test: `curl http://localhost:11434/api/tags`

### Symptom: Engine load timeout (60s)
**Cause**: Large indexes, slow CPU, VRAM bottleneck
**Fix**:
1. `/device rag cpu` (if GPU VRAM full)
2. Reduce active indexes (`/unload <name>`)
3. Check `ollama ps` for running models

### Symptom: Low-confidence warnings on every query
**Cause**: Threshold too high, index mismatch
**Fix**:
1. `/conf 0.1` (lower threshold)
2. Check index contents (may be wrong library)
3. Verify embedder consistency (rebuild if migrated from old version)

### Symptom: Slow first token (>10s)
**Cause**: Ollama model not preloaded, CPU inference
**Fix**:
1. `ollama run <model>` (preload model)
2. `/device llm gpu` (if on CPU)
3. Use smaller model (e.g., `deepseek-r1:1.5b`)

### Symptom: Chat history not preserved after reload
**Cause**: By design (only last 4 messages preserved)
**Fix**: Expected behavior, full history in `sessions.json`

### Symptom: Index not appearing in multiselect
**Cause**: Directory missing, ChromaDB corruption
**Fix**:
1. Check `~/.tensortruth/indexes/` for subdirectory
2. Verify `chroma.sqlite3` exists
3. Rebuild: `tensor-truth-build --modules <name>`

## Future Considerations (Not Implemented)

- **Multi-Index Metadata Filtering**: Filter by library/category before retrieval
- **Hybrid Search**: BM25 + dense retrieval fusion
- **Re-Ranking Cascade**: Multiple reranker stages
- **Query Expansion**: Synonyms, acronyms
- **Citation Links**: Deep links to source docs
- **Auto-Update Indexes**: Periodic doc refresh
- **API Server Mode**: REST API for programmatic access
- **Multi-User Sessions**: Separate session storage per user

---

## Quick Start Checklist for New Contributors

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"  # Editable install with all extras
   ```

2. **Run Tests**:
   ```bash
   pytest -v --cov=src/tensortruth
   ```

3. **Launch App**:
   ```bash
   ollama serve  # In separate terminal
   tensor-truth
   ```

4. **Build Custom Index**:
   ```bash
   mkdir -p library_docs/mylib
   # Add .md or .html files
   tensor-truth-build --modules mylib
   ```

5. **Code Style**:
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   ```

---

**End of Context Document**

For latest updates, see [README.md](README.md) and [pyproject.toml](pyproject.toml).
