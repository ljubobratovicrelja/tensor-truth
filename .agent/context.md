# Tensor-Truth: AI Agent Context Document

**Last Updated**: 2025-12-19 (Context refresh for v0.1.10)
**Version**: 0.1.10 (Code Cleanup & Documentation)
**Python**: 3.11+

---

## CRITICAL INSTRUCTIONS FOR AI AGENTS

### Document Reading Protocol
**WHEN INSTRUCTED TO "READ .agent/context.md" WITH NO FURTHER CONTEXT:**
- Assume you've been initialized to perform a specific task within this project
- Parse the document silently to understand the architecture and context
- **Do NOT provide a comprehensive response** explaining your understanding
- **Simply respond in one sentence** that you are ready to begin work

### Document Maintenance Protocol
**WHENEVER YOU COMPLETE A CODING TASK IN THIS PROJECT:**
1. **Before ending the conversation**, review this context document
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
- Any other .md files besides this context document

**ONLY update .agent/context.md** as part of your regular workflow to keep it accurate.

### Emoji Usage Policy
**DO NOT use emojis in code, logs, or comments unless explicitly instructed or functionally required.**

Acceptable emoji usage (functional only):
- UI elements where emojis serve a functional purpose (e.g., star icon for favorite toggle in `app.py`)
- User-facing status indicators with semantic meaning (e.g., status icons in session PDF upload UI: uploading, processing, indexed, error)

Prohibited emoji usage:
- Log messages (use plain text: "INFO", "ERROR", "WARNING")
- Code comments (describe functionality in words)
- Docstrings
- Console output from CLI tools
- Error messages
- Debug output
- Aesthetic decoration in any code context

**Rationale**: Emojis add visual noise, can cause encoding issues, and don't improve code clarity or functionality. They should only appear when they serve a specific user-facing feature requirement.

### Code Quality Workflow (MANDATORY END-OF-RESPONSE CHECKLIST)
**CRITICAL: AT THE END OF EVERY RESPONSE WHERE YOU EDIT CODE**

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
│   ├── fetch_sources.py       # Unified source fetching (libraries + papers)
│   ├── pdf_handler.py         # Session PDF upload/conversion (v0.1.9)
│   ├── session_index.py       # Session vector index builder (v0.1.9)
│   ├── utils/                 # Utilities (v0.1.10: chat, pdf, metadata)
│   ├── core/                  # System utils (VRAM, Ollama, device detection)
│   └── app_utils/             # Streamlit helpers (commands, presets, sessions, config)
├── config/sources.json        # Unified config: libraries + papers (v0.1.10)
├── tests/                     # Pytest suite (unit, integration)
└── ~/.tensortruth/            # User data (sessions, presets, indexes, config)
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
- **PDF dependencies are in core** - `pymupdf4llm` and `marker-pdf` included (session PDF upload feature)
- `tensor-truth-build` works without extras (uses core dependencies only)
- Unified `tensor-truth-docs` requires `[docs]` extra for both library and paper fetching

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

### 7. `utils/` Module - Utility Functions (NEW in v0.1.10)
**Purpose**: Organized utility functions for PDF processing, chat utilities, and metadata extraction

**Structure**:
- `utils/__init__.py` - Re-exports for backward compatibility
- `utils/chat.py` - Chat/thinking/LaTeX utilities
- `utils/pdf.py` - PDF processing (conversion, TOC extraction, splitting)
- `utils/metadata.py` - Document metadata extraction (LLM + explicit)

**Key Functions**:

**PDF Processing** (`utils/pdf.py`):
- `convert_pdf_to_markdown()` - PDF → Markdown with pymupdf4llm or marker
- `convert_with_marker()` - GPU-accelerated PDF conversion (better math)
- `post_process_math()` - Unicode → LaTeX symbol conversion
- `clean_filename()` - Sanitize filenames for filesystem
- `get_pdf_page_count()` - Extract page count from PDF

**Chat Utilities** (`utils/chat.py`):
- `parse_thinking_response()` - Extract <thought> tags from LLM output
- `convert_latex_delimiters()` - \[...\] → $$...$$ for Streamlit
- `convert_chat_to_markdown()` - Export chat history to markdown file

**Metadata Extraction** (`utils/metadata.py`):
- `extract_document_metadata()` - Main orchestrator for metadata extraction
- `extract_yaml_header_metadata()` - Parse YAML headers from markdown (# Title:, # Authors:)
- `extract_pdf_metadata()` - Extract from PDF metadata dict (PyMuPDF)
- `extract_metadata_with_llm()` - LLM-based fallback (uses qwen2.5:0.5b)
- `classify_document_type()` - Classify as paper, book, library_doc, uploaded_pdf
- `format_authors()` - Smart author formatting (et al. for >3 authors)
- `create_display_name()` - Generate "Title - Authors" citation string

**Metadata Extraction Strategy**:
1. **Explicit First**: YAML headers (markdown) or PDF metadata dict
2. **LLM Fallback**: If no explicit metadata and `use_llm_fallback=True`
3. **Graceful Degradation**: Filename as display_name if all else fails

**Use Cases**:
- Session PDF indexing: Extract title/authors from uploaded papers
- Build-time indexing: Enrich book chapters with shared metadata
- Source citations: Display pretty names in RAG responses

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

### 9. User Data Directory (`~/.tensortruth/`)
**Structure**:
```
~/.tensortruth/
├── chat_sessions.json      # Chat history (messages, params, modules, PDFs)
├── presets.json            # Saved configurations
├── config.yaml             # Global settings (Ollama URL)
├── indexes/                # Permanent vector databases
│   └── pytorch/            # ChromaDB instance (chroma.sqlite3, docstore.json)
└── sessions/               # Per-session data
    └── sess_abc123/
        ├── pdfs/           # Uploaded PDFs
        ├── markdown/       # Converted markdown
        └── index/          # Session-specific ChromaDB
```

**Sessions Structure** (key fields):
```json
{
  "sessions": {
    "sess_123": {
      "title": "PyTorch Autograd",
      "modules": ["pytorch"],
      "params": { /* model, temperature, etc. */ },
      "pdf_documents": [
        {
          "id": "pdf_xyz",
          "filename": "attention.pdf",
          "status": "indexed",
          "display_name": "Attention Is All You Need - Vaswani et al.",
          /* page_count, file_size, uploaded_at, etc. */
        }
      ],
      "pdf_metadata_cache": { /* extracted metadata by PDF ID */ },
      "has_temp_index": true,
      "messages": [ /* chat history */ ]
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
- `__init__(session_id, metadata_cache)` → initializes with optional metadata cache
- `build_index(markdown_files)` → creates ChromaDB index with metadata extraction
- `rebuild_index()` → reindexes after PDF deletion
- `delete_index()` → cleanup on session delete or last PDF removal
- `index_exists()` → checks for valid ChromaDB
- `get_metadata_cache()` → retrieves extracted metadata for persistence
- `close()` → explicitly releases ChromaDB connections (important on Windows)

**Metadata Extraction in Session Indexing**:
- Extracts title/authors from uploaded PDFs using LLM (qwen2.5:0.5b)
- Caches metadata per PDF ID to avoid re-extraction on rebuild
- Injects essential fields (display_name, authors, source_url, doc_type) into chunks
- Forces CPU embedding to avoid VRAM overflow (LLM already loaded)

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
- Metadata (title, authors) displayed in citations if extracted successfully

### Limitations & Design Decisions

**Session-Scoped Only**: PDFs cannot be shared across sessions (intentional isolation).

**Marker-PDF First**: Better for academic papers with math notation, but slower (~10-30s per paper).

**No Inline PDF Display**: Only filename + status shown, no preview/thumbnails.

**Rebuild on Delete**: Removing a PDF triggers full index rebuild (acceptable for 3-5 papers).

**No Page-Level Citations**: Sources show filename, not specific page numbers.

## Document Metadata Extraction System (v0.1.9+)

**Purpose**: Extract rich metadata (title, authors, URL) from documents for enhanced citation display in RAG responses.

**Architecture** ([utils/metadata.py](src/tensortruth/utils/metadata.py)):

**Extraction Strategy** (in order of priority):
1. **Explicit Metadata** (fastest, most reliable):
   - YAML headers in markdown: `# Title:`, `# Authors:`, `# Year:`, `# ArXiv ID:`
   - PDF metadata dict: Title, Author, CreationDate (PyMuPDF)
2. **LLM Extraction** (fallback, requires Ollama):
   - Uses `qwen2.5:0.5b` to extract from first 2000 chars
   - JSON prompt with strict format validation
3. **Filename Fallback** (last resort):
   - Sanitized filename as display_name

**Key Functions**:
- `extract_document_metadata()` - Main orchestrator, returns full metadata dict
- `extract_yaml_header_metadata()` - Parse markdown headers
- `extract_pdf_metadata()` - Extract from PDF info dict
- `extract_metadata_with_llm()` - LLM-based extraction with retry logic
- `classify_document_type()` - Returns: `paper`, `book`, `library_doc`, `uploaded_pdf`
- `format_authors()` - Smart formatting: "Doe et al." for >3 authors
- `create_display_name()` - Generate citation string: "Title - Authors"

**Usage Contexts**:
1. **Session PDF Indexing** (session_index.py:138-172):
   - Extract metadata from uploaded PDFs
   - Cache per PDF ID to avoid re-extraction on rebuild
   - Inject into chunk metadata for RAG retrieval
2. **Build-Time Indexing** (build_db.py:98-181):
   - Extract book metadata from PDF or first chapter
   - Share metadata across all chapters of same book
   - Override with config for library docs and paper collections
3. **Source Citations** (app.py:1000+):
   - Display pretty names in RAG response sources
   - Show author information when available

**Metadata Fields**:
```python
{
    "title": str,           # Document title
    "authors": str,         # Formatted author string
    "display_name": str,    # "Title - Authors" for UI
    "source_url": str,      # URL to original source
    "doc_type": str,        # paper|book|library_doc|uploaded_pdf
    "arxiv_id": str,        # ArXiv ID if applicable (optional)
    "year": str,            # Publication year (optional)
}
```

**Performance Considerations**:
- YAML/PDF extraction: <1ms per document
- LLM extraction: ~500-1000ms per document (depends on Ollama load)
- Caching prevents duplicate extractions on rebuild
- Metadata stored in session JSON for session PDFs

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
| `context_window` | `16384` | 2048-32768 | Must match model's context limit |
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

### Adding New Library/Paper Category
1. Add entry to `config/sources.json` (under `"libraries"` or `"papers"`)
2. Run `tensor-truth-docs --type library <name>` or `tensor-truth-docs --type papers --category <cat>`
3. Run `tensor-truth-build --modules <name>`
4. Restart app, module appears in multiselect

### Implementing New Command
1. Create `Command` subclass in `app_utils/commands.py`
2. Register in `_registry`, add to `CommandRegistry.get_help_text()`

### Changing Chunking Strategy
1. Edit `build_db.py` default `chunk_sizes=[2048, 512, 128]`
2. Rebuild with `tensor-truth-build --chunk-sizes <sizes>`

### Debugging RAG Retrieval
1. Enable `verbose=True` in `rag_engine.py:308`
2. Check node metadata: `node.metadata['file_name']`, `node.score`
3. Inspect `context_nodes` in app.py (pre/post-rerank)

### Custom Prompts
Edit templates in `rag_engine.py:51-144`:
- `CUSTOM_CONTEXT_PROMPT_TEMPLATE` (normal RAG)
- `CUSTOM_CONTEXT_PROMPT_LOW_CONFIDENCE` (soft warning)
- `CUSTOM_CONTEXT_PROMPT_NO_SOURCES` (fallback)

## Performance & Scaling

**Typical Query Latency** (M1 Max, DeepSeek-R1:8b):
- Retrieval: 0.5-1.5s | Reranking: 0.2-0.5s | LLM First Token: 1-2s
- **Total to First Token**: ~2-4s | Streaming: 20-40 tokens/sec

**Memory Usage**:
- Embedder: ~1.2GB | Reranker: ~800MB | LLM: ~5-6GB | ChromaDB: ~100-500MB per index

**Scaling Limits**:
- Max indexes: No hard limit (retrieval latency scales linearly)
- Max messages: Truncates to 3000 tokens (~2-4 turns)
- Max nodes: `similarity_top_k` * num_indexes (e.g., 6 * 5 = 30 pre-rerank)

## Edge Cases & Troubleshooting

**Engine Loading Race**: User queries before engine finishes → 60s timeout wait (app.py:807-827)

**Empty Retrieval**: Query matches nothing → synthetic warning node injected (app.py:955-981)

**Ollama Connection Failure**: Check `get_ollama_url()` via `/list`, update in Connection Settings

**VRAM OOM**: `/device rag cpu` to offload pipeline, or `/device llm cpu` for LLM

**Index Corruption**: Delete `~/.tensortruth/indexes/{module}/`, rebuild with `tensor-truth-build`

**Streamlit State Desync**: Use `skip_last_message_render` flag to prevent double-render (app.py:743-746)

**Common Symptoms**:
- "No models found" → Ollama not running (`ollama serve`)
- Engine timeout (60s) → Reduce active indexes or move to CPU
- Low-confidence warnings → Lower threshold (`/conf 0.1`) or verify index contents
- Slow first token (>10s) → Preload model (`ollama run <model>`) or use smaller model
- Index not in multiselect → Check directory exists, verify `chroma.sqlite3` present

## Version History (Key Changes)

### 0.1.10 (Current)
- **Code cleanup**: Formatting, linting, type hints
- **Documentation updates**: Enhanced inline docs, context.md refresh
- **No functional changes**: Pure maintenance release

### 0.1.9
- **Session-scoped PDF ingestion**: Upload PDFs in chat mode, converted with marker-pdf
- **Temporary vector indexes**: Per-session ChromaDB indexes for uploaded PDFs
- **Parallel retrieval**: Queries fetch from both permanent knowledge bases + session PDFs
- **Metadata extraction**: LLM-based title/author extraction for uploaded PDFs (new `utils/metadata.py` module)
- **Metadata caching**: Avoid re-extraction on index rebuild
- **CLI Unification**: Merged `tensor-truth-papers` into `tensor-truth-docs` command
- **Module Reorganization**: Created `utils/` package (chat.py, pdf.py, metadata.py modules)
- **Unified Config**: `sources.json` replaces `api.json` + `papers.json`
- **Breaking Changes**: Removed `tensor-truth-papers` CLI, `fetch_paper.py`, `scrape_docs.py`, `utils.py`
- New files: `pdf_handler.py`, `session_index.py`, `fetch_sources.py`, `utils/` module
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
| Chat Utilities | `utils/chat.py` | - |
| PDF Processing | `utils/pdf.py` | - |
| Metadata Extraction | `utils/metadata.py` | - |

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
GDRIVE_LINK = https://drive.google.com/file/d/12wZsBwrywl9nXOCLr50lpWB2SiFdu1XB/view?usp=sharing

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

**Single-Process ChromaDB**: PersistentClient (not server mode), no concurrent writes, safe for single-user local deployment

**No Authentication**: Streamlit without auth, assumes trusted local network, sessions in plaintext JSON

**Ollama as LLM Backend**: No OpenAI/Anthropic integration, assumes Ollama server running

**GPU Optional**: CPU fallback for all components (slower but functional, LLM on CPU not recommended for >3B models)

**Pre-Built Indexes**: First-run auto-download from GDrive (~500MB), manual build requires `[docs]` extra

**Streamlit UI**: Stateful via `st.session_state`, reruns on interaction, threading for background tasks (not asyncio)

**Markdown/HTML Sources**: Docs converted to Markdown before indexing, HTML preserved for Doxygen, PDFs via pymupdf4llm/marker

## Docker Deployment

**Base**: `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime` | **Port**: 8501 | **Volume**: `/root/.tensortruth`

**Run**:
```bash
docker run -d --name tensor-truth --gpus all -p 8501:8501 \
  -v ~/.tensortruth:/root/.tensortruth \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  ljubobratovicrelja/tensor-truth:latest
```

**Build locally**: `docker build -t tensor-truth .`

---

## Quick Start for Contributors

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Test & Run
pytest -v --cov=src/tensortruth
ollama serve  # separate terminal
tensor-truth

# Build custom index
mkdir -p library_docs/mylib  # Add .md/.html files
tensor-truth-build --modules mylib

# Code style
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/
```

---

**End of Context Document**

For latest updates, see [README.md](README.md) and [pyproject.toml](pyproject.toml).
