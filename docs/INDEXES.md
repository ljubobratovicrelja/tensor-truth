# Building Custom Indexes: A Guide to tensor-truth-docs and tensor-truth-build

This guide walks you through the process of building custom vector indexes for your own documentation, research papers, and textbooks using Tensor-Truth's CLI tools.

## Overview

Tensor-Truth uses a two-stage pipeline to create searchable indexes:

1. **`tensor-truth-docs`** (implemented in [fetch_sources.py](../src/tensortruth/fetch_sources.py))
   - Fetches and converts documentation sources to markdown
   - Supports three source types: **libraries** (Sphinx/Doxygen), **papers** (ArXiv), and **books** (PDF textbooks)
   - Features interactive `--add` mode for guided source addition
   - Outputs to `~/.tensortruth/library_docs/` with type-based prefixes (`library_*`, `papers_*`, `books_*`)

2. **`tensor-truth-build`** (implemented in [build_db.py](../src/tensortruth/build_db.py))
   - Builds vector indexes from fetched documentation using hierarchical chunking
   - Extracts metadata based on document type (library modules, paper citations, book chapters)
   - Outputs to `~/.tensortruth/indexes/` with matching directory names
   - Supports selective building with `--modules`, `--all`, `--books`, `--libraries`, or `--papers`

All data is stored in `~/.tensortruth/` by default (or `%USERPROFILE%\.tensortruth` on Windows).

### Configuration Paths (Environment Variables)

**All paths can be configured via environment variables** - this is especially useful for Docker deployments where you can prepare configurations upfront:

| Environment Variable | Default | Purpose |
|---------------------|---------|---------|
| `TENSOR_TRUTH_DOCS_DIR` | `~/.tensortruth/library_docs` | Source documentation directory |
| `TENSOR_TRUTH_SOURCES_CONFIG` | `~/.tensortruth/sources.json` | Sources configuration file |
| `TENSOR_TRUTH_INDEXES_DIR` | `~/.tensortruth/indexes` | Vector indexes output directory |

**Docker example:**
```bash
docker run -d \
  -e TENSOR_TRUTH_DOCS_DIR=/data/docs \
  -e TENSOR_TRUTH_SOURCES_CONFIG=/data/sources.json \
  -e TENSOR_TRUTH_INDEXES_DIR=/data/indexes \
  -v /host/data:/data \
  tensor-truth
```

This allows you to prepare your `sources.json` and pre-downloaded documentation before launching the container.

---

## The sources.json Configuration

The `sources.json` file is the heart of the system. It defines what documentation sources are available and how to fetch them.

### Default Configuration

A pre-configured `sources.json` is included with Tensor-Truth at [`config/sources.json`](../config/sources.json). It includes:

- **Libraries**: PyTorch, NumPy, SciPy, Matplotlib, Pandas, scikit-learn, and more
- **Research Papers**: Organized into categories (DL architectures, computer vision, NLP, etc.)
- **Textbooks**: Linear algebra, calculus, optimization, machine learning, etc.

### Configuration Structure

```json
{
  "libraries": {
    "pytorch_2.9": {
      "type": "sphinx",
      "version": "2.9",
      "doc_root": "https://pytorch.org/docs/stable/",
      "inventory_url": "https://pytorch.org/docs/stable/objects.inv",
      "selector": "div[role='main']"
    }
  },
  "papers": {
    "dl_architectures_optimization": {
      "display_name": "DL Foundations & Architectures",
      "description": "Core CNN/Transformer architectures and optimization methods",
      "type": "arxiv",
      "items": {
        "1512.03385": {
          "title": "Deep Residual Learning for Image Recognition",
          "arxiv_id": "1512.03385",
          "url": "https://arxiv.org/abs/1512.03385",
          "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
          "year": "2015"
        }
      }
    }
  },
  "books": {
    "linear_algebra_cherney": {
      "type": "pdf_book",
      "category": "linear_algebra",
      "title": "Linear Algebra",
      "authors": ["David Cherney", "Tom Denton", "Rohit Thomas", "Andrew Waldron"],
      "source": "https://www.math.ucdavis.edu/~linear/linear-guest.pdf",
      "split_method": "toc",
      "description": "Comprehensive free textbook from UC Davis",
      "items": {}
    }
  }
}
```

---

## Interactive Source Addition (Recommended)

The `--add` flag launches an interactive wizard that guides you through adding new sources:

```bash
tensor-truth-docs --add
```

### Features

The wizard will:
1. Ask what type of source you want to add (library, book, or paper)
2. Guide you through the configuration with smart prompts
3. Auto-fetch metadata where possible (ArXiv papers)
4. Validate your inputs
5. Save to sources.json
6. Optionally fetch the source immediately

### Adding Papers Interactively

**Fully Interactive:**
```bash
$ tensor-truth-docs --add

=== Interactive Source Addition ===
What would you like to add? (1/2/3 or library/book/paper): 3

=== Adding ArXiv Papers ===
Enter category name: computer_vision

✓ Using existing category: Computer Vision
  Description: Object detection, segmentation, and visual recognition
  Current papers: 5

Enter ArXiv IDs to add (space or comma separated):
Example: 1706.03762 2010.11929 1512.03385
ArXiv IDs: 1506.02640

Fetching metadata for 1 papers...
✓ 1506.02640: You Only Look Once: Unified, Real-Time Object Detection (2015)

=== Adding 1 papers to 'computer_vision' ===
  • You Only Look Once: Unified, Real-Time Object Detection (2015)

Add these papers? (y/n): y
✓ Added 1 papers to category 'computer_vision'

Fetch papers now? (y/n): y
```

**Skip Prompts with CLI Arguments:**
```bash
# Skip type selection
tensor-truth-docs --add --type paper

# Skip type and provide category
tensor-truth-docs --add --type paper --category computer_vision

# Fully non-interactive (provide all required info)
tensor-truth-docs --add --type paper --category computer_vision --arxiv-ids 1506.02640
```

## Case Study: Adding a Custom Paper Category

Let's walk through adding a new category of papers on **Gaussian Splatting** for 3D reconstruction.

### Step 1: List Available Sources

First, see what's already configured:

```bash
tensor-truth-docs --list
```

This shows all available libraries, paper categories, and books.

### Step 2: Create the Category in sources.json

The user configuration lives at `~/.tensortruth/sources.json`. On first run, it copies the default config. Edit this file to add your custom category:

```json
{
  "papers": {
    "3d_reconstruction": {
      "display_name": "3D Reconstruction & Rendering",
      "description": "Neural radiance fields, Gaussian splatting, and novel view synthesis",
      "type": "arxiv",
      "items": {
        "2308.04079": {
          "title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
          "arxiv_id": "2308.04079",
          "url": "https://arxiv.org/abs/2308.04079",
          "authors": "Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis",
          "year": "2023"
        },
        "2003.08934": {
          "title": "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
          "arxiv_id": "2003.08934",
          "url": "https://arxiv.org/abs/2003.08934",
          "authors": "Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng",
          "year": "2020"
        }
      }
    }
  }
}
```

### Step 3: Fetch the Papers

Download PDFs and convert to markdown:

```bash
# Fetch papers already defined in sources.json
tensor-truth-docs --type papers --category 3d_reconstruction

# Or add NEW papers by ArXiv ID (automatically updates sources.json)
tensor-truth-docs --type papers --category 3d_reconstruction --arxiv-ids 2308.04079 2003.08934

# Or use the new interactive mode (recommended)
tensor-truth-docs --add
```

**Important distinction:**
- **Without `--arxiv-ids`**: Fetches papers already listed in the category's `items` in sources.json
- **With `--arxiv-ids`**: Fetches ArXiv metadata, adds papers to sources.json automatically, then downloads them
- **With `--add`**: Interactive wizard guides you through adding sources (libraries, books, or papers)

**Options:**
- `--converter marker`: Use marker-pdf for better math equation support (slower)
- `--converter pymupdf`: Default, faster but less accurate for complex math
- `--format pdf`: Keep original PDFs without conversion

Papers are stored in `~/.tensortruth/library_docs/3d_reconstruction/`

### Step 4: Build the Index

Create vector embeddings from the markdown files:

```bash
tensor-truth-build --modules 3d_reconstruction
```

The index is built at `~/.tensortruth/indexes/papers_3d_reconstruction/`

**Build Options:**
- `--chunk-sizes 2048 512 256`: Hierarchical chunk sizes in tokens (default)
- `--chunk-overlap 64`: Overlap tokens between chunks (default: 64). Prevents information loss at boundaries.
- `--chunking-strategy hierarchical`: Chunking strategy (see [Chunking Strategies](#chunking-strategies) below)
- `--extensions .md .html .pdf`: File types to include
- `--embedding-model BAAI/bge-m3`: HuggingFace embedding model (default: BAAI/bge-m3)
- `--no-validate`: Skip HuggingFace Hub validation (for offline/private models)

**Chunk Size Strategy:**

Research shows that optimal chunk sizes vary by content type and use case. Based on empirical studies ([LlamaIndex](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5), [Pinecone](https://www.pinecone.io/learn/chunking-strategies/)) and community consensus:

**General findings:**
- **512-1024 tokens** balances faithfulness and relevancy for most content
- Smaller chunks (128-256) provide granular retrieval but risk missing context
- Larger chunks (2048+) preserve context but increase latency and "lost in the middle" problems

**Recommended strategies by content type:**

- **Books/Textbooks**: Preserve narrative flow with larger chunks
  - `--chunk-sizes 4096 1024 512` - Maintains chapter-level context
  - Books benefit from hierarchical retrieval (summary → sections → paragraphs)

- **API Documentation**: Default works well for precise reference lookup
  - `--chunk-sizes 2048 512 256` - Balances function-level detail with context (default)
  - Smaller leaf chunks help isolate individual API functions/classes

- **Research Papers**: Medium chunks capture complete ideas
  - `--chunk-sizes 3072 768 256` - Preserves mathematical proofs and arguments
  - Papers with heavy math benefit from larger chunks to avoid splitting equations

**Chunk Overlap:**

The `--chunk-overlap` parameter (default: 64 tokens) controls how much text overlaps between adjacent chunks. This prevents information loss at chunk boundaries where important context might be split.

- **Default (64)**: Good balance for most content
- **Higher (128-256)**: Better for dense technical content where sentences reference adjacent content
- **Lower (32)**: Faster indexing, acceptable for well-structured docs with clear section breaks

```bash
# Higher overlap for dense mathematical content
tensor-truth-build --modules linear_algebra_cherney --chunk-overlap 128

# Lower overlap for API docs with clear function boundaries
tensor-truth-build --modules pytorch --chunk-overlap 32
```

**Key principle:** The leaf size (last number) determines what gets embedded. Larger = more context but less precision. Start with defaults and iterate based on query patterns.

### Chunking Strategies

Tensor-Truth supports three chunking strategies via `--chunking-strategy`:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `hierarchical` | Fixed-size hierarchical chunks (default) | Fast indexing, uniform technical docs |
| `semantic` | Embedding-aware splits at natural boundaries | Narrative content, better context preservation |
| `semantic_hierarchical` | Two-pass: semantic split → hierarchical | Mixed content (narrative + code), highest quality |

**Examples:**

```bash
# Default: fast hierarchical chunking
tensor-truth-build --modules pytorch

# Semantic chunking for better context boundaries
tensor-truth-build --modules deep_learning_goodfellow --chunking-strategy semantic

# Two-pass for mixed content (papers with code examples)
tensor-truth-build --modules 3d_reconstruction --chunking-strategy semantic_hierarchical
```

#### Understanding Semantic Splitter Parameters

The semantic splitter uses embedding similarity to find natural content boundaries. Two parameters control its behavior:

**`--semantic-buffer-size`** (default: 1)

Controls how many sentences are grouped together before comparing embeddings. The splitter computes embeddings for sentence groups and splits where similarity drops significantly.

| Value | Effect | Use Case |
|-------|--------|----------|
| 1 | Fine-grained splits, sentence-level boundaries | API docs, reference material |
| 2-3 | Balanced splits, paragraph-level boundaries | General technical content |
| 4-5 | Coarse splits, section-level boundaries | Narrative content, textbooks |

**`--semantic-breakpoint-threshold`** (default: 95)

Percentile threshold for determining where to split. The splitter computes similarity scores between adjacent sentence groups, then splits at points where the similarity falls below this percentile threshold.

| Value | Effect | Use Case |
|-------|--------|----------|
| 80-85 | Aggressive splitting, many small chunks | Dense technical docs with distinct topics |
| 90-95 | Balanced splitting (default) | Most content types |
| 96-99 | Conservative splitting, fewer large chunks | Narrative content, proofs, arguments |

#### Recommendations by Document Type

**Books & Textbooks**

Books have narrative flow where ideas build across paragraphs. Use conservative settings to preserve context:

```bash
# Recommended for textbooks
tensor-truth-build --modules deep_learning_goodfellow \
  --chunking-strategy semantic \
  --semantic-buffer-size 3 \
  --semantic-breakpoint-threshold 95 \
  --chunk-sizes 4096 1024 512

# For math-heavy books (preserve proofs and derivations)
tensor-truth-build --modules linear_algebra_cherney \
  --chunking-strategy semantic \
  --semantic-buffer-size 4 \
  --semantic-breakpoint-threshold 97 \
  --chunk-overlap 128
```

**Academic Papers**

Papers have distinct sections (abstract, methods, results) but dense content within sections. Use `semantic_hierarchical` for best results:

```bash
# Recommended for research papers
tensor-truth-build --modules dl_architectures \
  --chunking-strategy semantic_hierarchical \
  --semantic-buffer-size 2 \
  --semantic-breakpoint-threshold 92 \
  --chunk-sizes 3072 768 256

# For papers with heavy math/equations
tensor-truth-build --modules optimization_papers \
  --chunking-strategy semantic_hierarchical \
  --semantic-buffer-size 3 \
  --semantic-breakpoint-threshold 95 \
  --chunk-overlap 96
```

**Programming Libraries & API Docs**

API documentation has clear boundaries between functions/classes. Use `hierarchical` (fast) or fine-grained semantic:

```bash
# Fast option: hierarchical is usually sufficient for API docs
tensor-truth-build --modules pytorch \
  --chunking-strategy hierarchical \
  --chunk-sizes 2048 512 256

# If API docs have narrative tutorials mixed in
tensor-truth-build --modules fastapi \
  --chunking-strategy semantic \
  --semantic-buffer-size 1 \
  --semantic-breakpoint-threshold 88 \
  --chunk-sizes 2048 512 256
```

#### Quick Reference Table

| Document Type | Strategy | Buffer | Threshold | Chunk Sizes |
|---------------|----------|--------|-----------|-------------|
| API Documentation | `hierarchical` | - | - | 2048 512 256 |
| API + Tutorials | `semantic` | 1 | 88 | 2048 512 256 |
| Research Papers | `semantic_hierarchical` | 2 | 92 | 3072 768 256 |
| Math-heavy Papers | `semantic_hierarchical` | 3 | 95 | 3072 768 256 |
| Textbooks | `semantic` | 3 | 95 | 4096 1024 512 |
| Math Textbooks | `semantic` | 4 | 97 | 4096 1024 512 |

**Performance note:** Semantic strategies require computing embeddings during parsing, making them 3-5x slower than hierarchical. For large documentation sets (1000+ files), consider using `hierarchical` for speed, or process in batches.

### Step 5: Use in the App

Launch the web interface:

```bash
tensor-truth
```

Your new paper category will appear in the "Active Indexes" sidebar. Select it to enable retrieval from those papers.

---

## Adding Library Documentation

### Example: Adding FastAPI Documentation

FastAPI uses Sphinx documentation, so the process is straightforward.

#### 1. Add to sources.json

```json
{
  "libraries": {
    "fastapi_0.115": {
      "type": "sphinx",
      "version": "0.115",
      "doc_root": "https://fastapi.tiangolo.com/",
      "inventory_url": "https://fastapi.tiangolo.com/objects.inv",
      "selector": "article.md-content"
    }
  }
}
```

**Key fields:**
- `type`: Either `"sphinx"` or `"doxygen"`
- `doc_root`: Base URL of the documentation
- `inventory_url`: Location of `objects.inv` file (Sphinx inventory)
- `selector`: CSS selector for the main content area

#### 2. Fetch Documentation

```bash
tensor-truth-docs fastapi_0.115
```

**Options:**
- `--workers 10`: Number of parallel downloads (default: 20)
- `--format markdown`: Output format (markdown, html, or pdf)
- `--cleanup`: Aggressive HTML cleanup (useful for Doxygen)
- `--min-size 100`: Skip files smaller than 100 characters

#### 3. Build the Index

```bash
tensor-truth-build --modules fastapi_0.115
```

---

## Adding Textbooks

Textbooks require special handling because they need to be split into chapters or chunks.

### Example: Adding a Custom Textbook

Let's add a deep learning textbook.

#### 1. Add to sources.json

```json
{
  "books": {
    "deep_learning_goodfellow": {
      "type": "pdf_book",
      "category": "machine_learning",
      "title": "Deep Learning",
      "authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"],
      "source": "https://github.com/janishar/mit-deep-learning-book-pdf/raw/master/complete-book-pdf/deeplearningbook.pdf",
      "split_method": "toc",
      "description": "Comprehensive textbook on deep learning theory",
      "items": {}
    }
  }
}
```

**Split methods:**
- `"toc"`: Automatically detect chapters from table of contents (recommended)
- `"none"`: Manual chunking by page count
- `"manual"`: Pre-define chapters with page ranges in `items`

#### 2. Fetch and Convert

```bash
# Fetch with automatic chapter detection
tensor-truth-docs --type books deep_learning_goodfellow --converter marker

# Or fetch all books in a category
tensor-truth-docs --type books --category machine_learning

# Or fetch all books
tensor-truth-docs --type books --all
```

**Options:**
- `--pages-per-chunk 15`: Pages per chunk when `split_method="none"` (default: 15)
- `--max-pages-per-chapter 0`: Split large chapters (0 = no limit)

#### 3. Build the Index

```bash
tensor-truth-build --modules deep_learning_goodfellow
```

---

## Advanced Workflows

### Batch Operations

Fetch and build everything at once:

```bash
# Fetch all sources
tensor-truth-docs --type papers --all
tensor-truth-docs --type books --all

# Build all indexes
tensor-truth-build --all

# Or by category
tensor-truth-build --papers    # All paper categories
tensor-truth-build --books     # All books
tensor-truth-build --libraries # All libraries
```

### Custom Paths

Override default paths with environment variables or CLI arguments (see [Configuration Paths](#-configuration-paths-environment-variables) for details):

```bash
# Using environment variables (recommended for Docker)
export TENSOR_TRUTH_DOCS_DIR=/data/docs
export TENSOR_TRUTH_SOURCES_CONFIG=/data/sources.json
export TENSOR_TRUTH_INDEXES_DIR=/data/indexes

# Or using CLI arguments
tensor-truth-docs pytorch \
  --library-docs-dir /data/docs \
  --sources-config /data/sources.json

tensor-truth-build --modules pytorch \
  --library-docs-dir /data/docs \
  --indexes-dir /data/indexes \
  --sources-config /data/sources.json
```

### Validation

Verify your configuration:

```bash
# Check sources.json structure and filesystem
tensor-truth-docs --validate
```

This validates:
- JSON syntax
- Required fields
- File existence
- Index availability

---

## Tips and Best Practices

### 1. Use Marker for Math-Heavy Content
For papers with complex equations, use `--converter marker`:
```bash
tensor-truth-docs --type papers --category 3d_reconstruction --converter marker
```

### 2. Filter Small Files
Library docs often include tiny navigation files. Skip them:
```bash
tensor-truth-docs pytorch --min-size 200
```

### 3. Monitor GPU Memory
Building indexes uses GPU for embeddings. If you run out of memory, close Ollama temporarily:
```bash
# Stop Ollama
pkill ollama  # Linux/Mac
taskkill /F /IM ollama.exe  # Windows

# Build indexes
tensor-truth-build --modules my_module

# Restart Ollama
ollama serve
```

### 4. Incremental Updates
Only rebuild what changed:
```bash
# Fetch new version
tensor-truth-docs pytorch_2.10

# Build only the new version
tensor-truth-build --modules pytorch_2.10
```

### 5. Note on Retrieval Evaluation

Tensor-Truth does not currently implement automated retrieval metrics (e.g., faithfulness, relevancy, hit rate, MRR). The chunk size recommendations are based on published research and community best practices rather than automated evaluation on your specific dataset.

**Planned features:** We intend to integrate the **LlamaIndex Response Evaluator** for measuring faithfulness and relevancy, and experiment with **RAGAS** for comprehensive retrieval quality metrics. These will help users optimize chunk sizes and retrieval strategies for their specific use cases.

For now, if you need to evaluate retrieval performance, consider manually testing query accuracy with different chunk configurations.

---

## Troubleshooting

### PDFs Won't Download
- Check the source URL is accessible
- Some sites block automated downloads (use manual download + file:// URL)

### Build Fails with "No documents found"
- Ensure you ran `tensor-truth-docs` first
- Check the module name matches between fetch and build
- Verify files exist in `~/.tensortruth/library_docs/<module_name>/` (or wherever `TENSOR_TRUTH_DOCS_DIR` points)

### Book Chapters Not Detected
- Try `--converter marker` for better TOC extraction
- Fall back to manual splitting with `split_method: "none"`
- Pre-define chapters manually in sources.json with page ranges
- **Large chapters detected?** Use `--max-pages-per-chapter 20` to split them into smaller files. Smaller chunks (15-25 pages) work better for RAG retrieval as they allow more focused context without overwhelming the LLM's context window.

---

## Example: Complete Workflow

Here's a complete example adding a new computer vision paper category:

```bash
# 1. Edit ~/.tensortruth/sources.json
cat >> ~/.tensortruth/sources.json << 'EOF'
{
  "papers": {
    "object_detection": {
      "display_name": "Object Detection",
      "description": "Modern object detection architectures",
      "type": "arxiv",
      "items": {
        "1506.01497": {
          "title": "Faster R-CNN: Towards Real-Time Object Detection",
          "arxiv_id": "1506.01497",
          "url": "https://arxiv.org/abs/1506.01497",
          "authors": "Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun",
          "year": "2015"
        }
      }
    }
  }
}
EOF

# 2. Fetch papers
tensor-truth-docs --type papers --category object_detection --converter marker

# 3. Build index
tensor-truth-build --modules object_detection

# 4. Launch app
tensor-truth
```

Your new papers are now searchable in the UI!

---

## Summary

The workflow is simple:

1. **Configure**: Edit `~/.tensortruth/sources.json` with your sources
2. **Fetch**: Run `tensor-truth-docs` to download and convert
3. **Build**: Run `tensor-truth-build` to create vector indexes
4. **Use**: Launch `tensor-truth` and select your indexes

For the full default configuration, see [`config/sources.json`](../config/sources.json).

For Docker deployment and advanced configuration, see [DOCKER.md](DOCKER.md).
