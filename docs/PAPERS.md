# Building Custom Indexes: A Guide to tensor-truth-docs and tensor-truth-build

This guide walks you through the process of building custom vector indexes for your own documentation, research papers, and textbooks using Tensor-Truth's CLI tools.

## Overview

Tensor-Truth uses a two-stage pipeline to create searchable indexes:

1. **`tensor-truth-docs`** - Fetch and convert documentation sources to markdown
2. **`tensor-truth-build`** - Build vector indexes from the markdown files

All data is stored in `~/.tensortruth/` by default (or `%USERPROFILE%\.tensortruth` on Windows).

### 🔧 Configuration Paths (Environment Variables)

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
tensor-truth-docs --type papers --category 3d_reconstruction --ids 2308.04079 2003.08934
```

**Important distinction:**
- **Without `--ids`**: Fetches papers already listed in the category's `items` in sources.json
- **With `--ids`**: Fetches ArXiv metadata, adds papers to sources.json automatically, then downloads them

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
- `--chunk-sizes 2048 512 128`: Control hierarchical chunk sizes (default)
- `--extensions .md .html .pdf`: File types to include

**💡 Chunk Size Strategy:**

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
  - `--chunk-sizes 2048 512 128` - Balances function-level detail with context
  - Smaller leaf chunks help isolate individual API functions/classes

- **Research Papers**: Medium chunks capture complete ideas
  - `--chunk-sizes 3072 768 256` - Preserves mathematical proofs and arguments
  - Papers with heavy math benefit from larger chunks to avoid splitting equations

**Key principle:** The leaf size (last number) determines what gets embedded. Larger = more context but less precision. Start with defaults and iterate based on query patterns.

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
