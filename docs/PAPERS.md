# arXiv Paper Management System

A unified script for managing, fetching, and organizing AI/ML/CV papers from arXiv for RAG pipelines.

## Structure

```
.
├── fetch_paper.py           # Main unified script (improved)
├── papers_config.json        # Paper configuration organized by category
└── library_docs/            # Output directory (created automatically)
    ├── dl_foundations/
    ├── vision_2d_generative/
    └── 3d_reconstruction_rendering/
```

## Configuration File

The `papers_config.json` organizes papers into categories with this structure:

```json
{
  "category_id": {
    "description": "Category description",
    "items": [
      {
        "title": "Paper Title",
        "arxiv_id": "YYMM.NNNNN",
        "url": "https://arxiv.org/abs/YYMM.NNNNN"
      }
    ]
  }
}
```

The provided config includes 40 influential papers in 3 categories:
- **dl_foundations** (13 papers) - Core DL architectures, Transformers, foundation models
- **vision_2d_generative** (14 papers) - Object detection, segmentation, GANs, diffusion
- **3d_reconstruction_rendering** (13 papers) - NeRF, Gaussian Splatting, relighting, 4D

## Usage Modes

### 1. Add New Papers (Write to Config + Download)

Add papers to a category. This will:
- Fetch and download the paper
- Convert to Markdown
- **Add the paper metadata to the JSON config**

```bash
# Add single paper
python fetch_paper.py --config papers_config.json --category dl_foundations --ids 2401.12345

# Add multiple papers
python fetch_paper.py --config papers_config.json --category 3d_reconstruction_rendering --ids 2401.12345 2402.67890 2403.13111
```

### 2. Rebuild from Config (Read from Config + Download Missing)

Rebuild/update papers from existing config. This will:
- Read all papers in the specified category from config
- Download and convert any missing papers
- Skip papers that already exist

```bash
# Rebuild single category
python fetch_paper.py --config papers_config.json --rebuild dl_foundations

# Rebuild multiple categories
python fetch_paper.py --config papers_config.json --rebuild dl_foundations vision_2d_generative

# Rebuild ALL categories
python fetch_paper.py --config papers_config.json --rebuild-all

# Rebuild with parallel workers (much faster!)
python fetch_paper.py --config papers_config.json --rebuild-all --workers 8

# Default is 4 workers, adjust based on your connection
python fetch_paper.py --config papers_config.json --rebuild dl_foundations --workers 12
```

### 3. List Categories

View all categories and paper counts:

```bash
python fetch_paper.py --config papers_config.json --list
```

Output:
```
Available categories:
============================================================

dl_foundations
  Description: Deep Learning Foundations & Architectures - Core architectures...
  Papers: 13

vision_2d_generative
  Description: 2D Vision Tasks & Generative Models - Object detection...
  Papers: 14

3d_reconstruction_rendering
  Description: 3D/4D Reconstruction, Rendering & Relighting - Neural scene...
  Papers: 13
============================================================
```

## Key Features

### Parallel Processing
- Download multiple papers simultaneously
- Configurable worker count (default: 4)
- Significantly faster for large collections
- Thread-safe logging and statistics

### Intelligent Skipping
- Checks for existing PDF + MD files
- Only processes missing papers
- Saves time on rebuilds

### Automatic Config Management
- Adding papers automatically updates the JSON config
- No manual JSON editing needed
- Config stays in sync with downloaded papers

### Category Organization
Papers are organized by category in `library_docs/`:
```
library_docs/
├── dl_foundations/
│   ├── 1512.03385.pdf
│   ├── Deep_Residual_Learning_for_Image_Recognition.md
│   └── ...
├── vision_2d_generative/
│   └── ...
└── 3d_reconstruction_rendering/
    └── ...
```

### Progress Tracking
Detailed logging shows:
- Paper being processed
- Status (processing, skipping, or error)
- Summary statistics after rebuild

## Workflow Examples

### Starting a New Category

```bash
# Add papers to a new category
python fetch_paper.py --config papers_config.json --category diffusion_models --ids 2006.11239 2112.10752 2204.06125

# The config will automatically create the new category
```

### Updating an Existing Category

```bash
# Add newly released papers
python fetch_paper.py --config papers_config.json --category 3d_reconstruction_rendering --ids 2412.12345

# Then rebuild to ensure everything is downloaded
python fetch_paper.py --config papers_config.json --rebuild 3d_reconstruction_rendering
```

### Fresh Setup on New Machine

```bash
# Clone your repo with papers_config.json
git clone your-repo

# Rebuild all categories to download papers (with 8 parallel workers)
python fetch_paper.py --config papers_config.json --rebuild-all --workers 8

# This will download all 40 papers much faster than sequential!
```

### Performance Tuning

```bash
# Conservative (good for unstable connections)
python fetch_paper.py --config papers_config.json --rebuild-all --workers 2

# Balanced (default)
python fetch_paper.py --config papers_config.json --rebuild-all --workers 4

# Aggressive (fast connection, many papers)
python fetch_paper.py --config papers_config.json --rebuild-all --workers 12
```

## Output Format

Each Markdown file includes metadata headers ideal for RAG:

```markdown
# Title: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
# Authors: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
# Year: 2023
# ArXiv ID: 2308.04079
# Abstract: 
Radiance Field methods have recently revolutionized novel-view synthesis...

---

[Converted paper content with preserved layout, tables, and figures...]
```

This structure enables your RAG system to:
- Index by category
- Search by metadata (author, year, arXiv ID)
- Retrieve relevant sections with full context
- Handle complex layouts (2-column papers, tables)

## Requirements

```bash
pip install arxiv pymupdf4llm
```

## Command Reference

```bash
# Add papers (writes to config)
python fetch_paper.py --config CONFIG --category CATEGORY --ids ID1 ID2 ...

# Rebuild specific categories (reads from config)
python fetch_paper.py --config CONFIG --rebuild CATEGORY1 CATEGORY2 ...

# Rebuild ALL categories
python fetch_paper.py --config CONFIG --rebuild-all

# Use parallel workers (faster downloads)
python fetch_paper.py --config CONFIG --rebuild-all --workers N

# List categories
python fetch_paper.py --config CONFIG --list

# Help
python fetch_paper.py --help
```

## Configuration Defaults

- Default config file: `papers_config.json`
- Default output directory: `./library_docs`
- Default parallel workers: `4`

You can specify a different config with `--config path/to/config.json`

## Performance Notes

**Worker Count Recommendations:**
- **2-4 workers**: Stable/slow connections, prevents timeouts
- **4-8 workers**: Normal connections (recommended)
- **8-16 workers**: Fast connections, large batches

**Expected Speed:**
- Sequential (1 worker): ~40 papers in 30-45 minutes
- Parallel (4 workers): ~40 papers in 10-15 minutes
- Parallel (8 workers): ~40 papers in 8-12 minutes

Times vary based on paper sizes and connection speed.