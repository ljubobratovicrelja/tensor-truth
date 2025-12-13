# Testing Framework for Tensor-Truth

## Overview

This document outlines the testing strategy for the tensor-truth RAG pipeline project.

## Testing Stack

- **pytest**: Primary testing framework
- **pytest-cov**: Code coverage reporting
- **pytest-mock**: Mocking support
- **pytest-asyncio**: Async test support (for future async features)
- **hypothesis**: Property-based testing for edge cases
- **responses**: HTTP request mocking
- **faker**: Test data generation

## Installation

```bash
pip install pytest pytest-cov pytest-mock pytest-asyncio hypothesis responses faker
```

## Project Structure

```
tensor-truth/
├── src/tensortruth/
│   ├── __init__.py
│   ├── build_db.py
│   ├── fetch_paper.py
│   ├── rag_engine.py
│   ├── scrape_docs.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures and configuration
│   ├── unit/                     # Unit tests (isolated functions)
│   │   ├── __init__.py
│   │   ├── test_utils.py
│   │   ├── test_fetch_paper.py
│   │   ├── test_scrape_docs.py
│   │   └── test_rag_engine.py
│   ├── integration/              # Integration tests (components together)
│   │   ├── __init__.py
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_rag_pipeline.py
│   │   └── test_db_building.py
│   ├── e2e/                      # End-to-end tests
│   │   ├── __init__.py
│   │   └── test_app.py
│   └── fixtures/                 # Test data and fixtures
│       ├── sample_papers/
│       ├── sample_configs/
│       └── mock_responses/
├── pytest.ini                    # Pytest configuration
└── .coveragerc                   # Coverage configuration
```

## Test Categories

### 1. Unit Tests (Fast, No External Dependencies)

#### `test_utils.py`
- ✅ `parse_thinking_response()` - Various input formats
- ✅ `get_max_memory_gb()` - CUDA/MPS/CPU detection
- ✅ `convert_chat_to_markdown()` - Session formatting
- ✅ `stop_model()` - API interaction (mocked)
- ✅ `get_running_models()` - Ollama API parsing (mocked)

#### `test_fetch_paper.py`
- ✅ `clean_filename()` - Sanitization edge cases
- ✅ `paper_already_processed()` - File detection
- ✅ `book_already_processed()` - Book detection
- ✅ `extract_toc()` - PDF TOC parsing (mocked PDF)
- ✅ `post_process_math()` - Math symbol conversion
- ✅ `url_to_filename()` - URL sanitization

#### `test_scrape_docs.py`
- ✅ `clean_doxygen_html()` - HTML cleanup
- ✅ `url_to_filename()` - Filename generation
- ✅ `load_config()` - JSON parsing
- ✅ `detect_category_type()` - Papers vs books

#### `test_rag_engine.py`
- ✅ `MultiIndexRetriever._retrieve()` - Query routing
- ⚠️  `get_embed_model()` - Model initialization (mock heavy deps)
- ⚠️  `get_llm()` - LLM config validation (mock Ollama)
- ⚠️  `get_reranker()` - Reranker initialization (mock model)

### 2. Integration Tests (Moderate, Local Resources)

#### `test_ingestion_pipeline.py`
- ⚙️  Fetch paper → Convert → Index (small test paper)
- ⚙️  Book splitting (TOC, manual, none)
- ⚙️  Config-based category rebuilding

#### `test_rag_pipeline.py`
- ⚙️  Load index → Query → Retrieve sources
- ⚙️  Multi-index retrieval merging
- ⚙️  Reranking effectiveness

#### `test_db_building.py`
- ⚙️  Build module with sample docs
- ⚙️  Hierarchical node parsing
- ⚙️  ChromaDB persistence

### 3. End-to-End Tests (Slow, Full System)

#### `test_app.py`
- 🔄 Session creation flow
- 🔄 Chat interaction (mocked LLM)
- 🔄 Preset save/load
- 🔄 Command processing (/load, /status, etc.)
- 🔄 Memory management

## Test Fixtures (conftest.py)

```python
# Sample fixtures to be created:

@pytest.fixture
def sample_paper_metadata():
    """Mock arXiv paper metadata"""

@pytest.fixture
def sample_pdf_path(tmp_path):
    """Generate a minimal test PDF"""

@pytest.fixture
def mock_ollama_api(responses):
    """Mock Ollama API responses"""

@pytest.fixture
def mock_chroma_db(tmp_path):
    """Temporary ChromaDB instance"""

@pytest.fixture
def sample_markdown_content():
    """Sample markdown for testing parsers"""

@pytest.fixture
def mock_embedding_model():
    """Mock HuggingFace embedding model"""
```

## Testing Priorities by Component

### High Priority (Critical Path)
1. ✅ **utils.py** - Core utilities, easy to test
2. ✅ **fetch_paper.py** - Data ingestion, many edge cases
3. ⚙️ **build_db.py** - Index building logic
4. 🔄 **app.py** - User-facing commands and session management

### Medium Priority
5. ⚠️ **rag_engine.py** - Requires mocking heavy ML models
6. ⚙️ Integration tests - End-to-end validation

### Lower Priority (Complex Mocking)
7. 🔄 **scrape_docs.py** - Network-dependent, best tested with fixtures
8. 🔄 Full E2E with Streamlit rendering

## Testing Best Practices

### Mocking Strategy
- **External APIs**: Mock all Ollama, arXiv, web scraping calls
- **ML Models**: Mock HuggingFace models, use small test embeddings
- **File System**: Use `tmp_path` fixtures for temp files
- **ChromaDB**: Use in-memory or temp directory instances

### Test Isolation
- Each test should be independent
- Use `setUp`/`tearDown` or fixtures for cleanup
- No shared state between tests

### Property-Based Testing
Use `hypothesis` for functions with complex input spaces:
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=100))
def test_clean_filename_never_crashes(title):
    result = clean_filename(title)
    assert isinstance(result, str)
    assert len(result) <= 50
```

## Coverage Targets

- **Overall**: 80%+
- **Utils**: 95%+ (pure functions, easy to test)
- **RAG Engine**: 70%+ (mock-heavy)
- **Integration**: 60%+ (system-dependent)

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tensortruth --cov-report=html

# Run specific category
pytest tests/unit/
pytest tests/integration/

# Run specific file
pytest tests/unit/test_utils.py

# Run with verbose output
pytest -v

# Run with markers
pytest -m "not slow"  # Skip slow tests

# Run failed tests only
pytest --lf

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest --cov=tensortruth --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Test Data Management

### Fixtures Directory Structure
```
tests/fixtures/
├── sample_papers/
│   ├── sample.pdf          # Minimal test PDF
│   └── sample.md           # Expected markdown output
├── sample_configs/
│   ├── papers.json         # Test paper config
│   └── books.json          # Test book config
├── mock_responses/
│   ├── ollama_models.json  # Mock Ollama API
│   └── arxiv_metadata.json # Mock arXiv response
└── sample_docs/
    ├── pytorch_sample.html
    └── numpy_sample.html
```

## Next Steps

1. ✅ Create base testing infrastructure (pytest.ini, conftest.py)
2. ✅ Implement high-priority unit tests (utils.py)
3. ⚙️ Add integration tests for core workflows
4. 🔄 Set up CI/CD pipeline
5. 📊 Establish coverage baseline and targets

## Legend
- ✅ Easy to implement, high value
- ⚙️ Moderate complexity, requires setup
- ⚠️ Requires extensive mocking
- 🔄 Complex, end-to-end scenarios
