# Tests for Tensor-Truth

This directory contains the test suite for the tensor-truth RAG pipeline project.

## Quick Start

### Install Test Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tensortruth --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Skip slow tests
pytest -m "not slow"

# Run tests requiring GPU
pytest -m requires_gpu

# Run tests requiring Ollama
pytest -m requires_ollama
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Fast, isolated unit tests
│   ├── test_utils.py
│   ├── test_fetch_paper.py
│   └── test_rag_engine.py
├── integration/             # Integration tests
│   └── test_ingestion_pipeline.py
├── e2e/                     # End-to-end tests (future)
└── fixtures/                # Test data
    ├── sample_papers/
    ├── sample_configs/
    ├── mock_responses/
    └── sample_docs/
```

## Writing Tests

### Unit Test Example

```python
import pytest
from tensortruth.utils import parse_thinking_response

@pytest.mark.unit
def test_parse_thinking_standard():
    """Test standard thinking tag parsing."""
    raw = "<thought>Thinking</thought>Answer"
    thought, answer = parse_thinking_response(raw)

    assert thought == "Thinking"
    assert answer == "Answer"
```

### Using Fixtures

```python
@pytest.mark.unit
def test_with_sample_data(sample_paper_metadata):
    """Test using fixture data."""
    assert sample_paper_metadata["arxiv_id"] == "1234.56789"
```

### Mocking External Dependencies

```python
from unittest.mock import patch

@pytest.mark.unit
@patch('tensortruth.utils.requests.get')
def test_api_call(mock_get):
    """Test with mocked API call."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"models": []}

    result = get_running_models()
    assert result == []
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st

@pytest.mark.unit
@given(st.text())
def test_clean_filename_never_crashes(title):
    """Test that function handles any input."""
    result = clean_filename(title)
    assert isinstance(result, str)
    assert len(result) <= 50
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.requires_gpu` - Tests requiring CUDA/MPS
- `@pytest.mark.requires_ollama` - Tests requiring Ollama server
- `@pytest.mark.requires_network` - Tests requiring internet

## Available Fixtures

See [conftest.py](conftest.py) for all available fixtures:

### Path Fixtures
- `fixtures_dir` - Path to test fixtures directory
- `temp_dir` - Temporary directory for test files
- `temp_index_dir` - Temporary directory for test indexes
- `temp_library_dir` - Temporary directory for library docs

### Mock Data Fixtures
- `sample_paper_metadata` - Mock arXiv paper metadata
- `sample_chat_session` - Sample chat session data
- `sample_thinking_response` - Response with thinking tags
- `sample_markdown_content` - Sample markdown content

### Mock API Fixtures
- `mock_ollama_models_response` - Mock Ollama /api/tags response
- `mock_ollama_ps_response` - Mock Ollama /api/ps response
- `mock_requests_get` - Mocked requests.get

### Mock ML Model Fixtures
- `mock_embedding_model` - Mock HuggingFace embedding model
- `mock_llm` - Mock Ollama LLM
- `mock_reranker` - Mock sentence transformer reranker

### Environment Fixtures
- `mock_cuda_available` - Mock CUDA as available
- `mock_cuda_unavailable` - Mock CUDA as unavailable
- `mock_mps_available` - Mock MPS as available

### Factory Fixtures
- `create_test_pdf(filename, content)` - Create test PDF files
- `create_test_markdown(filename, content)` - Create test markdown files

## Coverage Goals

We aim for the following coverage targets:

- **Overall**: 80%+
- **Utils Module**: 95%+ (pure functions, easy to test)
- **Fetch Paper**: 85%+
- **RAG Engine**: 70%+ (requires extensive mocking)
- **Integration**: 60%+ (system-dependent)

Current coverage:

```bash
# Generate coverage report
pytest --cov=tensortruth --cov-report=term-missing
```

## Continuous Integration

Tests run automatically on GitHub Actions for:
- Python 3.9, 3.10, 3.11
- Ubuntu and macOS
- On push to main/develop branches
- On pull requests

See [.github/workflows/tests.yml](../.github/workflows/tests.yml) for CI configuration.

## Debugging Failed Tests

### Run with verbose output

```bash
pytest -vv
```

### Run specific test

```bash
pytest tests/unit/test_utils.py::TestParseThinkingResponse::test_standard_thinking_tags
```

### Drop into debugger on failure

```bash
pytest --pdb
```

### Show print statements

```bash
pytest -s
```

### Run last failed tests only

```bash
pytest --lf
```

### Run failed tests first

```bash
pytest --ff
```

## Performance Testing

### Run tests in parallel

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run with auto-detected CPU count
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

### Profile slow tests

```bash
pytest --durations=10
```

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on other tests
2. **Use Fixtures**: Leverage fixtures for common setup/teardown
3. **Mock External Dependencies**: Mock APIs, file system, and ML models
4. **Descriptive Names**: Use clear, descriptive test function names
5. **One Assertion Focus**: Each test should verify one specific behavior
6. **Fast Tests**: Keep unit tests fast; mark slow tests appropriately
7. **Documentation**: Add docstrings to explain what each test verifies

## Troubleshooting

### Import Errors

If you get import errors, ensure the package is installed in development mode:

```bash
pip install -e .
```

### Missing Dependencies

Install all development dependencies:

```bash
pip install -r requirements-dev.txt
```

### ChromaDB Issues

Some integration tests require ChromaDB. If you encounter issues:

```bash
pip install --upgrade chromadb
```

### CUDA/MPS Tests

Tests marked with `requires_gpu` need CUDA or MPS. Skip them on CPU-only systems:

```bash
pytest -m "not requires_gpu"
```

## Contributing

When adding new features:

1. Write tests first (TDD approach recommended)
2. Ensure tests pass locally
3. Maintain or improve coverage
4. Add appropriate test markers
5. Update test documentation if needed

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [hypothesis documentation](https://hypothesis.readthedocs.io/)
- [pytest-mock documentation](https://pytest-mock.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
