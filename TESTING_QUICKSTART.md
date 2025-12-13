# Testing Quick Start Guide

## Install & Run

```bash
# 1. Install testing dependencies
pip install -r requirements-dev.txt

# 2. Run all tests
pytest

# 3. Run with coverage
pytest --cov=tensortruth --cov-report=html

# 4. View coverage report
open htmlcov/index.html
```

## Common Commands

```bash
# Run only unit tests (fast)
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_utils.py

# Run specific test class
pytest tests/unit/test_utils.py::TestParseThinkingResponse

# Run specific test method
pytest tests/unit/test_utils.py::TestParseThinkingResponse::test_standard_thinking_tags

# Skip slow tests
pytest -m "not slow"

# Run tests in parallel (faster)
pytest -n auto

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Verbose output with full diffs
pytest -vv

# Generate coverage and open HTML report
pytest --cov=tensortruth --cov-report=html && open htmlcov/index.html
```

## Test Structure

```
tests/
├── unit/               # Fast, isolated tests
│   ├── test_utils.py
│   ├── test_fetch_paper.py
│   └── test_rag_engine.py
├── integration/        # Component integration tests
│   └── test_ingestion_pipeline.py
└── conftest.py        # Shared fixtures
```

## Writing a New Test

```python
import pytest
from tensortruth.module import function_to_test

@pytest.mark.unit
def test_my_function(temp_dir):
    """Test description."""
    # Arrange
    input_data = "test"

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected_output
```

## Using Fixtures

```python
@pytest.mark.unit
def test_with_fixture(sample_paper_metadata, temp_dir):
    """Use fixtures for common setup."""
    # sample_paper_metadata and temp_dir are available
    assert sample_paper_metadata["arxiv_id"] == "1234.56789"
```

## Mocking External APIs

```python
from unittest.mock import patch

@pytest.mark.unit
@patch('tensortruth.utils.requests.get')
def test_api_call(mock_get):
    """Mock external API calls."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"data": "test"}

    result = function_that_calls_api()
    assert result is not None
```

## Test Markers

```python
@pytest.mark.unit           # Fast unit test
@pytest.mark.integration    # Integration test
@pytest.mark.slow           # Slow test
@pytest.mark.requires_gpu   # Needs CUDA/MPS
@pytest.mark.requires_ollama # Needs Ollama running
```

## Coverage Goals

- **Overall**: 80%+
- **Utils**: 95%+
- **Core modules**: 85%+

## Current Status

✅ 60 tests passing
⚠️ 3 tests need fixing (memory detection mocks)
📈 Coverage measurement ready

## Documentation

- Full strategy: [TESTING.md](TESTING.md)
- Detailed guide: [tests/README.md](tests/README.md)

## CI/CD

Tests run automatically on:
- Push to main/develop
- Pull requests
- Python 3.9, 3.10, 3.11
- Ubuntu & macOS

See results in GitHub Actions tab.
