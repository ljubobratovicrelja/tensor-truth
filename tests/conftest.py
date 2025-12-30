"""
Pytest configuration and shared fixtures.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_configs_dir(fixtures_dir):
    """Return path to sample configs directory."""
    return fixtures_dir / "sample_configs"


@pytest.fixture
def mock_responses_dir(fixtures_dir):
    """Return path to mock responses directory."""
    return fixtures_dir / "mock_responses"


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def temp_index_dir(tmp_path):
    """Provide a temporary directory for test indexes."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    return index_dir


@pytest.fixture
def temp_library_dir(tmp_path):
    """Provide a temporary directory for library docs."""
    lib_dir = tmp_path / "library_docs"
    lib_dir.mkdir()
    return lib_dir


# ============================================================================
# Mock Data Fixtures
# ============================================================================


@pytest.fixture
def sample_paper_metadata():
    """Mock arXiv paper metadata."""
    return {
        "title": "Test Paper: Attention Is All You Need",
        "authors": ["Author One", "Author Two"],
        "abstract": "This is a test abstract for testing purposes.",
        "arxiv_id": "1234.56789",
        "year": 2023,
        "source": "https://arxiv.org/abs/1234.56789",
    }


@pytest.fixture
def sample_chat_session():
    """Sample chat session data."""
    return {
        "title": "Test Session",
        "created_at": "2023-12-13T10:00:00",
        "messages": [
            {"role": "user", "content": "What is PyTorch?"},
            {
                "role": "assistant",
                "content": "PyTorch is a deep learning framework.",
                "sources": [{"file": "pytorch_intro.md", "score": 0.95}],
                "time_taken": 2.5,
            },
        ],
        "modules": ["pytorch"],
        "params": {
            "model": "deepseek-r1:8b",
            "temperature": 0.3,
            "context_window": 4096,
        },
    }


@pytest.fixture
def sample_thinking_response():
    """Sample response with thinking tags."""
    return """<thought>
Let me analyze this question step by step.
First, I need to understand what the user is asking.
Then, I'll formulate a clear response.
</thought>

Here is the actual answer to your question."""


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for testing."""
    return """# Test Document

This is a test document with some **bold** and *italic* text.

## Code Example

```python
def hello():
    print("Hello, World!")
```

## Math

The formula is: $E = mc^2$

Some symbols: α β γ × ÷ ≤ ≥
"""


# ============================================================================
# Mock API Fixtures
# ============================================================================


@pytest.fixture
def mock_ollama_models_response():
    """Mock Ollama /api/tags response."""
    return {
        "models": [
            {
                "name": "deepseek-r1:8b",
                "size": 5500000000,
                "digest": "abc123",
                "modified_at": "2023-12-13T10:00:00Z",
            },
            {
                "name": "llama2:7b",
                "size": 3800000000,
                "digest": "def456",
                "modified_at": "2023-12-13T10:00:00Z",
            },
        ]
    }


@pytest.fixture
def mock_ollama_ps_response():
    """Mock Ollama /api/ps response."""
    return {
        "models": [
            {
                "name": "deepseek-r1:8b",
                "size_vram": 5500000000,
                "expires_at": "2023-12-13T11:00:00Z",
            }
        ]
    }


@pytest.fixture
def mock_requests_get(monkeypatch):
    """Mock requests.get for API calls."""

    def mock_get(url, *args, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200

        if "tags" in url:
            mock_response.json.return_value = {"models": [{"name": "deepseek-r1:8b"}]}
        elif "ps" in url:
            mock_response.json.return_value = {"models": []}
        else:
            mock_response.json.return_value = {}

        return mock_response

    monkeypatch.setattr("requests.get", mock_get)
    return mock_get


# ============================================================================
# Mock ML Model Fixtures
# ============================================================================


@pytest.fixture
def mock_embedding_model():
    """Mock HuggingFace embedding model."""
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1] * 1024  # Mock embedding vector
    mock_model.embed_documents.return_value = [[0.1] * 1024]
    return mock_model


@pytest.fixture
def mock_llm():
    """Mock Ollama LLM."""
    mock = MagicMock()
    mock.complete.return_value.text = "This is a test response."
    return mock


@pytest.fixture
def mock_reranker():
    """Mock sentence transformer reranker."""
    mock = MagicMock()
    mock.postprocess_nodes.return_value = []
    return mock


# ============================================================================
# Mock ChromaDB Fixtures
# ============================================================================


@pytest.fixture
def mock_chroma_client(temp_index_dir):
    """Mock ChromaDB client with temporary storage."""
    import chromadb

    client = chromadb.PersistentClient(path=str(temp_index_dir))
    return client


# ============================================================================
# Environment Fixtures
# ============================================================================


@pytest.fixture
def mock_cuda_available(monkeypatch):
    """Mock torch.cuda.is_available() to return True."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def mock_cuda_unavailable(monkeypatch):
    """Mock torch.cuda.is_available() to return False."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def mock_mps_available(monkeypatch):
    """Mock torch.backends.mps.is_available() to return True."""
    import torch

    if not hasattr(torch.backends, "mps"):
        # Create mock mps module if it doesn't exist
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        monkeypatch.setattr(torch.backends, "mps", mock_mps)
    else:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


# ============================================================================
# Config File Fixtures
# ============================================================================


@pytest.fixture
def sample_papers_config():
    """Sample papers configuration."""
    return {
        "papers": {
            "description": "Test papers category",
            "items": [
                {
                    "title": "Attention Is All You Need",
                    "arxiv_id": "1706.03762",
                    "source": "https://arxiv.org/abs/1706.03762",
                }
            ],
        }
    }


@pytest.fixture
def sample_books_config():
    """Sample books configuration."""
    return {
        "linear_algebra": {
            "description": "Linear algebra textbooks",
            "items": [
                {
                    "title": "Introduction to Linear Algebra",
                    "source": "https://example.com/linear_algebra.pdf",
                    "split_method": "none",
                }
            ],
        }
    }


# ============================================================================
# Test Data Generators
# ============================================================================


@pytest.fixture
def create_test_pdf(temp_dir):
    """Factory fixture to create minimal test PDFs."""

    def _create_pdf(filename="test.pdf", content="Test PDF Content"):
        try:
            import fitz  # PyMuPDF

            pdf_path = temp_dir / filename

            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), content)
            doc.save(str(pdf_path))
            doc.close()

            return pdf_path
        except ImportError:
            # If PyMuPDF not available, create dummy file
            pdf_path = temp_dir / filename
            pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
            return pdf_path

    return _create_pdf


@pytest.fixture
def create_test_markdown(temp_dir):
    """Factory fixture to create test markdown files."""

    def _create_markdown(filename="test.md", content="# Test\n\nTest content"):
        md_path = temp_dir / filename
        md_path.write_text(content, encoding="utf-8")
        return md_path

    return _create_markdown


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup happens automatically with tmp_path fixture
    pass
