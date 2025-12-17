"""Unit tests for metadata extraction utilities."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from llama_index.core.schema import Document

from tensortruth.utils.metadata import (
    _parse_llm_json_response,
    classify_document_type,
    create_display_name,
    extract_document_metadata,
    extract_explicit_metadata,
    extract_metadata_with_llm,
    extract_pdf_metadata,
    extract_yaml_header_metadata,
    format_authors,
    get_source_url_for_arxiv,
    get_source_url_for_library,
)

# ============================================================================
# Test YAML Header Extraction
# ============================================================================


def test_extract_yaml_header_metadata_valid():
    """Test extraction from valid YAML header."""
    content = """# Title: Attention Is All You Need
# Authors: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
# Year: 2017
# ArXiv ID: 1706.03762

Abstract text here...
"""
    result = extract_yaml_header_metadata(content)

    assert result is not None
    assert result["title"] == "Attention Is All You Need"
    assert "Vaswani" in result["authors"]
    assert result["year"] == "2017"
    assert result["arxiv_id"] == "1706.03762"


def test_extract_yaml_header_metadata_partial():
    """Test extraction with partial header fields."""
    content = """# Title: Some Paper
# Year: 2023

Content here...
"""
    result = extract_yaml_header_metadata(content)

    assert result is not None
    assert result["title"] == "Some Paper"
    assert result["year"] == "2023"
    assert "authors" not in result


def test_extract_yaml_header_metadata_no_header():
    """Test with no YAML header present."""
    content = """This is just regular markdown content.
No metadata here.
"""
    result = extract_yaml_header_metadata(content)
    assert result is None


def test_extract_yaml_header_metadata_case_insensitive():
    """Test that header keys are case-insensitive."""
    content = """# TITLE: Test Title
# AUTHOR: Test Author
# ARXIV ID: 1234.5678
"""
    result = extract_yaml_header_metadata(content)

    assert result is not None
    assert result["title"] == "Test Title"
    assert result["authors"] == "Test Author"
    assert result["arxiv_id"] == "1234.5678"


# ============================================================================
# Test PDF Metadata Extraction
# ============================================================================


@pytest.mark.skipif(True, reason="Skipped unless testing with actual PDF files")
def test_extract_pdf_metadata_real_file():
    """Test PDF metadata extraction with a real file."""
    # This test requires a real PDF file to be present
    # Skipped by default
    pass


def test_extract_pdf_metadata_nonexistent_file():
    """Test PDF metadata extraction with nonexistent file."""
    fake_path = Path("/nonexistent/file.pdf")
    result = extract_pdf_metadata(fake_path)
    assert result is None


# ============================================================================
# Test Explicit Metadata Extraction
# ============================================================================


def test_extract_explicit_metadata_markdown():
    """Test explicit extraction from markdown file."""
    content = """# Title: Test Paper
# Authors: John Doe
# Year: 2023
"""
    doc = Document(text=content)
    file_path = Path("test.md")

    result = extract_explicit_metadata(doc, file_path)

    assert result is not None
    assert result["title"] == "Test Paper"
    assert result["authors"] == "John Doe"


def test_extract_explicit_metadata_no_metadata():
    """Test explicit extraction with no metadata."""
    content = "Regular content without metadata."
    doc = Document(text=content)
    file_path = Path("test.md")

    result = extract_explicit_metadata(doc, file_path)
    assert result is None


# ============================================================================
# Test LLM-based Extraction
# ============================================================================


def test_extract_metadata_with_llm_success():
    """Test LLM extraction with successful response."""
    content = """Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit

Abstract: We propose a new simple network architecture..."""

    doc = Document(text=content)
    file_path = Path("paper.md")

    # Mock Ollama API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": json.dumps(
            {"title": "Attention Is All You Need", "authors": "Vaswani et al."}
        )
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_response):
        result = extract_metadata_with_llm(doc, file_path, "http://localhost:11434")

    assert result["title"] == "Attention Is All You Need"
    assert result["authors"] == "Vaswani et al."


def test_extract_metadata_with_llm_timeout():
    """Test LLM extraction with timeout."""
    doc = Document(text="Some content")
    file_path = Path("test.md")

    with patch("requests.post", side_effect=Exception("Timeout")):
        result = extract_metadata_with_llm(doc, file_path, "http://localhost:11434")

    # Should return None values on failure
    assert result["title"] is None
    assert result["authors"] is None


def test_parse_llm_json_response_valid():
    """Test parsing valid JSON response."""
    response = '{"title": "Test Title", "authors": "Test Author"}'
    result = _parse_llm_json_response(response)

    assert result["title"] == "Test Title"
    assert result["authors"] == "Test Author"


def test_parse_llm_json_response_with_markdown():
    """Test parsing JSON response wrapped in markdown."""
    response = """Sure, here's the JSON:
```json
{"title": "Test Title", "authors": "Test Author"}
```
"""
    result = _parse_llm_json_response(response)

    assert result["title"] == "Test Title"
    assert result["authors"] == "Test Author"


def test_parse_llm_json_response_invalid():
    """Test parsing invalid JSON response."""
    response = "This is not JSON at all"
    result = _parse_llm_json_response(response)

    assert result["title"] is None
    assert result["authors"] is None


# ============================================================================
# Test Helper Functions
# ============================================================================


def test_get_source_url_for_arxiv():
    """Test ArXiv URL generation."""
    # New format
    url = get_source_url_for_arxiv("1706.03762")
    assert url == "https://arxiv.org/abs/1706.03762"

    # Old format
    url = get_source_url_for_arxiv("hep-th/9901001")
    assert url == "https://arxiv.org/abs/hep-th/9901001"


def test_get_source_url_for_library_found():
    """Test library URL lookup with valid module."""
    sources_config = {
        "libraries": {
            "pytorch": {
                "doc_root": "https://pytorch.org/docs/stable/",
                "version": "2.9",
            }
        }
    }

    url = get_source_url_for_library("pytorch", sources_config)
    assert url == "https://pytorch.org/docs/stable/"


def test_get_source_url_for_library_not_found():
    """Test library URL lookup with invalid module."""
    sources_config = {"libraries": {}}

    url = get_source_url_for_library("nonexistent", sources_config)
    assert url is None


def test_format_authors_single():
    """Test author formatting with single author."""
    result = format_authors("John Doe")
    assert result == "John Doe"


def test_format_authors_multiple():
    """Test author formatting with 2-3 authors."""
    result = format_authors("John Doe, Jane Smith")
    assert result == "John Doe, Jane Smith"

    result = format_authors("John Doe, Jane Smith, Bob Johnson")
    assert result == "John Doe, Jane Smith, Bob Johnson"


def test_format_authors_many():
    """Test author formatting with >3 authors."""
    authors = "John Doe, Jane Smith, Bob Johnson, Alice Williams, Charlie Brown"
    result = format_authors(authors)

    assert "et al." in result
    assert "Doe et al." == result


def test_format_authors_with_and():
    """Test author formatting with 'and' separator."""
    result = format_authors("John Doe and Jane Smith")
    assert result == "John Doe, Jane Smith"


def test_format_authors_last_name_extraction():
    """Test extraction of last name for et al. format."""
    # "Last, First" format
    result = format_authors("Doe, John, Smith, Jane, Johnson, Bob, Williams, Alice")
    assert result == "Doe et al."

    # "First Last" format
    result = format_authors("John Doe, Jane Smith, Bob Johnson, Alice Williams")
    assert result == "Doe et al."


def test_format_authors_none():
    """Test author formatting with None input."""
    result = format_authors(None)
    assert result is None


def test_format_authors_empty():
    """Test author formatting with empty string."""
    result = format_authors("")
    assert result is None


def test_format_authors_list_input():
    """Test author formatting with list input."""
    authors_list = ["John Doe", "Jane Smith"]
    result = format_authors(authors_list)
    assert result == "John Doe, Jane Smith"


def test_create_display_name_with_authors():
    """Test display name creation with authors."""
    result = create_display_name("Test Paper", "Doe et al.")
    assert result == "Test Paper - Doe et al."


def test_create_display_name_without_authors():
    """Test display name creation without authors."""
    result = create_display_name("Test Paper")
    assert result == "Test Paper"


def test_create_display_name_no_title():
    """Test display name creation with no title."""
    result = create_display_name(None)
    assert result == "Unknown Document"


def test_classify_document_type_uploaded_pdf():
    """Test document type classification for uploaded PDFs."""
    file_path = Path("pdf_abc123xyz.md")
    doc_type = classify_document_type(file_path)
    assert doc_type == "uploaded_pdf"


def test_classify_document_type_book():
    """Test document type classification for books."""
    file_path = Path("library_docs/deep_learning_books/chapter1.md")
    doc_type = classify_document_type(file_path)
    assert doc_type == "book"


def test_classify_document_type_paper():
    """Test document type classification for papers."""
    file_path = Path("library_docs/papers/arxiv_1234.md")
    doc_type = classify_document_type(file_path)
    assert doc_type == "paper"


def test_classify_document_type_library_doc():
    """Test document type classification for library docs."""
    file_path = Path("library_docs/pytorch_2.9/nn.md")
    doc_type = classify_document_type(file_path, module_name="pytorch")
    assert doc_type == "library_doc"


def test_classify_document_type_default():
    """Test document type classification default behavior."""
    file_path = Path("some_random_file.md")
    doc_type = classify_document_type(file_path)
    assert doc_type == "paper"  # Default


# ============================================================================
# Test Main Extraction Orchestrator
# ============================================================================


def test_extract_document_metadata_with_explicit():
    """Test full metadata extraction with explicit metadata."""
    content = """# Title: Test Paper
# Authors: John Doe, Jane Smith
# Year: 2023
# ArXiv ID: 1234.5678

Content here...
"""
    doc = Document(text=content)
    file_path = Path("test_paper.md")

    result = extract_document_metadata(
        doc, file_path, module_name="papers", use_llm_fallback=False
    )

    assert result["display_name"] == "Test Paper - John Doe, Jane Smith"
    assert result["authors"] == "John Doe, Jane Smith"
    assert result["source_url"] == "https://arxiv.org/abs/1234.5678"
    assert result["doc_type"] == "paper"
    assert result["arxiv_id"] == "1234.5678"
    assert result["year"] == "2023"


def test_extract_document_metadata_with_llm_fallback():
    """Test full metadata extraction with LLM fallback."""
    content = """Attention Is All You Need

Ashish Vaswani, Noam Shazeer

Abstract: We propose..."""

    doc = Document(text=content)
    file_path = Path("paper.md")

    # Mock Ollama API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": json.dumps(
            {
                "title": "Attention Is All You Need",
                "authors": "Vaswani, Shazeer",
            }
        )
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_response):
        result = extract_document_metadata(
            doc,
            file_path,
            ollama_url="http://localhost:11434",
            use_llm_fallback=True,
        )

    assert "Attention Is All You Need" in result["display_name"]
    assert result["authors"] == "Vaswani, Shazeer"
    assert result["doc_type"] == "paper"


def test_extract_document_metadata_fallback_to_filename():
    """Test metadata extraction with fallback to filename."""
    content = "No metadata here at all."
    doc = Document(text=content)
    file_path = Path("My_Test_Document.md")

    result = extract_document_metadata(
        doc, file_path, use_llm_fallback=False, ollama_url=None
    )

    # Should use filename as fallback
    assert "My Test Document" in result["display_name"]
    assert result["source_url"] is None


def test_extract_document_metadata_with_library_url():
    """Test metadata extraction with library module URL."""
    content = """# Title: PyTorch Neural Networks
# Authors: PyTorch Team
"""
    doc = Document(text=content)
    file_path = Path("pytorch_docs/nn.md")

    sources_config = {
        "libraries": {"pytorch": {"doc_root": "https://pytorch.org/docs/stable/"}}
    }

    result = extract_document_metadata(
        doc,
        file_path,
        module_name="pytorch",
        sources_config=sources_config,
        use_llm_fallback=False,
    )

    assert result["display_name"] == "PyTorch Neural Networks - PyTorch Team"
    assert result["source_url"] == "https://pytorch.org/docs/stable/"


def test_extract_document_metadata_uploaded_pdf():
    """Test metadata extraction for uploaded PDF."""
    content = """# Document: attention.pdf
# Source: Session Upload

Attention Is All You Need

Vaswani et al.

Abstract..."""

    doc = Document(text=content)
    file_path = Path("pdf_abc123.md")

    # Mock LLM response
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": json.dumps(
            {"title": "Attention Is All You Need", "authors": "Vaswani et al."}
        )
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_response):
        result = extract_document_metadata(
            doc,
            file_path,
            ollama_url="http://localhost:11434",
            use_llm_fallback=True,
        )

    assert result["display_name"] == "Attention Is All You Need - Vaswani et al."
    assert result["doc_type"] == "uploaded_pdf"
    assert result["source_url"] is None  # No URL for uploaded PDFs
