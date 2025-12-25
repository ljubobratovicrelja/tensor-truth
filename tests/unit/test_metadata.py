"""Unit tests for metadata extraction utilities."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from llama_index.core.schema import Document

from tensortruth.utils.metadata import (
    _parse_llm_json_response,
    extract_arxiv_metadata_from_config,
    extract_explicit_metadata,
    extract_metadata_with_llm,
    extract_pdf_metadata,
    extract_yaml_header_metadata,
    format_authors,
    get_document_type_from_config,
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


# ============================================================================
# Test Document Type Detection
# ============================================================================


def test_get_document_type_from_config_library():
    """Test document type detection for libraries."""
    sources_config = {"libraries": {"pytorch": {"type": "sphinx", "version": "2.9"}}}
    from tensortruth.utils.metadata import DocumentType

    assert (
        get_document_type_from_config("pytorch", sources_config) == DocumentType.LIBRARY
    )


def test_get_document_type_from_config_library_with_version():
    """Test module name must match config key exactly (no version stripping)."""
    sources_config = {"libraries": {"pytorch": {"type": "sphinx"}}}
    # Module name must match config key exactly - version stripping not supported
    with pytest.raises(ValueError, match="is not found among sources"):
        get_document_type_from_config("pytorch_2.9", sources_config)


def test_get_document_type_from_config_paper_arxiv():
    """Test document type for ArXiv papers."""
    sources_config = {"papers": {"dl_foundations": {"type": "arxiv", "items": {}}}}
    from tensortruth.utils.metadata import DocumentType

    assert (
        get_document_type_from_config("dl_foundations", sources_config)
        == DocumentType.PAPERS
    )


def test_get_document_type_from_config_book():
    """Test document type for books."""
    sources_config = {"books": {"book_linear_algebra_cherney": {"type": "pdf_book"}}}
    from tensortruth.utils.metadata import DocumentType

    assert (
        get_document_type_from_config("book_linear_algebra_cherney", sources_config)
        == DocumentType.BOOK
    )


def test_get_document_type_from_config_default():
    """Test raises error for unknown modules."""
    sources_config = {"libraries": {}, "papers": {}, "books": {}}
    with pytest.raises(ValueError, match="is not found among sources"):
        get_document_type_from_config("unknown_module", sources_config)


# ============================================================================
# Test ArXiv Metadata Extraction
# ============================================================================


def test_extract_arxiv_metadata_from_config_found():
    """Test ArXiv metadata extraction when paper is in config."""
    file_path = Path("1512.03385.pdf")
    sources_config = {
        "papers": {
            "dl_foundations": {
                "items": {
                    "1512.03385": {
                        "title": "Deep Residual Learning",
                        "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
                        "year": "2015",
                        "url": "https://arxiv.org/abs/1512.03385",
                    }
                }
            }
        }
    }

    result = extract_arxiv_metadata_from_config(
        file_path, "dl_foundations", sources_config
    )

    assert result is not None
    assert result["title"] == "Deep Residual Learning"
    assert (
        result["authors"] == "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun"
    )  # Should be formatted
    assert result["display_name"] == "Deep Residual Learning, He et al."
    assert result["source_url"] == "https://arxiv.org/abs/1512.03385"
    assert result["doc_type"] == "paper"


def test_extract_arxiv_metadata_from_config_not_found():
    """Test when ArXiv ID not in config."""
    file_path = Path("9999.99999.pdf")
    sources_config = {"papers": {"dl_foundations": {"items": {}}}}

    with pytest.raises(ValueError):
        extract_arxiv_metadata_from_config(file_path, "dl_foundations", sources_config)


def test_extract_arxiv_metadata_from_config_invalid_filename():
    """Test with non-ArXiv filename."""
    file_path = Path("regular_paper.pdf")
    sources_config = {
        "papers": {
            "dl_foundations": {
                "items": {
                    "1512.03385": {
                        "title": "Deep Residual Learning",
                        "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
                        "year": "2015",
                        "url": "https://arxiv.org/abs/1512.03385",
                    }
                }
            }
        }
    }

    with pytest.raises(ValueError):
        extract_arxiv_metadata_from_config(file_path, "dl_foundations", sources_config)
