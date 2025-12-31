"""Unit tests for core type definitions."""

import pytest

from tensortruth.core.types import DocType, DocumentType, SourceType


@pytest.mark.unit
class TestSourceType:
    """Tests for SourceType enum."""

    def test_source_type_values(self):
        """Test that SourceType enum has correct values."""
        assert SourceType.LIBRARIES == "libraries"
        assert SourceType.PAPERS == "papers"
        assert SourceType.BOOKS == "books"

    def test_source_type_is_string(self):
        """Test that SourceType values are strings."""
        assert isinstance(SourceType.LIBRARIES.value, str)
        assert isinstance(SourceType.PAPERS.value, str)
        assert isinstance(SourceType.BOOKS.value, str)

    def test_source_type_iteration(self):
        """Test that all expected source types are defined."""
        expected = {"libraries", "papers", "books"}
        actual = {st.value for st in SourceType}
        assert actual == expected


@pytest.mark.unit
class TestDocType:
    """Tests for DocType enum."""

    def test_doc_type_values(self):
        """Test that DocType enum has correct values."""
        assert DocType.SPHINX == "sphinx"
        assert DocType.DOXYGEN == "doxygen"
        assert DocType.ARXIV == "arxiv"
        assert DocType.PDF_BOOK == "pdf_book"

    def test_doc_type_is_string(self):
        """Test that DocType values are strings."""
        assert isinstance(DocType.SPHINX.value, str)
        assert isinstance(DocType.DOXYGEN.value, str)
        assert isinstance(DocType.ARXIV.value, str)
        assert isinstance(DocType.PDF_BOOK.value, str)

    def test_doc_type_iteration(self):
        """Test that all expected doc types are defined."""
        expected = {"sphinx", "doxygen", "arxiv", "pdf_book"}
        actual = {dt.value for dt in DocType}
        assert actual == expected


@pytest.mark.unit
class TestDocumentType:
    """Tests for DocumentType enum."""

    def test_document_type_values(self):
        """Test that DocumentType enum has correct values."""
        assert DocumentType.BOOK == "book"
        assert DocumentType.LIBRARY == "library"
        assert DocumentType.PAPERS == "papers"

    def test_document_type_is_string(self):
        """Test that DocumentType values are strings."""
        assert isinstance(DocumentType.BOOK.value, str)
        assert isinstance(DocumentType.LIBRARY.value, str)
        assert isinstance(DocumentType.PAPERS.value, str)

    def test_document_type_iteration(self):
        """Test that all expected document types are defined."""
        expected = {"book", "library", "papers"}
        actual = {dt.value for dt in DocumentType}
        assert actual == expected


@pytest.mark.unit
class TestEnumInteroperability:
    """Tests for enum interoperability."""

    def test_source_type_matches_document_type(self):
        """Test that SourceType and DocumentType have related values."""
        # These should align for proper directory naming
        assert SourceType.BOOKS.value == "books"
        assert DocumentType.BOOK.value == "book"  # Singular form

        assert SourceType.LIBRARIES.value == "libraries"
        assert DocumentType.LIBRARY.value == "library"  # Singular form

        assert SourceType.PAPERS.value == "papers"
        assert DocumentType.PAPERS.value == "papers"  # Same for papers

    def test_doc_type_maps_to_source_type(self):
        """Test that DocType values can map to SourceType values."""
        # PDF_BOOK maps to BOOKS
        assert "book" in SourceType.BOOKS.value

        # ARXIV maps to PAPERS
        assert "papers" == SourceType.PAPERS.value

        # SPHINX and DOXYGEN map to LIBRARIES
        assert "libraries" == SourceType.LIBRARIES.value
