"""Core type definitions for Tensor-Truth.

This module defines enums and type constants used across the codebase
to ensure consistency and prevent duplication.
"""

from enum import Enum


class SourceType(str, Enum):
    """Source configuration section names in sources.json.

    Used to categorize different types of documentation sources.
    """

    LIBRARIES = "libraries"
    PAPERS = "papers"
    BOOKS = "books"


class DocType(str, Enum):
    """Documentation generator/format types.

    Identifies the tool or format used to generate the documentation,
    which determines the scraping strategy.
    """

    SPHINX = "sphinx"
    DOXYGEN = "doxygen"
    ARXIV = "arxiv"
    PDF_BOOK = "pdf_book"


class DocumentType(str, Enum):
    """High-level document type categories.

    Used for metadata extraction and index organization.
    Maps to SourceType but represents the semantic type of content.
    """

    BOOK = "book"
    LIBRARY = "library"
    PAPERS = "papers"
