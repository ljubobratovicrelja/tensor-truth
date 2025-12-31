"""Vector indexing module for Tensor-Truth.

This module provides functionality for building hierarchical vector indexes
from documentation sources with metadata extraction.
"""

from .builder import build_module, extract_metadata

__all__ = [
    "build_module",
    "extract_metadata",
]
