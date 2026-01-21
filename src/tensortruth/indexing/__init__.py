"""Vector indexing module for Tensor-Truth.

This module provides functionality for building hierarchical vector indexes
from documentation sources with metadata extraction.
"""

from .builder import build_module, extract_metadata
from .metadata import (
    INDEX_METADATA_FILENAME,
    INDEX_VERSION,
    detect_legacy_index,
    get_available_embedding_models,
    get_embedding_model_from_index,
    get_embedding_model_id_from_index,
    is_valid_index_dir,
    list_indexes_for_model,
    read_index_metadata,
    sanitize_model_id,
    write_index_metadata,
)
from .migration import (
    check_and_migrate_on_startup,
    detect_legacy_indexes,
    get_migration_status,
    migrate_legacy_indexes,
)

__all__ = [
    "build_module",
    "extract_metadata",
    # Metadata utilities
    "INDEX_METADATA_FILENAME",
    "INDEX_VERSION",
    "detect_legacy_index",
    "get_available_embedding_models",
    "get_embedding_model_from_index",
    "get_embedding_model_id_from_index",
    "is_valid_index_dir",
    "list_indexes_for_model",
    "read_index_metadata",
    "sanitize_model_id",
    "write_index_metadata",
    # Migration utilities
    "check_and_migrate_on_startup",
    "detect_legacy_indexes",
    "get_migration_status",
    "migrate_legacy_indexes",
]
