"""Index metadata management for versioned embedding indexes.

This module provides utilities for managing index metadata, including:
- Model ID sanitization for filesystem-safe paths
- Index metadata reading/writing
- Embedding model detection from existing indexes
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

INDEX_METADATA_FILENAME = "index_metadata.json"
INDEX_VERSION = "1.0"


def sanitize_model_id(model_name: str) -> str:
    """Convert HuggingFace model name to filesystem-safe ID.

    Extracts the model identifier from a full HuggingFace path and makes it
    safe for use in directory names.

    Args:
        model_name: Full model path (e.g., "BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B")

    Returns:
        Sanitized model ID (e.g., "bge-m3", "qwen3-embedding-0.6b")

    Examples:
        >>> sanitize_model_id("BAAI/bge-m3")
        'bge-m3'
        >>> sanitize_model_id("Qwen/Qwen3-Embedding-0.6B")
        'qwen3-embedding-0.6b'
        >>> sanitize_model_id("sentence-transformers/all-MiniLM-L6-v2")
        'all-minilm-l6-v2'
    """
    # Extract just the model name if it includes org prefix
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    # Convert to lowercase and replace unsafe characters
    model_id = model_name.lower()
    model_id = re.sub(r"[^a-z0-9\-_.]", "-", model_id)
    # Remove consecutive dashes
    model_id = re.sub(r"-+", "-", model_id)
    # Remove leading/trailing dashes
    model_id = model_id.strip("-")

    return model_id


def write_index_metadata(
    index_dir: Path,
    embedding_model: str,
    chunk_sizes: List[int],
    chunk_overlap: Optional[int] = None,
    chunking_strategy: Optional[str] = None,
    version: str = INDEX_VERSION,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write metadata file to an index directory.

    Creates an index_metadata.json file containing information about how
    the index was built, enabling future compatibility checks.

    Args:
        index_dir: Path to the index directory
        embedding_model: Full HuggingFace model path (e.g., "BAAI/bge-m3")
        chunk_sizes: List of hierarchical chunk sizes used during indexing
        chunk_overlap: Overlap tokens between chunks (for reproducibility)
        chunking_strategy: Strategy used for chunking (hierarchical, semantic, etc.)
        version: Index format version (default: current INDEX_VERSION)
        extra_metadata: Optional additional metadata to include
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "embedding_model": embedding_model,
        "embedding_model_id": sanitize_model_id(embedding_model),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_version": version,
        "chunk_sizes": chunk_sizes,
        "chunk_overlap": chunk_overlap,
        "chunking_strategy": chunking_strategy,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = index_dir / INDEX_METADATA_FILENAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.debug(f"Wrote index metadata to {metadata_path}")


def read_index_metadata(index_dir: Path) -> Optional[Dict[str, Any]]:
    """Read metadata from an index directory.

    Args:
        index_dir: Path to the index directory

    Returns:
        Metadata dictionary if found, None otherwise
    """
    index_dir = Path(index_dir)
    metadata_path = index_dir / INDEX_METADATA_FILENAME

    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read index metadata from {metadata_path}: {e}")
        return None


def get_embedding_model_from_index(index_dir: Path) -> Optional[str]:
    """Get the embedding model used to build an index.

    Args:
        index_dir: Path to the index directory

    Returns:
        Full embedding model path if metadata exists, None otherwise
    """
    metadata = read_index_metadata(index_dir)
    if metadata:
        return metadata.get("embedding_model")
    return None


def get_embedding_model_id_from_index(index_dir: Path) -> Optional[str]:
    """Get the sanitized embedding model ID from an index.

    Args:
        index_dir: Path to the index directory

    Returns:
        Sanitized model ID if metadata exists, None otherwise
    """
    metadata = read_index_metadata(index_dir)
    if metadata:
        return metadata.get("embedding_model_id")
    return None


def is_valid_index_dir(index_dir: Path) -> bool:
    """Check if a directory contains a valid vector index.

    Validates that the directory contains the expected ChromaDB files.

    Args:
        index_dir: Path to check

    Returns:
        True if directory contains a valid index
    """
    index_dir = Path(index_dir)
    if not index_dir.is_dir():
        return False

    # Check for ChromaDB marker file
    chroma_sqlite = index_dir / "chroma.sqlite3"
    return chroma_sqlite.exists()


def detect_legacy_index(index_dir: Path) -> bool:
    """Detect if an index directory is a legacy (pre-versioned) index.

    Legacy indexes lack metadata and are stored directly in indexes/{module_name}/
    rather than indexes/{embedding_model_id}/{module_name}/.

    Args:
        index_dir: Path to the index directory

    Returns:
        True if this appears to be a legacy index
    """
    if not is_valid_index_dir(index_dir):
        return False

    # Legacy indexes don't have metadata
    return read_index_metadata(index_dir) is None


def list_indexes_for_model(
    indexes_base_dir: Path, embedding_model_id: str
) -> List[Path]:
    """List all module indexes for a specific embedding model.

    Args:
        indexes_base_dir: Base indexes directory (e.g., ~/.tensortruth/indexes)
        embedding_model_id: Sanitized model ID (e.g., "bge-m3")

    Returns:
        List of paths to valid index directories
    """
    indexes_base_dir = Path(indexes_base_dir)
    model_dir = indexes_base_dir / embedding_model_id

    if not model_dir.is_dir():
        return []

    return [
        subdir
        for subdir in model_dir.iterdir()
        if subdir.is_dir() and is_valid_index_dir(subdir)
    ]


def get_available_embedding_models(indexes_base_dir: Path) -> List[Dict[str, Any]]:
    """Get information about all available embedding models with indexes.

    Args:
        indexes_base_dir: Base indexes directory

    Returns:
        List of dicts with model info: {
            "model_id": str,
            "model_name": str or None,  # Full name if metadata available
            "index_count": int,
            "modules": List[str]
        }
    """
    indexes_base_dir = Path(indexes_base_dir)
    if not indexes_base_dir.is_dir():
        return []

    result = []

    for model_dir in indexes_base_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # Skip if this looks like a legacy index (module directly in indexes/)
        if is_valid_index_dir(model_dir):
            continue

        model_id = model_dir.name
        indexes = list_indexes_for_model(indexes_base_dir, model_id)

        if not indexes:
            continue

        # Try to get full model name from one of the indexes
        model_name = None
        modules = []
        for idx in indexes:
            modules.append(idx.name)
            if model_name is None:
                model_name = get_embedding_model_from_index(idx)

        result.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "index_count": len(indexes),
                "modules": modules,
            }
        )

    return result
