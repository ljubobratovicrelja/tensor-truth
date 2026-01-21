"""Migration utilities for legacy index structures.

This module provides utilities to migrate indexes from the flat structure
(indexes/{module}/) to the versioned structure (indexes/{model_id}/{module}/).
"""

import logging
import shutil
from pathlib import Path
from typing import List, Tuple

from .metadata import (
    is_valid_index_dir,
    sanitize_model_id,
    write_index_metadata,
)

logger = logging.getLogger(__name__)

# The default embedding model used for legacy indexes (pre-versioned)
DEFAULT_LEGACY_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_LEGACY_CHUNK_SIZES = [2048, 512, 256]


def detect_legacy_indexes(indexes_dir: Path) -> List[Path]:
    """Detect indexes in flat structure that need migration.

    Indexes directly in indexes/{module}/ need to be moved to
    indexes/{model_id}/{module}/.

    Args:
        indexes_dir: Base indexes directory (e.g., ~/.tensortruth/indexes)

    Returns:
        List of paths to index directories that need migration
    """
    indexes_dir = Path(indexes_dir)
    if not indexes_dir.is_dir():
        return []

    legacy_indexes = []

    for item in indexes_dir.iterdir():
        if not item.is_dir():
            continue

        # If this directory directly contains chroma.sqlite3, it's in the
        # flat structure and needs migration (regardless of metadata)
        if is_valid_index_dir(item):
            legacy_indexes.append(item)

    return legacy_indexes


def migrate_legacy_indexes(
    indexes_dir: Path,
    target_model: str = DEFAULT_LEGACY_EMBEDDING_MODEL,
    dry_run: bool = False,
) -> Tuple[List[str], List[str]]:
    """Migrate legacy indexes to the versioned directory structure.

    Moves indexes from indexes/{module}/ to indexes/{model_id}/{module}/
    and writes metadata files.

    Args:
        indexes_dir: Base indexes directory
        target_model: Embedding model to associate with migrated indexes
        dry_run: If True, only report what would be migrated without moving

    Returns:
        Tuple of (migrated_modules, failed_modules)
    """
    indexes_dir = Path(indexes_dir)
    legacy_indexes = detect_legacy_indexes(indexes_dir)

    if not legacy_indexes:
        logger.info("No legacy indexes found to migrate")
        return [], []

    model_id = sanitize_model_id(target_model)
    target_dir = indexes_dir / model_id

    migrated = []
    failed = []

    for legacy_path in legacy_indexes:
        module_name = legacy_path.name
        new_path = target_dir / module_name

        logger.info(f"Migrating: {legacy_path} -> {new_path}")

        if dry_run:
            migrated.append(module_name)
            continue

        try:
            # Check if this index has metadata (might already know its model)
            from .metadata import read_index_metadata

            existing_metadata = read_index_metadata(legacy_path)

            # Determine target model from existing metadata or default
            if existing_metadata and existing_metadata.get("embedding_model_id"):
                actual_model_id = existing_metadata["embedding_model_id"]
                actual_target_dir = indexes_dir / actual_model_id
            else:
                actual_model_id = model_id
                actual_target_dir = target_dir

            new_path = actual_target_dir / module_name

            # Create target directory if needed
            actual_target_dir.mkdir(parents=True, exist_ok=True)

            # Check if destination already exists
            if new_path.exists():
                logger.warning(f"Destination already exists, skipping: {new_path}")
                failed.append(module_name)
                continue

            # Move the index directory
            shutil.move(str(legacy_path), str(new_path))

            # Only write metadata if it doesn't exist
            if not existing_metadata:
                write_index_metadata(
                    index_dir=new_path,
                    embedding_model=target_model,
                    chunk_sizes=DEFAULT_LEGACY_CHUNK_SIZES,
                    extra_metadata={"migrated_from_legacy": True},
                )

            migrated.append(module_name)
            logger.info(f"Successfully migrated: {module_name} -> {actual_model_id}/")

        except Exception as e:
            logger.error(f"Failed to migrate {module_name}: {e}")
            failed.append(module_name)

    return migrated, failed


def check_and_migrate_on_startup(indexes_dir: Path) -> bool:
    """Check for legacy indexes and migrate them on startup.

    This is designed to be called during application startup to
    automatically migrate any legacy indexes.

    Args:
        indexes_dir: Base indexes directory

    Returns:
        True if migration was performed, False if no migration needed
    """
    indexes_dir = Path(indexes_dir)
    legacy_indexes = detect_legacy_indexes(indexes_dir)

    if not legacy_indexes:
        return False

    logger.info(
        f"Found {len(legacy_indexes)} legacy indexes, starting automatic migration..."
    )

    migrated, failed = migrate_legacy_indexes(indexes_dir)

    if migrated:
        logger.info(f"Successfully migrated {len(migrated)} indexes: {migrated}")
    if failed:
        logger.warning(f"Failed to migrate {len(failed)} indexes: {failed}")

    return True


def get_migration_status(indexes_dir: Path) -> dict:
    """Get detailed status of index migration.

    Args:
        indexes_dir: Base indexes directory

    Returns:
        Dict with migration status information
    """
    indexes_dir = Path(indexes_dir)

    legacy_indexes = detect_legacy_indexes(indexes_dir)

    # Count versioned indexes by model
    versioned_by_model = {}
    for item in indexes_dir.iterdir():
        if not item.is_dir():
            continue
        # Skip if this is a legacy index
        if is_valid_index_dir(item):
            continue
        # This is a model directory
        model_id = item.name
        module_count = sum(
            1
            for subdir in item.iterdir()
            if subdir.is_dir() and is_valid_index_dir(subdir)
        )
        if module_count > 0:
            versioned_by_model[model_id] = module_count

    return {
        "has_legacy": len(legacy_indexes) > 0,
        "legacy_count": len(legacy_indexes),
        "legacy_modules": [p.name for p in legacy_indexes],
        "versioned_by_model": versioned_by_model,
        "total_versioned": sum(versioned_by_model.values()),
    }
