"""Validation utilities for tensor-truth CLI commands."""

import logging
import os

logger = logging.getLogger(__name__)


def validate_module_for_build(module_name, library_docs_dir, sources_config):
    """
    Validate module exists in filesystem and optionally in config.

    Args:
        module_name: Module to validate
        library_docs_dir: Base directory for docs
        sources_config: Loaded sources.json config

    Raises:
        ValueError if validation fails
    """
    # Check filesystem
    from .metadata import get_document_type_from_config

    doc_type = get_document_type_from_config(module_name, sources_config)

    source_dir = os.path.join(library_docs_dir, f"{doc_type.value}_{module_name}")
    if not os.path.exists(source_dir):
        raise ValueError(
            f"Module '{module_name}' not found in {library_docs_dir}.\n"
            f"Run: tensor-truth-docs {module_name}"
        )

    # Check if docs directory is empty
    if not any(os.scandir(source_dir)):
        raise ValueError(f"Module '{module_name}' directory is empty: {source_dir}")

    # Warn if not in config (not fatal)
    all_sources = {
        **sources_config.get("libraries", {}),
        **sources_config.get("papers", {}),
        **sources_config.get("books", {}),
    }

    if module_name not in all_sources:
        logger.warning(
            f"Module '{module_name}' not found in sources config. "
            f"Metadata may be incomplete."
        )


def validate_sources(sources_config_path, library_docs_dir):
    """
    Validate sources.json against filesystem.

    Reports:
    - Sources in config without docs on disk
    - Docs on disk not in config (orphaned)
    - Validation errors

    Args:
        sources_config_path: Path to sources.json
        library_docs_dir: Path to library_docs directory
    """
    # Import here to avoid circular dependency
    from .sources_config import load_user_sources

    config = load_user_sources(sources_config_path)

    print("\n=== Validating Sources ===\n")

    # Check libraries
    for lib_name, lib_config in config.get("libraries", {}).items():
        version = lib_config.get("version", "")
        dir_name = f"{lib_name}_{version}" if version else lib_name
        path = os.path.join(library_docs_dir, dir_name)

        if os.path.exists(path):
            print(f"✓ {lib_name}: Found at {path}")
        else:
            print(f"✗ {lib_name}: NOT FOUND (run: tensor-truth-docs {lib_name})")

    # Check papers
    for cat_name in config.get("papers", {}).keys():
        path = os.path.join(library_docs_dir, cat_name)
        if os.path.exists(path):
            print(f"✓ {cat_name} (papers): Found at {path}")
        else:
            print(f"✗ {cat_name} (papers): NOT FOUND")

    # Check for orphaned directories
    print("\n=== Orphaned Directories (not in config) ===\n")
    if os.path.exists(library_docs_dir):
        all_dirs = {
            d
            for d in os.listdir(library_docs_dir)
            if os.path.isdir(os.path.join(library_docs_dir, d))
        }

        config_dirs = set()
        for lib_name, lib_config in config.get("libraries", {}).items():
            version = lib_config.get("version", "")
            config_dirs.add(f"{lib_name}_{version}" if version else lib_name)
        config_dirs.update(config.get("papers", {}).keys())

        orphaned = all_dirs - config_dirs
        if orphaned:
            for dirname in orphaned:
                print(f"⚠️  {dirname}/: Not in config (orphaned)")
        else:
            print("(none)")
    else:
        print(f"Library docs directory does not exist: {library_docs_dir}")
