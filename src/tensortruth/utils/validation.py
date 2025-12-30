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
    - Config schema validation errors
    - Deprecated field usage

    Args:
        sources_config_path: Path to sources.json
        library_docs_dir: Path to library_docs directory

    Returns:
        0 if valid, 1 if has errors
    """
    # Import here to avoid circular dependency
    from .sources_config import load_user_sources

    config = load_user_sources(sources_config_path)

    errors = []
    warnings = []

    print("\n=== Validating Sources Configuration ===")
    print(f"Config: {sources_config_path}")
    print(f"Docs:   {library_docs_dir}\n")

    # Validate config schema
    print("--- Config Schema Validation ---\n")

    # Check libraries
    for lib_name, lib_config in config.get("libraries", {}).items():
        # Required fields
        if "type" not in lib_config:
            errors.append(f"libraries.{lib_name}: Missing 'type' field")
        elif lib_config["type"] not in ["sphinx", "doxygen"]:
            errors.append(
                f"libraries.{lib_name}: Invalid type '{lib_config['type']}' "
                f"(expected: sphinx or doxygen)"
            )

        if "doc_root" not in lib_config:
            errors.append(f"libraries.{lib_name}: Missing 'doc_root' field")

        if "version" not in lib_config:
            warnings.append(f"libraries.{lib_name}: Missing 'version' field")

    # Check papers
    for cat_name, cat_config in config.get("papers", {}).items():
        if "type" not in cat_config:
            errors.append(f"papers.{cat_name}: Missing 'type' field")

        if "items" not in cat_config:
            errors.append(f"papers.{cat_name}: Missing 'items' field")
            continue

        items = cat_config.get("items", {})
        if not items:
            warnings.append(f"papers.{cat_name}: Empty category (no papers)")

        # Validate individual papers
        for paper_id, paper_data in items.items():
            for field in ["title", "arxiv_id", "source", "authors", "year"]:
                if field not in paper_data:
                    errors.append(
                        f"papers.{cat_name}.items.{paper_id}: Missing '{field}' field"
                    )

            # Check for deprecated 'url' field
            if "url" in paper_data:
                errors.append(
                    f"papers.{cat_name}.items.{paper_id}: "
                    f"Uses deprecated 'url' field (should be 'source')"
                )

    # Check books
    for book_name, book_config in config.get("books", {}).items():
        if "type" not in book_config:
            errors.append(f"books.{book_name}: Missing 'type' field")

        for field in ["title", "authors", "source", "category", "split_method"]:
            if field not in book_config:
                errors.append(f"books.{book_name}: Missing '{field}' field")

        # Check for deprecated 'url' field
        if "url" in book_config:
            errors.append(
                f"books.{book_name}: Uses deprecated 'url' field (should be 'source')"
            )

        if "split_method" in book_config:
            if book_config["split_method"] not in ["toc", "none", "manual"]:
                errors.append(
                    f"books.{book_name}: Invalid split_method "
                    f"'{book_config['split_method']}' (expected: toc, none, or manual)"
                )

    if errors:
        print(f"❌ Config Errors ({len(errors)}):\n")
        for err in errors:
            print(f"  • {err}")
        print()
    else:
        print("✓ No config schema errors\n")

    if warnings:
        print(f"⚠️  Config Warnings ({len(warnings)}):\n")
        for warn in warnings:
            print(f"  • {warn}")
        print()

    # Validate filesystem
    print("--- Filesystem Validation ---\n")

    missing = []
    found = []

    # Check libraries
    for lib_name in config.get("libraries", {}).keys():
        dir_name = f"library_{lib_name}"
        path = os.path.join(library_docs_dir, dir_name)

        if os.path.exists(path):
            found.append(f"libraries.{lib_name}")
        else:
            missing.append((f"libraries.{lib_name}", dir_name))

    # Check papers
    for cat_name in config.get("papers", {}).keys():
        dir_name = f"papers_{cat_name}"
        path = os.path.join(library_docs_dir, dir_name)

        if os.path.exists(path):
            found.append(f"papers.{cat_name}")
        else:
            missing.append((f"papers.{cat_name}", dir_name))

    # Check books
    for book_name in config.get("books", {}).keys():
        dir_name = f"books_{book_name}"
        path = os.path.join(library_docs_dir, dir_name)

        if os.path.exists(path):
            found.append(f"books.{book_name}")
        else:
            missing.append((f"books.{book_name}", dir_name))

    if found:
        print(f"✓ Found ({len(found)}):\n")
        for item in found:
            print(f"  • {item}")
        print()

    if missing:
        print(f"✗ Missing ({len(missing)}):\n")
        for item, dirname in missing:
            print(f"  • {item} → {dirname}/ not found")
            print(f"    Run: tensor-truth-docs {item.split('.')[1]}")
        print()

    # Check for orphaned directories
    print("--- Orphaned Directories ---\n")
    if os.path.exists(library_docs_dir):
        all_dirs = {
            d
            for d in os.listdir(library_docs_dir)
            if os.path.isdir(os.path.join(library_docs_dir, d))
            and not d.startswith(".")
        }

        # Expected directory names
        config_dirs = set()
        for lib_name in config.get("libraries", {}).keys():
            config_dirs.add(f"library_{lib_name}")
        for cat_name in config.get("papers", {}).keys():
            config_dirs.add(f"papers_{cat_name}")
        for book_name in config.get("books", {}).keys():
            config_dirs.add(f"books_{book_name}")

        orphaned = all_dirs - config_dirs
        if orphaned:
            print(f"⚠️  Orphaned ({len(orphaned)}):\n")
            for dirname in sorted(orphaned):
                print(f"  • {dirname}/ (not in config)")
            print()
        else:
            print("✓ No orphaned directories\n")
    else:
        warnings.append(f"Library docs directory does not exist: {library_docs_dir}")

    # Summary
    print("=" * 60)
    total_sources = (
        len(config.get("libraries", {}))
        + len(config.get("papers", {}))
        + len(config.get("books", {}))
    )

    if errors:
        print("\n❌ VALIDATION FAILED")
        print(f"   {len(errors)} error(s), {len(warnings)} warning(s)")
        print(f"   {len(found)}/{total_sources} sources have docs on disk")
        return 1
    elif missing:
        print("\n⚠️  VALIDATION INCOMPLETE")
        print(f"   {len(missing)} source(s) missing docs on disk")
        print(f"   {len(found)}/{total_sources} sources have docs")
        print("\n   Run tensor-truth-docs to fetch missing sources")
        return 0  # Not an error, just incomplete
    else:
        print("\n✅ VALIDATION PASSED")
        print(f"   All {total_sources} sources configured and fetched")
        return 0
