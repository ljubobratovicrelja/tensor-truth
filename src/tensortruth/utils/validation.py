"""Validation utilities for tensor-truth CLI commands."""

import logging
import os
import re

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
        0 if valid or incomplete (warnings only), 1 if has errors
    """
    # Import here to avoid circular dependency
    from .sources_config import load_user_sources

    config = load_user_sources(sources_config_path)

    errors = []
    warnings = []

    logger.info("=" * 60)
    logger.info("Validating Sources Configuration")
    logger.info(f"Config: {sources_config_path}")
    logger.info(f"Docs:   {library_docs_dir}")
    logger.info("=" * 60)

    # Validate config schema
    logger.info("\n--- Config Schema Validation ---")

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
        logger.error(f"Config Errors ({len(errors)}):")
        for err in errors:
            logger.error(f"  • {err}")
    else:
        logger.info("✓ No config schema errors")

    if warnings:
        logger.warning(f"Config Warnings ({len(warnings)}):")
        for warn in warnings:
            logger.warning(f"  • {warn}")

    # Validate filesystem
    logger.info("\n--- Filesystem Validation ---")

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
        logger.info(f"✓ Found ({len(found)}):")
        for item in found:
            logger.info(f"  • {item}")

    if missing:
        logger.warning(f"✗ Missing ({len(missing)}):")
        for item, dirname in missing:
            logger.warning(f"  • {item} → {dirname}/ not found")
            logger.warning(f"    Run: tensor-truth-docs {item.split('.')[1]}")

    # Check for orphaned directories
    logger.info("\n--- Orphaned Directories ---")
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
            logger.warning(f"⚠️  Orphaned ({len(orphaned)}):")
            for dirname in sorted(orphaned):
                logger.warning(f"  • {dirname}/ (not in config)")
        else:
            logger.info("✓ No orphaned directories")
    else:
        warnings.append(f"Library docs directory does not exist: {library_docs_dir}")

    # Summary
    logger.info("=" * 60)
    total_sources = (
        len(config.get("libraries", {}))
        + len(config.get("papers", {}))
        + len(config.get("books", {}))
    )

    if errors:
        logger.error("❌ VALIDATION FAILED")
        logger.error(f"   {len(errors)} error(s), {len(warnings)} warning(s)")
        logger.error(f"   {len(found)}/{total_sources} sources have docs on disk")
        return 1
    elif missing:
        logger.warning("⚠️  VALIDATION INCOMPLETE")
        logger.warning(f"   {len(missing)} source(s) missing docs on disk")
        logger.warning(f"   {len(found)}/{total_sources} sources have docs")
        logger.info("   Run tensor-truth-docs to fetch missing sources")
        return 0  # Not an error, just incomplete
    else:
        logger.info("✅ VALIDATION PASSED")
        logger.info(f"   All {total_sources} sources configured and fetched")
        return 0


def validate_url(url: str) -> bool:
    """Validate URL format and accessibility.

    Args:
        url: URL to validate

    Returns:
        True if URL is valid and accessible, False otherwise
    """
    # Basic regex check
    url_pattern = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    if not url_pattern.match(url):
        return False

    # Try HEAD request to check accessibility
    try:
        import requests

        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code < 400
    except Exception:
        # If HEAD fails, try GET
        try:
            import requests

            response = requests.get(url, timeout=10, allow_redirects=True)
            return response.status_code < 400
        except Exception:
            return False


def prompt_for_url(prompt_message: str, examples: list[str] | None = None) -> str:
    """Prompt user for URL with validation and retry loop.

    Args:
        prompt_message: Message to display when prompting for URL
        examples: Optional list of example URLs to display

    Returns:
        Valid URL string

    Raises:
        SystemExit: If user cancels (exits with code 1)
    """
    print(prompt_message)
    if examples:
        print("Examples:")
        for example in examples:
            print(f"  - {example}")

    url = input("\nURL: ").strip()

    # Retry loop for URL validation
    while not validate_url(url):
        logger.error(f"Invalid or inaccessible URL: {url}")
        url = input("\nTry again (or press Enter to cancel): ").strip()
        if not url:
            print("Cancelled.")
            raise SystemExit(1)

    return url


def sanitize_config_key(name: str) -> str:
    """Sanitize name to valid sources.json key.

    Args:
        name: Name to sanitize

    Returns:
        Sanitized name (lowercase, alphanumeric + underscore)
    """
    # Convert to lowercase
    name = name.lower()
    # Replace non-alphanumeric with underscore
    name = re.sub(r"[^a-z0-9_-]", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name


def validate_arxiv_id(arxiv_id: str):
    """Validate and normalize ArXiv ID.

    Supports formats:
    - 1234.5678 (new format)
    - arch-ive/1234567 (old format)
    - https://arxiv.org/abs/1234.5678 (URL)

    Args:
        arxiv_id: ArXiv ID to validate

    Returns:
        Normalized ID or None if invalid
    """
    arxiv_id = arxiv_id.strip()

    # Extract from URL
    if "arxiv.org" in arxiv_id:
        match = re.search(r"(\d{4}\.\d{4,5})", arxiv_id)
        if match:
            return match.group(1)
        match = re.search(r"([a-z\-]+/\d{7})", arxiv_id)
        if match:
            return match.group(1)
        return None

    # Validate format
    # New format: YYMM.NNNNN
    if re.match(r"^\d{4}\.\d{4,5}$", arxiv_id):
        return arxiv_id

    # Old format: arch-ive/YYMMNNN
    if re.match(r"^[a-z\-]+/\d{7}$", arxiv_id):
        return arxiv_id

    logger.warning(f"Invalid ArXiv ID format: {arxiv_id}")
    return None
