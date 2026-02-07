"""Path utilities for CLI commands.

This module provides path resolution functions for CLI commands without
importing app_utils dependencies.
"""

import os
from pathlib import Path
from typing import Optional


def get_user_data_dir() -> Path:
    """
    Get the platform-specific user data directory for TensorTruth.

    Returns:
        Path to ~/.tensortruth on all platforms (Windows, macOS, Linux)

    Examples:
        - macOS/Linux: /Users/username/.tensortruth
        - Windows: C:\\Users\\username\\.tensortruth
    """
    home = Path.home()
    data_dir = home / ".tensortruth"

    # Create the directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


def get_library_docs_dir(override: Optional[str] = None) -> str:
    """
    Get library documentation directory with override support.

    Priority order:
    1. override argument (from CLI --library-docs-dir)
    2. TENSOR_TRUTH_DOCS_DIR environment variable
    3. ~/.tensortruth/library_docs (default)

    Args:
        override: Optional path override from CLI argument

    Returns:
        Absolute path to library_docs directory (created if doesn't exist)

    Environment Variables:
        TENSOR_TRUTH_DOCS_DIR: Override default library docs directory

    Examples:
        >>> get_library_docs_dir()
        '/home/user/.tensortruth/library_docs'

        >>> get_library_docs_dir('/custom/path')
        '/custom/path'

        >>> os.environ['TENSOR_TRUTH_DOCS_DIR'] = '/env/path'
        >>> get_library_docs_dir()
        '/env/path'
    """
    if override:
        path = Path(override).expanduser().resolve()
    elif env_path := os.getenv("TENSOR_TRUTH_DOCS_DIR"):
        path = Path(env_path).expanduser().resolve()
    else:
        path = get_user_data_dir() / "library_docs"

    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_sources_config_path(override: Optional[str] = None) -> str:
    """
    Get sources.json configuration file path with override support.

    Priority order:
    1. override argument (from CLI --sources-config)
    2. TENSOR_TRUTH_SOURCES_CONFIG environment variable
    3. ~/.tensortruth/sources.json (default)

    Args:
        override: Optional path override from CLI argument

    Returns:
        Absolute path to sources.json file (NOT auto-created)

    Environment Variables:
        TENSOR_TRUTH_SOURCES_CONFIG: Override default sources config path

    Examples:
        >>> get_sources_config_path()
        '/home/user/.tensortruth/sources.json'

        >>> get_sources_config_path('/custom/sources.json')
        '/custom/sources.json'
    """
    if override:
        return str(Path(override).expanduser().resolve())
    elif env_path := os.getenv("TENSOR_TRUTH_SOURCES_CONFIG"):
        return str(Path(env_path).expanduser().resolve())
    else:
        return str(get_user_data_dir() / "sources.json")


def get_base_indexes_dir(override: Optional[str] = None) -> str:
    """
    Get vector indexes directory with override support.

    Priority order:
    1. override argument (from CLI --indexes-dir)
    2. TENSOR_TRUTH_INDEXES_DIR environment variable
    3. ~/.tensortruth/indexes (default)

    Args:
        override: Optional path override from CLI argument

    Returns:
        Absolute path to indexes directory (created if doesn't exist)

    Environment Variables:
        TENSOR_TRUTH_INDEXES_DIR: Override default indexes directory

    Examples:
        >>> get_base_indexes_dir()
        '/home/user/.tensortruth/indexes'

        >>> get_base_indexes_dir('/data/indexes')
        '/data/indexes'
    """
    if override:
        path = Path(override).expanduser().resolve()
    elif env_path := os.getenv("TENSOR_TRUTH_INDEXES_DIR"):
        path = Path(env_path).expanduser().resolve()
    else:
        path = get_user_data_dir() / "indexes"

    path.mkdir(parents=True, exist_ok=True)
    return str(path)
