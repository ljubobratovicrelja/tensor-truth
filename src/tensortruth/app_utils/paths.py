"""Cross-platform path management for TensorTruth user data."""

import os
from pathlib import Path
from typing import Optional

from tensortruth.indexing.metadata import sanitize_model_id


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


def get_sessions_file() -> Path:
    """Get the path to the chat sessions file."""
    return get_user_data_dir() / "chat_sessions.json"


def get_presets_file() -> Path:
    """Get the path to the presets file."""
    return get_user_data_dir() / "presets.json"


def get_indexes_dir() -> Path:
    """Get the path to the indexes directory."""
    indexes_dir = get_user_data_dir() / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    return indexes_dir


def get_sessions_data_dir() -> Path:
    """Get the path to the sessions data directory (~/.tensortruth/sessions/)."""
    sessions_dir = get_user_data_dir() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def get_session_dir(session_id: str) -> Path:
    """Get the path to a specific session's directory."""
    session_dir = get_sessions_data_dir() / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_sessions_index_file() -> Path:
    """Get the path to the sessions index file.

    The index file contains lightweight metadata for fast session listing:
    {current_id, sessions: {id -> {title, created_at}}}
    """
    return get_sessions_data_dir() / "sessions_index.json"


def get_session_data_file(session_id: str) -> Path:
    """Get the path to a session's data file.

    The session data file contains full session data:
    {title, created_at, messages, modules, params, title_needs_update}

    Note: Does not create the session directory - call get_session_dir() first
    if directory creation is needed.
    """
    return get_sessions_data_dir() / session_id / "session.json"


def get_session_pdfs_dir(session_id: str) -> Path:
    """Get the path to a session's PDF storage directory."""
    pdfs_dir = get_session_dir(session_id) / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    return pdfs_dir


def get_session_markdown_dir(session_id: str) -> Path:
    """Get the path to a session's markdown storage directory."""
    markdown_dir = get_session_dir(session_id) / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    return markdown_dir


def get_session_index_dir(session_id: str) -> Path:
    """Get the path to a session's vector index directory."""
    index_dir = get_session_dir(session_id) / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


# Enhanced path utilities with ENV var and CLI override support


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


def get_indexes_dir_for_model(
    embedding_model: str, base_dir_override: Optional[str] = None
) -> Path:
    """Get the indexes directory for a specific embedding model.

    Args:
        embedding_model: HuggingFace model name (e.g., "BAAI/bge-m3")
        base_dir_override: Optional base indexes directory override

    Returns:
        Path to indexes/{model_id}/ directory (created if doesn't exist)

    Examples:
        >>> get_indexes_dir_for_model("BAAI/bge-m3")
        Path('/home/user/.tensortruth/indexes/bge-m3')

        >>> get_indexes_dir_for_model("Qwen/Qwen3-Embedding-0.6B")
        Path('/home/user/.tensortruth/indexes/qwen3-embedding-0.6b')
    """
    base_dir = Path(get_base_indexes_dir(base_dir_override))
    model_id = sanitize_model_id(embedding_model)
    model_dir = base_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_module_index_dir(
    module_name: str, embedding_model: str, base_dir_override: Optional[str] = None
) -> Path:
    """Get the index directory for a specific module and embedding model.

    Args:
        module_name: Module name (e.g., "library_pytorch_2.9")
        embedding_model: HuggingFace model name (e.g., "BAAI/bge-m3")
        base_dir_override: Optional base indexes directory override

    Returns:
        Path to indexes/{model_id}/{module_name}/ directory

    Examples:
        >>> get_module_index_dir("library_pytorch_2.9", "BAAI/bge-m3")
        Path('/home/user/.tensortruth/indexes/bge-m3/library_pytorch_2.9')
    """
    model_dir = get_indexes_dir_for_model(embedding_model, base_dir_override)
    return model_dir / module_name


def get_active_embedding_model() -> str:
    """Get the currently configured default embedding model from config.

    Returns:
        The default embedding model name from config, or fallback default

    Note:
        This function loads config lazily to avoid circular imports.
    """
    try:
        from tensortruth.app_utils.config import load_config

        config = load_config()
        return config.rag.default_embedding_model
    except Exception:
        return "BAAI/bge-m3"


def get_active_embedding_indexes_dir(base_dir_override: Optional[str] = None) -> Path:
    """Get the indexes directory for the currently active embedding model.

    This is a convenience function that combines get_active_embedding_model()
    and get_indexes_dir_for_model().

    Args:
        base_dir_override: Optional base indexes directory override

    Returns:
        Path to indexes/{active_model_id}/ directory
    """
    return get_indexes_dir_for_model(get_active_embedding_model(), base_dir_override)
