"""Cross-platform path management for TensorTruth user data."""

from pathlib import Path


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
