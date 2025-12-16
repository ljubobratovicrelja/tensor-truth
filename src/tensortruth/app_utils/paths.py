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


def get_sessions_file() -> str:
    """Get the path to the chat sessions file."""
    return str(get_user_data_dir() / "chat_sessions.json")


def get_presets_file() -> str:
    """Get the path to the presets file."""
    return str(get_user_data_dir() / "presets.json")


def get_indexes_dir() -> str:
    """Get the path to the indexes directory."""
    indexes_dir = get_user_data_dir() / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    return str(indexes_dir)
