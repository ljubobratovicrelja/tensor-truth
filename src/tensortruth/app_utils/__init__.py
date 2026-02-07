"""App utilities for TensorTruth."""

# These modules are safe to import eagerly
from .config import get_config_file_path, load_config, save_config, update_config
from .logging_config import logger
from .paths import (
    get_indexes_dir,
    get_presets_file,
    get_sessions_file,
    get_user_data_dir,
)


def __getattr__(name: str):
    """Lazy import for modules with heavy dependencies."""
    # Helpers
    if name == "free_memory":
        from .helpers import free_memory

        return free_memory
    if name == "get_available_modules":
        from .helpers import get_available_modules

        return get_available_modules
    if name == "get_ollama_models":
        from .helpers import get_ollama_models

        return get_ollama_models
    if name == "get_ollama_ps":
        from .helpers import get_ollama_ps

        return get_ollama_ps
    if name == "get_random_generating_message":
        from .helpers import get_random_generating_message

        return get_random_generating_message
    if name == "get_random_rag_processing_message":
        from .helpers import get_random_rag_processing_message

        return get_random_rag_processing_message
    if name == "get_system_devices":
        from .helpers import get_system_devices

        return get_system_devices

    # Presets
    if name == "delete_preset":
        from .presets import delete_preset

        return delete_preset
    if name == "get_favorites":
        from .presets import get_favorites

        return get_favorites
    if name == "load_presets":
        from .presets import load_presets

        return load_presets
    if name == "save_preset":
        from .presets import save_preset

        return save_preset
    if name == "toggle_favorite":
        from .presets import toggle_favorite

        return toggle_favorite

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Helpers
    "free_memory",
    "get_available_modules",
    "get_ollama_models",
    "get_ollama_ps",
    "get_random_generating_message",
    "get_random_rag_processing_message",
    "get_system_devices",
    # Logging
    "logger",
    # Paths
    "get_indexes_dir",
    "get_presets_file",
    "get_sessions_file",
    "get_user_data_dir",
    # Presets
    "delete_preset",
    "get_favorites",
    "load_presets",
    "save_preset",
    "toggle_favorite",
    # Config
    "get_config_file_path",
    "load_config",
    "save_config",
    "update_config",
]
