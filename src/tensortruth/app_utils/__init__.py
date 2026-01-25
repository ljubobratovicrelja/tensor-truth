"""App utilities for Streamlit interface.

NOTE: This __init__.py uses lazy imports to avoid loading Streamlit
when only non-UI modules (config, paths, logging) are needed.
This allows the API server to import from app_utils without Streamlit warnings.
"""

# These modules are Streamlit-free and safe to import eagerly
from .config import get_config_file_path, load_config, save_config, update_config
from .logging_config import logger
from .paths import (
    get_indexes_dir,
    get_presets_file,
    get_sessions_file,
    get_user_data_dir,
)


def __getattr__(name: str):
    """Lazy import for Streamlit-dependent modules."""
    # App State
    if name == "init_app_state":
        from .app_state import init_app_state

        return init_app_state

    # Commands
    if name == "process_command":
        from .commands import process_command

        return process_command

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
    if name == "quick_launch_preset":
        from .presets import quick_launch_preset

        return quick_launch_preset
    if name == "save_preset":
        from .presets import save_preset

        return save_preset
    if name == "toggle_favorite":
        from .presets import toggle_favorite

        return toggle_favorite

    # Session
    if name == "create_session":
        from .session import create_session

        return create_session
    if name == "load_sessions":
        from .session import load_sessions

        return load_sessions
    if name == "rename_session":
        from .session import rename_session

        return rename_session
    if name == "save_sessions":
        from .session import save_sessions

        return save_sessions
    if name == "update_title":
        from .session import update_title

        return update_title

    # Setup State
    if name == "build_params_from_session_state":
        from .setup_state import build_params_from_session_state

        return build_params_from_session_state
    if name == "get_session_params_with_defaults":
        from .setup_state import get_session_params_with_defaults

        return get_session_params_with_defaults
    if name == "init_setup_defaults_from_config":
        from .setup_state import init_setup_defaults_from_config

        return init_setup_defaults_from_config

    # Title Generation
    if name == "generate_smart_title":
        from .title_generation import generate_smart_title

        return generate_smart_title

    # VRAM
    if name == "estimate_vram_usage":
        from .vram import estimate_vram_usage

        return estimate_vram_usage
    if name == "get_vram_breakdown":
        from .vram import get_vram_breakdown

        return get_vram_breakdown
    if name == "render_vram_gauge":
        from .vram import render_vram_gauge

        return render_vram_gauge

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # App State
    "init_app_state",
    # Commands
    "process_command",
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
    "quick_launch_preset",
    "save_preset",
    "toggle_favorite",
    # Config
    "get_config_file_path",
    "load_config",
    "save_config",
    "update_config",
    # Session
    "create_session",
    "load_sessions",
    "rename_session",
    "save_sessions",
    "update_title",
    # Setup State
    "build_params_from_session_state",
    "get_session_params_with_defaults",
    "init_setup_defaults_from_config",
    # Title Generation
    "generate_smart_title",
    # VRAM
    "estimate_vram_usage",
    "get_vram_breakdown",
    "render_vram_gauge",
]
