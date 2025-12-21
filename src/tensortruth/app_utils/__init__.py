"""App utilities for Streamlit interface."""

from .commands import process_command
from .config import get_config_file_path, load_config, save_config, update_config
from .helpers import (
    download_indexes_with_ui,
    ensure_engine_loaded,
    free_memory,
    get_available_modules,
    get_ollama_models,
    get_ollama_ps,
    get_random_generating_message,
    get_random_rag_processing_message,
    get_system_devices,
)
from .logging_config import logger
from .paths import (
    get_indexes_dir,
    get_presets_file,
    get_sessions_file,
    get_user_data_dir,
)
from .presets import (
    apply_preset,
    delete_preset,
    get_favorites,
    load_presets,
    quick_launch_preset,
    save_preset,
    toggle_favorite,
)
from .session import (
    create_session,
    load_sessions,
    rename_session,
    save_sessions,
    update_title,
)
from .setup_state import (
    build_params_from_session_state,
    get_session_params_with_defaults,
    init_setup_defaults_from_config,
)
from .title_generation import generate_smart_title
from .vram import estimate_vram_usage, get_vram_breakdown, render_vram_gauge

__all__ = [
    # Commands
    "process_command",
    # Helpers
    "download_indexes_with_ui",
    "ensure_engine_loaded",
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
    "apply_preset",
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
