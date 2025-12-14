"""App utilities for Streamlit interface."""

__all__ = [
    # Commands
    "process_command",
    # Helpers
    "download_indexes_with_ui",
    "ensure_engine_loaded",
    "free_memory",
    "get_available_modules",
    "get_ollama_models",
    "get_system_devices",
    # Logging
    "logger",
    # Presets
    "apply_preset",
    "delete_preset",
    "load_presets",
    "save_preset",
    # Session
    "create_session",
    "load_sessions",
    "rename_session",
    "save_sessions",
    "update_title",
    # Title Generation
    "generate_smart_title",
    # VRAM
    "estimate_vram_usage",
    "get_vram_breakdown",
    "render_vram_gauge",
]


def __getattr__(name):
    """Lazy import implementation to avoid circular dependencies and streamlit requirement."""

    # Commands
    if name == "process_command":
        from .commands import process_command

        return process_command

    # Helpers
    if name in (
        "download_indexes_with_ui",
        "ensure_engine_loaded",
        "free_memory",
        "get_available_modules",
        "get_ollama_models",
        "get_system_devices",
    ):
        from .helpers import (
            download_indexes_with_ui,
            ensure_engine_loaded,
            free_memory,
            get_available_modules,
            get_ollama_models,
            get_system_devices,
        )

        return locals()[name]

    # Logging
    if name == "logger":
        from .logging_config import logger

        return logger

    # Presets
    if name in ("apply_preset", "delete_preset", "load_presets", "save_preset"):
        from .presets import apply_preset, delete_preset, load_presets, save_preset

        return locals()[name]

    # Session
    if name in (
        "create_session",
        "load_sessions",
        "rename_session",
        "save_sessions",
        "update_title",
    ):
        from .session import (
            create_session,
            load_sessions,
            rename_session,
            save_sessions,
            update_title,
        )

        return locals()[name]

    # Title Generation
    if name == "generate_smart_title":
        from .title_generation import generate_smart_title

        return generate_smart_title

    # VRAM
    if name in ("estimate_vram_usage", "get_vram_breakdown", "render_vram_gauge"):
        from .vram import estimate_vram_usage, get_vram_breakdown, render_vram_gauge

        return locals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
