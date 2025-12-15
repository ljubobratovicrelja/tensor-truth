"""App utilities for Streamlit interface."""

from .commands import process_command
from .helpers import (
    download_indexes_with_ui,
    ensure_engine_loaded,
    free_memory,
    get_available_modules,
    get_ollama_models,
    get_random_rag_processing_message,
    get_random_thinking_message,
    get_system_devices,
)
from .logging_config import logger
from .presets import apply_preset, delete_preset, load_presets, save_preset
from .session import (
    create_session,
    load_sessions,
    rename_session,
    save_sessions,
    update_title,
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
    "get_random_thinking_message",
    "get_random_rag_processing_message",
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
