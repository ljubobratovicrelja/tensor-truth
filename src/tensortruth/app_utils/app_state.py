"""Centralized application state initialization and management.

This module handles one-time initialization of app constants and state,
storing them in st.session_state to avoid recomputation and simplify
function signatures across the application.
"""

import logging
import os
from pathlib import Path

import streamlit as st

from tensortruth import get_max_memory_gb

from .config import load_config
from .paths import (
    get_indexes_dir,
    get_presets_file,
    get_sessions_file,
    get_user_data_dir,
)
from .session import load_sessions

logger = logging.getLogger(__name__)

# Module-level flag to track if models have been loaded in this process.
# This survives Streamlit reruns but resets on browser refresh (new process).
_MODELS_LOADED = False


def mark_models_loaded():
    """Mark that GPU models have been loaded in this process.

    Call this after successfully loading the RAG engine to enable
    browser refresh detection.
    """
    global _MODELS_LOADED
    _MODELS_LOADED = True


def init_app_state():
    """Initialize application state once at startup.

    Stores all constants and configuration in st.session_state to:
    - Avoid recomputation on every rerun
    - Simplify function signatures (no need to pass paths everywhere)
    - Provide single source of truth for app-wide constants
    - Cache config file to avoid repeated reads

    Also detects browser refresh and cleans up GPU memory from previous session.
    """
    global _MODELS_LOADED

    # Detect browser refresh: models were loaded but session_state is fresh.
    # This happens when user refreshes the page - Streamlit session resets
    # but the Python process (with loaded GPU models) continues running.
    if _MODELS_LOADED and "app_initialized" not in st.session_state:
        logger.info("Browser refresh detected - cleaning up GPU memory")
        from tensortruth.app_utils.helpers import free_memory

        free_memory()
        _MODELS_LOADED = False

    if "app_initialized" in st.session_state:
        return  # Already initialized

    # File paths (stored as Path objects for consistency)
    st.session_state.sessions_file = get_sessions_file()
    st.session_state.presets_file = get_presets_file()
    st.session_state.user_dir = get_user_data_dir()
    st.session_state.index_dir = get_indexes_dir()

    # Load and cache config (avoids re-reading file multiple times)
    st.session_state.config = load_config()

    # Constants
    st.session_state.max_vram_gb = get_max_memory_gb()

    # Media paths
    app_root = Path(__file__).parent.parent
    st.session_state.icon_path = app_root / "media" / "tensor_truth_icon_256.png"
    st.session_state.logo_path = app_root / "media" / "tensor_truth_banner.png"
    st.session_state.css_path = app_root / "media" / "app_styles.css"

    # Load CSS once
    with open(st.session_state.css_path) as f:
        st.session_state.css_data = f"<style>{f.read()}</style>"

    # Load initial data
    if "chat_data" not in st.session_state:
        st.session_state.chat_data = load_sessions(st.session_state.sessions_file)

    # Initialize mode
    if "mode" not in st.session_state:
        st.session_state.mode = "setup"

    # Initialize engine state
    if "loaded_config" not in st.session_state:
        st.session_state.loaded_config = None
    if "engine" not in st.session_state:
        st.session_state.engine = None

    # Initialize debug flags from environment
    st.session_state.debug_context = os.environ.get("TENSOR_TRUTH_DEBUG_CONTEXT") == "1"

    # Mark as initialized
    st.session_state.app_initialized = True
