"""Centralized application state initialization and management.

This module handles one-time initialization of app constants and state,
storing them in st.session_state to avoid recomputation and simplify
function signatures across the application.
"""

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


def init_app_state():
    """Initialize application state once at startup.

    Stores all constants and configuration in st.session_state to:
    - Avoid recomputation on every rerun
    - Simplify function signatures (no need to pass paths everywhere)
    - Provide single source of truth for app-wide constants
    - Cache config file to avoid repeated reads
    """
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

    # Mark as initialized
    st.session_state.app_initialized = True
