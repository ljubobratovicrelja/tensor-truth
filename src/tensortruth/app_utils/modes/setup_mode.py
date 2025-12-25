"""Setup mode - Session configuration and creation."""

import time

import streamlit as st

from tensortruth.core.ollama import get_ollama_url

from ..config import update_config
from ..dialog_manager import open_no_rag_dialog
from ..helpers import get_available_modules, get_ollama_models, get_system_devices
from ..presets import get_favorites, load_presets, save_preset
from ..presets_ui import render_favorite_preset_cards, render_presets_manager
from ..session import create_session
from ..setup_state import (
    build_params_from_session_state,
    init_setup_defaults_from_config,
)


def render_setup_mode():
    """Render the setup mode UI for creating new sessions."""
    with st.container():
        # Fetch data
        available_mods_tuples = get_available_modules(st.session_state.index_dir)
        module_to_display = {mod: disp for mod, disp in available_mods_tuples}
        display_to_module = {disp: mod for mod, disp in available_mods_tuples}
        available_mods = [mod for mod, _ in available_mods_tuples]

        available_models = get_ollama_models()
        system_devices = get_system_devices()
        presets = load_presets(st.session_state.presets_file)

        # Initialize setup defaults from config on first run
        init_setup_defaults_from_config()

        # Set default model if not already set
        if st.session_state.setup_model is None and available_models:
            default_model_idx = 0
            for i, m in enumerate(available_models):
                if "deepseek-r1:8b" in m:
                    default_model_idx = i
            st.session_state.setup_model = available_models[default_model_idx]

        st.markdown("### Start a New Research Session")

        # Quick launch favorites
        favorites = get_favorites(st.session_state.presets_file)
        if favorites:
            render_favorite_preset_cards(
                favorites,
                available_mods,
                st.session_state.presets_file,
                st.session_state.sessions_file,
            )

        # All presets manager
        if presets:
            render_presets_manager(
                presets,
                available_mods,
                available_models,
                system_devices,
                st.session_state.presets_file,
            )

        # Manual configuration
        expand_config = st.session_state.get("expand_config_section", False)
        with st.expander("Configure New Session", expanded=expand_config):
            with st.form("launch_form"):
                st.subheader("1. Knowledge Base")
                available_display_names = [
                    module_to_display[mod] for mod in available_mods
                ]
                current_display_selection = [
                    module_to_display[mod]
                    for mod in st.session_state.get("setup_mods", [])
                    if mod in module_to_display
                ]

                selected_display_names = st.multiselect(
                    "Active Indices:",
                    available_display_names,
                    default=current_display_selection,
                )

                selected_mods = [
                    display_to_module[disp] for disp in selected_display_names
                ]
                st.session_state.setup_mods = selected_mods

                st.subheader("2. Model Selection")

                model_col, context_win_col, temperature_col = st.columns(3)

                with model_col:
                    if available_models:
                        st.selectbox("LLM:", available_models, key="setup_model")
                    else:
                        st.error("No models found in Ollama.")

                with context_win_col:
                    st.select_slider(
                        "Context Window",
                        options=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
                        key="setup_ctx",
                    )

                with temperature_col:
                    st.slider("Temperature", 0.0, 1.0, step=0.1, key="setup_temp")

                # Max tokens for thinking models
                st.select_slider(
                    "Max Tokens (for thinking models)",
                    options=[1024, 2048, 4096, 8192, 16384],
                    key="setup_max_tokens",
                    help=(
                        "Limits total output (thinking + response) for "
                        "reasoning models to prevent runaway loops",
                    ),
                )

                st.subheader("3. RAG Parameters")

                rerank_col, top_n_col, conf_col = st.columns(3)
                with rerank_col:
                    st.selectbox(
                        "Reranker",
                        options=[
                            "BAAI/bge-reranker-v2-m3",
                            "BAAI/bge-reranker-base",
                            "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        ],
                        key="setup_reranker",
                    )
                with top_n_col:
                    st.number_input(
                        "Top N (Final Context)",
                        min_value=1,
                        max_value=20,
                        key="setup_top_n",
                    )

                with conf_col:
                    st.slider(
                        "Confidence Warning Threshold",
                        0.0,
                        1.0,
                        step=0.05,
                        key="setup_conf",
                        help=(
                            "Show a warning if the best similarity score is below "
                            "this threshold (soft hint, doesn't filter results)"
                        ),
                    )

                st.slider(
                    "Confidence Cutoff (Hard Filter)",
                    0.0,
                    1.0,
                    step=0.05,
                    key="setup_conf_cutoff_hard",
                    help=(
                        "Hard cutoff - all sources with reranked scores below "
                        "this threshold will be filtered out after reranking. "
                        "Should be lower than the warning threshold."
                    ),
                )

                st.text_area(
                    "System Instructions:",
                    height=68,
                    placeholder="Optional...",
                    key="setup_sys_prompt",
                )

                st.markdown("#### Hardware Allocation")
                h1, h2 = st.columns(2)

                with h1:
                    st.selectbox(
                        "Pipeline Device (Embed/Rerank)",
                        options=system_devices,
                        help=(
                            "Run Retrieval on specific hardware. "
                            "CPU saves VRAM but is slower."
                        ),
                        key="setup_rag_device",
                    )
                with h2:
                    st.selectbox(
                        "Model Device (Ollama)",
                        options=["gpu", "cpu"],
                        help="Force Ollama to run on CPU to save VRAM for other tasks.",
                        key="setup_llm_device",
                    )

                st.markdown("---")

                submitted_start = st.form_submit_button(
                    "Start Session", type="primary", use_container_width=True
                )

            if submitted_start:
                # Build params from session_state
                params = build_params_from_session_state()

                if not selected_mods:
                    open_no_rag_dialog()
                    st.session_state.pending_params = params
                    st.rerun()
                else:
                    with st.spinner("Creating session..."):
                        create_session(
                            selected_mods, params, st.session_state.sessions_file
                        )
                        st.session_state.mode = "chat"
                        st.session_state.sidebar_state = "collapsed"
                    st.rerun()

        # Save preset section
        with st.expander("Save Current Configuration as Preset", expanded=False):
            new_preset_name = st.text_input(
                "Preset Name", placeholder="e.g. 'Deep Search 32B'"
            )
            new_preset_description = st.text_input(
                "Description (optional)",
                placeholder="Brief description of this preset...",
            )
            mark_as_favorite = st.checkbox("Mark as Favorite", value=False)

            if st.button("Save Preset", use_container_width=True, type="primary"):
                if new_preset_name:
                    # Build preset config from session_state
                    preset_config = build_params_from_session_state()
                    preset_config["modules"] = st.session_state.setup_mods

                    if new_preset_description:
                        preset_config["description"] = new_preset_description

                    if mark_as_favorite:
                        all_presets = load_presets(st.session_state.presets_file)
                        max_order = -1
                        for preset in all_presets.values():
                            if preset.get("favorite", False):
                                order = preset.get("favorite_order", 0)
                                if order > max_order:
                                    max_order = order
                        preset_config["favorite"] = True
                        preset_config["favorite_order"] = max_order + 1

                    save_preset(
                        new_preset_name, preset_config, st.session_state.presets_file
                    )
                    st.success(f"Saved: {new_preset_name}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("Please enter a preset name")

        # Connection settings
        with st.expander("Connection Settings", expanded=False):
            current_url = get_ollama_url()

            new_url = st.text_input(
                "Ollama Base URL",
                value=current_url,
                help="e.g. http://localhost:11434 or http://192.168.1.50:11434",
            )

            if st.button("Save Connection URL"):
                if new_url != current_url:
                    try:
                        update_config(ollama_base_url=new_url)
                        get_ollama_models.clear()
                        st.success(
                            "Configuration saved! Model list will refresh from new URL."
                        )
                    except Exception as e:
                        st.error(f"Failed to save config: {e}")
                else:
                    st.info("No changes made.")
