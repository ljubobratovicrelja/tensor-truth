"""Preset configuration management."""

import json
import os


def load_presets(presets_file: str):
    """Load presets from JSON file.

    If the file doesn't exist, generates it from defaults.
    """
    # Try to ensure presets exist (generates from defaults if missing)
    try:
        from tensortruth.preset_defaults import ensure_presets_exist

        ensure_presets_exist(presets_file)
    except Exception:
        pass  # Continue even if generation fails

    if os.path.exists(presets_file):
        try:
            with open(presets_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_preset(name, config, presets_file: str):
    """Save a preset configuration."""
    presets = load_presets(presets_file)
    presets[name] = config
    with open(presets_file, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)


def delete_preset(name, presets_file: str):
    """Delete a preset configuration."""
    presets = load_presets(presets_file)
    if name in presets:
        del presets[name]
        with open(presets_file, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)


def apply_preset(
    name, available_mods, available_models, available_devices, presets_file: str
):
    """Apply a preset configuration to session state.

    Gracefully handles missing models by attempting to resolve a suitable alternative.
    """
    import streamlit as st

    presets = load_presets(presets_file)
    if name not in presets:
        return

    p = presets[name]

    # Update Session State Keys directly - only if present in preset

    # 1. Modules
    if "modules" in p:
        valid_mods = [m for m in p["modules"] if m in available_mods]
        st.session_state.setup_mods = valid_mods

    # 2. Model - with fallback to model preference resolution
    if "model" in p:
        if p["model"] in available_models:
            st.session_state.setup_model = p["model"]
        else:
            # Model not available - try to resolve from preference if it exists
            try:
                from tensortruth.preset_defaults import (
                    get_default_presets,
                    resolve_model_for_preset,
                )

                defaults = get_default_presets()
                if name in defaults and "model_preference" in defaults[name]:
                    fallback = resolve_model_for_preset(
                        defaults[name], available_models
                    )
                    if fallback:
                        st.session_state.setup_model = fallback
                        st.warning(
                            f"Model '{p['model']}' not available. Using '{fallback}' instead."
                        )
            except Exception:
                pass  # Keep existing model if resolution fails

    # 3. Parameters - only update if present in preset
    if "reranker_model" in p:
        st.session_state.setup_reranker = p["reranker_model"]
    if "context_window" in p:
        st.session_state.setup_ctx = p["context_window"]
    if "temperature" in p:
        st.session_state.setup_temp = p["temperature"]
    if "reranker_top_n" in p:
        st.session_state.setup_top_n = p["reranker_top_n"]
    if "confidence_cutoff" in p:
        st.session_state.setup_conf = p["confidence_cutoff"]
    if "system_prompt" in p:
        st.session_state.setup_sys_prompt = p["system_prompt"]

    # 4. Devices - only update if present in preset and valid
    if "rag_device" in p and p["rag_device"] in available_devices:
        st.session_state.setup_rag_device = p["rag_device"]

    if "llm_device" in p and p["llm_device"] in ["cpu", "gpu"]:
        st.session_state.setup_llm_device = p["llm_device"]
