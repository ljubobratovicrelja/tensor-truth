"""Preset configuration management."""

import json
import os

import streamlit as st


def load_presets(presets_file: str):
    """Load presets from JSON file."""
    if os.path.exists(presets_file):
        try:
            with open(presets_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
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


def apply_preset(name, available_mods, available_models, available_devices, presets_file: str):
    """Apply a preset configuration to session state."""
    presets = load_presets(presets_file)
    if name not in presets:
        return

    p = presets[name]

    # Update Session State Keys directly - only if present in preset

    # 1. Modules
    if "modules" in p:
        valid_mods = [m for m in p["modules"] if m in available_mods]
        st.session_state.setup_mods = valid_mods

    # 2. Model
    if "model" in p and p["model"] in available_models:
        st.session_state.setup_model = p["model"]

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
