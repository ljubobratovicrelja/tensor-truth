"""Preset configuration management."""

import json
import os

# UI constraints - these define the valid values for preset parameters
ALLOWED_CONTEXT_WINDOWS = [2048, 4096, 8192, 16384, 32768]
ALLOWED_RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]


def _normalize_preset(preset: dict) -> dict:
    """Normalize preset values to match UI constraints.

    Ensures manually edited presets don't break the UI.
    Modifies the preset in-place.

    Args:
        preset: Preset configuration dictionary

    Returns:
        Normalized preset dictionary (same reference)
    """
    # Normalize context_window to nearest allowed value
    if "context_window" in preset:
        ctx = preset["context_window"]
        if ctx not in ALLOWED_CONTEXT_WINDOWS:
            nearest = min(ALLOWED_CONTEXT_WINDOWS, key=lambda x: abs(x - ctx))
            preset["context_window"] = nearest

    # Clamp temperature to [0.0, 1.0]
    if "temperature" in preset:
        temp = preset["temperature"]
        preset["temperature"] = max(0.0, min(1.0, temp))

    # Ensure reranker_top_n is within reasonable bounds [1, 20]
    if "reranker_top_n" in preset:
        top_n = preset["reranker_top_n"]
        preset["reranker_top_n"] = max(1, min(20, int(top_n)))

    # Clamp confidence_cutoff to [0.0, 1.0]
    if "confidence_cutoff" in preset:
        conf = preset["confidence_cutoff"]
        preset["confidence_cutoff"] = max(0.0, min(1.0, conf))

    # Validate reranker_model (fallback to default if invalid)
    if "reranker_model" in preset:
        if preset["reranker_model"] not in ALLOWED_RERANKER_MODELS:
            preset["reranker_model"] = ALLOWED_RERANKER_MODELS[0]

    # Validate llm_device (must be 'cpu' or 'gpu')
    if "llm_device" in preset:
        if preset["llm_device"] not in ["cpu", "gpu"]:
            preset["llm_device"] = "gpu"

    # Ensure system_prompt is a string (handle list/array edge cases)
    if "system_prompt" in preset:
        prompt = preset["system_prompt"]
        if isinstance(prompt, list):
            preset["system_prompt"] = " ".join(str(p) for p in prompt)
        elif not isinstance(prompt, str):
            preset["system_prompt"] = str(prompt)

    return preset


def load_presets(presets_file: str):
    """Load presets from JSON file.

    Generates from defaults if file doesn't exist.
    Automatically normalizes all presets to match UI constraints.

    Args:
        presets_file: Path to presets JSON file

    Returns:
        Dictionary of preset configurations
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
                presets = json.load(f)
                # Normalize all presets on load
                for name, preset in presets.items():
                    _normalize_preset(preset)
                return presets
        except Exception:
            pass
    return {}


def save_preset(name, config, presets_file: str):
    """Save a preset configuration.

    Normalizes the config before saving to ensure consistency.

    Args:
        name: Preset name
        config: Configuration dictionary
        presets_file: Path to presets JSON file
    """
    presets = load_presets(presets_file)
    # Normalize the config before saving
    presets[name] = _normalize_preset(config.copy())
    with open(presets_file, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)


def delete_preset(name, presets_file: str):
    """Delete a preset configuration."""
    presets = load_presets(presets_file)
    if name in presets:
        del presets[name]
        with open(presets_file, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)


def toggle_favorite(name, presets_file: str):
    """Toggle favorite status for a preset."""
    presets = load_presets(presets_file)
    if name in presets:
        current_status = presets[name].get("favorite", False)
        presets[name]["favorite"] = not current_status

        # Set favorite_order if becoming a favorite
        if not current_status:
            # Find the highest favorite_order and add 1
            max_order = -1
            for preset in presets.values():
                if preset.get("favorite", False):
                    order = preset.get("favorite_order", 0)
                    if order > max_order:
                        max_order = order
            presets[name]["favorite_order"] = max_order + 1

        with open(presets_file, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)


def get_favorites(presets_file: str):
    """Get all favorite presets sorted by favorite_order."""
    presets = load_presets(presets_file)
    favorites = {
        name: config
        for name, config in presets.items()
        if config.get("favorite", False)
    }
    # Sort by favorite_order
    sorted_favorites = sorted(
        favorites.items(), key=lambda x: x[1].get("favorite_order", 999)
    )
    return dict(sorted_favorites)


def quick_launch_preset(name, available_mods, presets_file: str, sessions_file: str):
    """Quick launch a session directly from a preset.

    Args:
        name: Preset name to launch
        available_mods: List of available module names
        presets_file: Path to presets file
        sessions_file: Path to sessions file

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    from tensortruth.app_utils.session import create_session

    presets = load_presets(presets_file)
    if name not in presets:
        return False, f"Preset '{name}' not found"

    preset = presets[name]

    # Validate modules
    modules = preset.get("modules", [])
    valid_mods = [m for m in modules if m in available_mods]

    if not valid_mods and modules:
        return False, "None of the preset modules are available"

    # Build params from preset
    params = {
        "model": preset.get("model", "deepseek-r1:8b"),
        "temperature": preset.get("temperature", 0.3),
        "context_window": preset.get("context_window", 4096),
        "system_prompt": preset.get("system_prompt", ""),
        "reranker_model": preset.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
        "reranker_top_n": preset.get("reranker_top_n", 3),
        "confidence_cutoff": preset.get("confidence_cutoff", 0.3),
        "rag_device": preset.get("rag_device", "cpu"),
        "llm_device": preset.get("llm_device", "gpu"),
    }

    # Create session
    create_session(valid_mods, params, sessions_file)
    return True, None


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

    # 3. Parameters - already normalized by load_presets()
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
