"""Preset configuration management."""

import json
from pathlib import Path
from typing import Union

# UI constraints - these define the valid values for preset parameters
ALLOWED_CONTEXT_WINDOWS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
ALLOWED_MAX_TOKENS = [1024, 2048, 4096, 8192, 16384]
ALLOWED_RERANKER_MODELS = [
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
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

    # Normalize max_tokens to nearest allowed value
    if "max_tokens" in preset:
        max_tok = preset["max_tokens"]
        if max_tok not in ALLOWED_MAX_TOKENS:
            nearest = min(ALLOWED_MAX_TOKENS, key=lambda x: abs(x - max_tok))
            preset["max_tokens"] = nearest

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


def load_presets(presets_file: Union[str, Path]):
    """Load presets from JSON file.

    Generates from defaults if file doesn't exist.
    Automatically normalizes all presets to match UI constraints.

    Args:
        presets_file: Path to presets JSON file (str or Path)

    Returns:
        Dictionary of preset configurations
    """
    presets_file = Path(presets_file)
    # Try to ensure presets exist (generates from defaults if missing)
    try:
        from tensortruth.preset_defaults import ensure_presets_exist

        ensure_presets_exist(presets_file)
    except Exception:
        pass  # Continue even if generation fails

    if presets_file.exists():
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


def save_preset(name, config, presets_file: Union[str, Path]):
    """Save a preset configuration.

    Normalizes the config before saving to ensure consistency.

    Args:
        name: Preset name
        config: Configuration dictionary
        presets_file: Path to presets JSON file (str or Path)
    """
    presets_file = Path(presets_file)
    presets = load_presets(presets_file)
    # Normalize the config before saving
    presets[name] = _normalize_preset(config.copy())
    with open(presets_file, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)


def delete_preset(name, presets_file: Union[str, Path]):
    """Delete a preset configuration.

    Args:
        name: Preset name to delete
        presets_file: Path to presets JSON file (str or Path)
    """
    presets_file = Path(presets_file)
    presets = load_presets(presets_file)
    if name in presets:
        del presets[name]
        with open(presets_file, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)


def toggle_favorite(name, presets_file: Union[str, Path]):
    """Toggle favorite status for a preset.

    Args:
        name: Preset name
        presets_file: Path to presets JSON file (str or Path)
    """
    presets_file = Path(presets_file)
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


def get_favorites(presets_file: Union[str, Path]):
    """Get all favorite presets sorted by favorite_order.

    Args:
        presets_file: Path to presets JSON file (str or Path)

    Returns:
        Dictionary of favorite presets sorted by order
    """
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


def build_preset_config(
    name,
    available_mods,
    available_models,
    available_devices,
    presets_file: Union[str, Path],
):
    """Build preset configuration with validation and fallbacks.

    Returns a dict of validated preset values that can be applied to session state.
    Gracefully handles missing models by attempting to resolve a suitable alternative.

    Args:
        name: Preset name
        available_mods: List of available module names
        available_models: List of available model names
        available_devices: List of available devices
        presets_file: Path to presets file (str or Path)

    Returns:
        dict: Configuration dict with keys like 'setup_mods', 'setup_model', etc.
              Returns None if preset doesn't exist.
              Also includes a 'warnings' key with list of warning messages.
    """
    presets = load_presets(presets_file)
    if name not in presets:
        return None

    p = presets[name]
    config = {}
    warnings = []

    # 1. Modules
    if "modules" in p:
        valid_mods = [m for m in p["modules"] if m in available_mods]
        config["setup_mods"] = valid_mods

    # 2. Model - apply if available, warn otherwise
    if "model" in p:
        if p["model"] in available_models:
            config["setup_model"] = p["model"]
        else:
            warnings.append(
                f"Model '{p['model']}' not available. Please select an available model."
            )

    # 3. Parameters - already normalized by load_presets()
    if "reranker_model" in p:
        config["setup_reranker"] = p["reranker_model"]
    if "context_window" in p:
        config["setup_ctx"] = p["context_window"]
    if "temperature" in p:
        config["setup_temp"] = p["temperature"]
    if "max_tokens" in p:
        config["setup_max_tokens"] = p["max_tokens"]
    if "reranker_top_n" in p:
        config["setup_top_n"] = p["reranker_top_n"]
    if "confidence_cutoff" in p:
        config["setup_conf"] = p["confidence_cutoff"]
    if "confidence_cutoff_hard" in p:
        config["setup_conf_cutoff_hard"] = p["confidence_cutoff_hard"]
    if "system_prompt" in p:
        config["setup_sys_prompt"] = p["system_prompt"]

    # 4. Devices - only include if present in preset and valid
    if "rag_device" in p and p["rag_device"] in available_devices:
        config["setup_rag_device"] = p["rag_device"]

    if "llm_device" in p and p["llm_device"] in ["cpu", "gpu"]:
        config["setup_llm_device"] = p["llm_device"]

    config["warnings"] = warnings
    return config
