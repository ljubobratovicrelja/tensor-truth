"""Configuration management for Tensor-Truth."""

import yaml

from tensortruth.app_utils.config_schema import TensorTruthConfig
from tensortruth.app_utils.paths import get_user_data_dir

# Use the centralized user data directory from paths.py
CONFIG_DIR = get_user_data_dir()
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def get_config_file_path():
    """Get the path to the config file."""
    return CONFIG_FILE


def load_config() -> TensorTruthConfig:
    """Load configuration from YAML file, creating default if it doesn't exist."""
    if not CONFIG_FILE.exists():
        # Create default config on first run
        config = TensorTruthConfig.create_default()
        save_config(config)
        return config

    try:
        with open(CONFIG_FILE, "r") as f:
            data = yaml.safe_load(f) or {}
        return TensorTruthConfig.from_dict(data)
    except Exception:
        # If config is corrupted, return defaults
        return TensorTruthConfig.create_default()


def save_config(config: TensorTruthConfig):
    """Save configuration to YAML file."""
    # Ensure directory exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def update_config(**kwargs):
    """
    Update specific config values.

    Examples:
        update_config(ollama_base_url="http://192.168.1.100:11434")
        update_config(ui_default_temperature=0.5)
        update_config(rag_default_device="cuda")
        update_config(agent_max_iterations=15)
    """
    config = load_config()

    # Parse kwargs to update nested config
    for key, value in kwargs.items():
        if key.startswith("ollama_"):
            attr_name = key.replace("ollama_", "")
            if hasattr(config.ollama, attr_name):
                setattr(config.ollama, attr_name, value)
        elif key.startswith("ui_"):
            attr_name = key.replace("ui_", "")
            if hasattr(config.ui, attr_name):
                setattr(config.ui, attr_name, value)
        elif key.startswith("rag_"):
            attr_name = key.replace("rag_", "")
            if hasattr(config.rag, attr_name):
                setattr(config.rag, attr_name, value)
        elif key.startswith("agent_"):
            attr_name = key.replace("agent_", "")
            if hasattr(config.agent, attr_name):
                setattr(config.agent, attr_name, value)
        elif key.startswith("models_"):
            attr_name = key.replace("models_", "")
            if hasattr(config.models, attr_name):
                setattr(config.models, attr_name, value)

    save_config(config)


def compute_config_hash(
    modules: list,
    params: dict,
    has_pdf_index: bool = False,
    session_id: str = None,
):
    """Compute a hashable configuration tuple for cache invalidation.

    This creates a stable hash of the current engine configuration to detect
    when the engine needs to be reloaded.

    Args:
        modules: List of active module names
        params: Session parameters dict
        has_pdf_index: Whether session has a temporary PDF index
        session_id: Current session ID to ensure engine reloads on session switch

    Returns:
        Tuple suitable for comparison (modules_tuple, params_frozenset, pdf_flag, session_id)
    """
    # Sort modules for consistent ordering
    modules_tuple = tuple(sorted(modules)) if modules else None

    # Convert params dict to sorted frozenset for hashing
    param_items = sorted([(k, v) for k, v in params.items()])
    param_hash = frozenset(param_items)

    # Return complete config tuple (None if no modules and no PDF index)
    # IMPORTANT: Include session_id to prevent session index contamination
    if modules_tuple or has_pdf_index:
        return (modules_tuple, param_hash, has_pdf_index, session_id)
    else:
        return None
