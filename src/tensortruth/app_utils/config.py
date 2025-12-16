"""Configuration management for Tensor-Truth."""

import yaml

from tensortruth.app_utils.paths import get_user_data_dir

# Use the centralized user data directory from paths.py
CONFIG_DIR = get_user_data_dir()
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def get_config_file_path():
    return CONFIG_FILE


def load_config():
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        return {}

    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f) or {}


def save_config(new_config):
    """Save configuration to YAML file, preserving existing keys."""
    current = load_config()
    current.update(new_config)

    # Ensure directory exists (handled by paths.py usually, but safe to double check)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(current, f, default_flow_style=False)
