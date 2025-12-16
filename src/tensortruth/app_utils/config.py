"""Configuration management for Tensor-Truth."""

import os

import yaml

from tensortruth.app_utils.paths import get_user_data_dir

# Standard Ollama env var, or default local
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Use the centralized user data directory from paths.py
CONFIG_DIR = get_user_data_dir()
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def get_config_file_path():
    return CONFIG_FILE


def load_config():
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def save_config(new_config):
    """Save configuration to YAML file, preserving existing keys."""
    current = load_config()
    current.update(new_config)

    # Ensure directory exists (handled by paths.py usually, but safe to double check)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(current, f, default_flow_style=False)


def get_ollama_url():
    """
    Get the effective Ollama URL.
    Priority:
    1. config.yaml ('ollama_url')
    2. OLLAMA_HOST environment variable
    3. Default (http://localhost:11434)
    """
    # 1. Check Config File
    config = load_config()
    if config.get("ollama_url"):
        return config["ollama_url"].rstrip("/")

    # 2. Check Environment Variable
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        # Handle cases where OLLAMA_HOST might be just "0.0.0.0:11434"
        if not env_host.startswith("http"):
            return f"http://{env_host}".rstrip("/")
        return env_host.rstrip("/")

    # 3. Return Default
    return DEFAULT_OLLAMA_URL
