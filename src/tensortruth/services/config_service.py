"""Configuration service - wraps existing config module with dependency injection.

The existing config module is already clean (no Streamlit dependencies).
This service provides a class-based interface with configurable paths.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import yaml

from tensortruth.app_utils.config_schema import TensorTruthConfig
from tensortruth.app_utils.paths import get_user_data_dir


class ConfigService:
    """Service for managing TensorTruth configuration.

    Wraps the existing config module with dependency injection support
    and a class-based interface.
    """

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize config service.

        Args:
            config_file: Path to config file. Defaults to ~/.tensortruth/config.yaml
        """
        if config_file is None:
            self.config_dir = get_user_data_dir()
            self.config_file = self.config_dir / "config.yaml"
        else:
            self.config_file = Path(config_file)
            self.config_dir = self.config_file.parent

    def load(self) -> TensorTruthConfig:
        """Load configuration from YAML file.

        Creates default config if file doesn't exist.

        Returns:
            TensorTruthConfig instance.
        """
        if not self.config_file.exists():
            config = TensorTruthConfig.create_default()
            self.save(config)
            return config

        try:
            with open(self.config_file, "r") as f:
                data = yaml.safe_load(f) or {}
            return TensorTruthConfig.from_dict(data)
        except (yaml.YAMLError, IOError, KeyError):
            return TensorTruthConfig.create_default()

    def save(self, config: TensorTruthConfig) -> None:
        """Save configuration to YAML file.

        Args:
            config: TensorTruthConfig to save.
        """
        self.config_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_file, "w") as f:
            yaml.safe_dump(
                config.to_dict(), f, default_flow_style=False, sort_keys=False
            )

    def update(self, **kwargs: Any) -> TensorTruthConfig:
        """Update specific config values.

        Supports nested updates using prefixed keys:
        - ollama_*: Updates ollama config
        - ui_*: Updates UI config
        - rag_*: Updates RAG config
        - agent_*: Updates agent config
        - models_*: Updates models config
        - history_cleaning_*: Updates history cleaning config

        Args:
            **kwargs: Config values to update.

        Returns:
            Updated TensorTruthConfig.

        Examples:
            service.update(ollama_base_url="http://192.168.1.100:11434")
            service.update(ui_default_temperature=0.5)
        """
        config = self.load()

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
            elif key.startswith("history_cleaning_"):
                attr_name = key.replace("history_cleaning_", "")
                if hasattr(config.history_cleaning, attr_name):
                    setattr(config.history_cleaning, attr_name, value)
            elif key.startswith("web_search_"):
                attr_name = key.replace("web_search_", "")
                if hasattr(config.web_search, attr_name):
                    setattr(config.web_search, attr_name, value)

        self.save(config)
        return config

    def compute_hash(
        self,
        modules: Optional[List[str]],
        params: dict,
        has_pdf_index: bool = False,
        session_id: Optional[str] = None,
    ) -> Optional[Tuple]:
        """Compute a hashable configuration tuple for cache invalidation.

        This creates a stable hash of the current engine configuration to detect
        when the engine needs to be reloaded.

        Args:
            modules: List of active module names.
            params: Session parameters dict.
            has_pdf_index: Whether session has a temporary PDF index.
            session_id: Current session ID to ensure engine reloads on session switch.

        Returns:
            Tuple suitable for comparison, or None if no modules and no PDF index.
        """
        # Sort modules for consistent ordering
        modules_tuple = tuple(sorted(modules)) if modules else None

        # Convert params dict to sorted frozenset for hashing
        param_items = sorted([(k, v) for k, v in params.items()])
        param_hash = frozenset(param_items)

        # Return complete config tuple (None if no modules and no PDF index)
        if modules_tuple or has_pdf_index:
            return (modules_tuple, param_hash, has_pdf_index, session_id)
        else:
            return None

    def get_ollama_url(self) -> str:
        """Get the Ollama base URL from config.

        Returns:
            Ollama base URL string.
        """
        config = self.load()
        return config.ollama.base_url

    def get_default_model(self) -> str:
        """Get the default RAG model from config.

        Returns:
            Default model name.
        """
        config = self.load()
        return config.models.default_rag_model

    def get_intent_classifier_model(self) -> str:
        """Get the intent classifier model from config.

        Returns:
            Intent classifier model name.
        """
        config = self.load()
        return config.agent.intent_classifier_model

    def is_natural_language_agents_enabled(self) -> bool:
        """Check if natural language agent routing is enabled.

        Returns:
            True if enabled.
        """
        config = self.load()
        return config.agent.enable_natural_language_agents
