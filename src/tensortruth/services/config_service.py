"""Configuration service - wraps existing config module with dependency injection.

This service provides a class-based interface with configurable paths.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import yaml

from tensortruth.app_utils.config_schema import TensorTruthConfig
from tensortruth.app_utils.paths import get_user_data_dir

logger = logging.getLogger(__name__)


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
        - llm_*: Updates LLM config
        - rag_*: Updates RAG config
        - conversation_*: Updates conversation config
        - agent_*: Updates agent config
        - history_cleaning_*: Updates history cleaning config
        - web_search_*: Updates web search config

        Args:
            **kwargs: Config values to update.

        Returns:
            Updated TensorTruthConfig.

        Examples:
            service.update(ollama_base_url="http://192.168.1.100:11434")
            service.update(llm_default_temperature=0.5)
        """
        config = self.load()

        # Map of prefix -> config section for dispatching updates
        sections = {
            "ollama_": config.ollama,
            "llm_": config.llm,
            "rag_": config.rag,
            "conversation_": config.conversation,
            "agent_": config.agent,
            "history_cleaning_": config.history_cleaning,
            "web_search_": config.web_search,
        }

        for key, value in kwargs.items():
            updated = False
            for prefix, section in sections.items():
                if key.startswith(prefix):
                    attr_name = key.replace(prefix, "", 1)
                    if hasattr(section, attr_name):
                        setattr(section, attr_name, value)
                        updated = True
                    else:
                        logger.warning(
                            "Config key '%s' matched prefix '%s' but attribute "
                            "'%s' not found on %s",
                            key,
                            prefix,
                            attr_name,
                            type(section).__name__,
                        )
                    break
            if not updated:
                logger.warning("Unknown config key ignored: '%s'", key)

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
        return config.llm.default_model

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
