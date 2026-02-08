"""Unit tests for ConfigService."""

from pathlib import Path

import pytest
import yaml

from tensortruth.app_utils.config_schema import AgentConfig, TensorTruthConfig
from tensortruth.services.config_service import ConfigService


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file path."""
    return tmp_path / "config.yaml"


@pytest.fixture
def config_service(temp_config_file: Path) -> ConfigService:
    """Create a ConfigService instance with a temp file."""
    return ConfigService(temp_config_file)


class TestConfigServiceLoad:
    """Tests for ConfigService.load()."""

    def test_load_creates_default_when_missing(
        self, config_service: ConfigService, temp_config_file: Path
    ):
        """Load creates default config when file doesn't exist."""
        config = config_service.load()

        # Default values
        assert config.ollama.base_url == "http://localhost:11434"
        assert config.ui.default_temperature == 0.7
        assert config.agent.max_iterations == 10

        # File should be created
        assert temp_config_file.exists()

    def test_load_existing_config(
        self, config_service: ConfigService, temp_config_file: Path
    ):
        """Load reads existing config correctly."""
        existing_config = {
            "ollama": {"base_url": "http://custom:11434", "timeout": 600},
            "ui": {"default_temperature": 0.5},
            "rag": {"default_device": "cuda"},
            "models": {"default_rag_model": "custom-model"},
            "agent": {"max_iterations": 20},
        }
        temp_config_file.write_text(yaml.safe_dump(existing_config))

        config = config_service.load()

        assert config.ollama.base_url == "http://custom:11434"
        assert config.ollama.timeout == 600
        assert config.ui.default_temperature == 0.5
        assert config.agent.max_iterations == 20

    def test_load_partial_config_uses_defaults(
        self, config_service: ConfigService, temp_config_file: Path
    ):
        """Load uses defaults for missing fields."""
        partial_config = {
            "ollama": {"base_url": "http://custom:11434"},
            # Other sections missing
        }
        temp_config_file.write_text(yaml.safe_dump(partial_config))

        config = config_service.load()

        # Custom value loaded
        assert config.ollama.base_url == "http://custom:11434"
        # Default values for missing fields
        assert config.ui.default_temperature == 0.7
        assert config.agent.max_iterations == 10


class TestConfigServiceSave:
    """Tests for ConfigService.save()."""

    def test_save_creates_file(
        self, config_service: ConfigService, temp_config_file: Path
    ):
        """Save creates config file with correct content."""
        config = config_service.load()
        config.ollama.base_url = "http://saved:11434"

        config_service.save(config)

        saved_data = yaml.safe_load(temp_config_file.read_text())
        assert saved_data["ollama"]["base_url"] == "http://saved:11434"

    def test_save_creates_parent_directory(self, tmp_path: Path):
        """Save creates parent directory if needed."""
        nested_path = tmp_path / "subdir" / "config.yaml"
        service = ConfigService(nested_path)

        config = service.load()
        service.save(config)

        assert nested_path.exists()


class TestConfigServiceUpdate:
    """Tests for ConfigService.update()."""

    def test_update_ollama_config(self, config_service: ConfigService):
        """Update modifies ollama config values."""
        config = config_service.update(ollama_base_url="http://updated:11434")

        assert config.ollama.base_url == "http://updated:11434"

    def test_update_ui_config(self, config_service: ConfigService):
        """Update modifies UI config values."""
        config = config_service.update(ui_default_temperature=0.8)

        assert config.ui.default_temperature == 0.8

    def test_update_rag_config(self, config_service: ConfigService):
        """Update modifies RAG config values."""
        config = config_service.update(rag_default_device="cuda")

        assert config.rag.default_device == "cuda"

    def test_update_agent_config(self, config_service: ConfigService):
        """Update modifies agent config values."""
        config = config_service.update(agent_max_iterations=25)

        assert config.agent.max_iterations == 25

    def test_update_models_config(self, config_service: ConfigService):
        """Update modifies models config values."""
        config = config_service.update(models_default_rag_model="custom-model")

        assert config.models.default_rag_model == "custom-model"

    def test_update_multiple_values(self, config_service: ConfigService):
        """Update modifies multiple values at once."""
        config = config_service.update(
            ollama_base_url="http://multi:11434",
            ui_default_temperature=0.3,
            agent_max_iterations=15,
        )

        assert config.ollama.base_url == "http://multi:11434"
        assert config.ui.default_temperature == 0.3
        assert config.agent.max_iterations == 15

    def test_update_persists_changes(
        self, config_service: ConfigService, temp_config_file: Path
    ):
        """Update saves changes to disk."""
        config_service.update(ollama_base_url="http://persisted:11434")

        # Reload and verify
        saved_data = yaml.safe_load(temp_config_file.read_text())
        assert saved_data["ollama"]["base_url"] == "http://persisted:11434"


class TestConfigServiceComputeHash:
    """Tests for ConfigService.compute_hash()."""

    def test_compute_hash_with_modules(self, config_service: ConfigService):
        """Compute hash returns tuple for modules."""
        hash_val = config_service.compute_hash(
            modules=["mod1", "mod2"],
            params={"temperature": 0.5},
            has_pdf_index=False,
        )

        assert hash_val is not None
        assert isinstance(hash_val, tuple)

    def test_compute_hash_same_params_same_hash(self, config_service: ConfigService):
        """Same parameters produce same hash."""
        hash1 = config_service.compute_hash(
            modules=["mod1", "mod2"],
            params={"a": 1, "b": 2},
        )
        hash2 = config_service.compute_hash(
            modules=["mod2", "mod1"],  # Different order
            params={"b": 2, "a": 1},  # Different order
        )

        assert hash1 == hash2

    def test_compute_hash_different_params_different_hash(
        self, config_service: ConfigService
    ):
        """Different parameters produce different hash."""
        hash1 = config_service.compute_hash(
            modules=["mod1"],
            params={"temperature": 0.5},
        )
        hash2 = config_service.compute_hash(
            modules=["mod1"],
            params={"temperature": 0.7},
        )

        assert hash1 != hash2

    def test_compute_hash_none_without_modules(self, config_service: ConfigService):
        """Compute hash returns None when no modules and no PDF index."""
        hash_val = config_service.compute_hash(
            modules=None,
            params={},
            has_pdf_index=False,
        )

        assert hash_val is None

    def test_compute_hash_with_pdf_index_only(self, config_service: ConfigService):
        """Compute hash returns tuple when PDF index present (no modules)."""
        hash_val = config_service.compute_hash(
            modules=None,
            params={},
            has_pdf_index=True,
        )

        assert hash_val is not None

    def test_compute_hash_session_id_affects_hash(self, config_service: ConfigService):
        """Session ID is included in hash for cache invalidation."""
        hash1 = config_service.compute_hash(
            modules=["mod1"],
            params={},
            session_id="session-1",
        )
        hash2 = config_service.compute_hash(
            modules=["mod1"],
            params={},
            session_id="session-2",
        )

        assert hash1 != hash2


class TestConfigServiceHelpers:
    """Tests for helper methods."""

    def test_get_ollama_url(self, config_service: ConfigService):
        """Get ollama URL returns correct value."""
        url = config_service.get_ollama_url()

        assert url == "http://localhost:11434"

    def test_get_default_model(self, config_service: ConfigService):
        """Get default model returns correct value."""
        model = config_service.get_default_model()

        assert model is not None
        assert isinstance(model, str)

    def test_get_intent_classifier_model(self, config_service: ConfigService):
        """Get intent classifier model returns correct value."""
        model = config_service.get_intent_classifier_model()

        assert model == "llama3.2:3b"

    def test_is_natural_language_agents_enabled(self, config_service: ConfigService):
        """Check natural language agents enabled returns correct value."""
        enabled = config_service.is_natural_language_agents_enabled()

        assert enabled is True  # Default


class TestHistoryCleaningConfig:
    """Tests for history cleaning configuration."""

    def test_default_config_has_history_cleaning(self, config_service: ConfigService):
        """Default config includes history_cleaning section."""
        config = config_service.load()
        assert hasattr(config, "history_cleaning")
        assert config.history_cleaning.enabled is True

    def test_update_history_cleaning_enabled(self, config_service: ConfigService):
        """Can update history_cleaning_enabled via prefixed key."""
        config = config_service.update(history_cleaning_enabled=False)
        assert config.history_cleaning.enabled is False

    def test_update_history_cleaning_remove_emojis(self, config_service: ConfigService):
        """Can update history_cleaning_remove_emojis via prefixed key."""
        config = config_service.update(history_cleaning_remove_emojis=False)
        assert config.history_cleaning.remove_emojis is False

    def test_update_multiple_history_cleaning_options(
        self, config_service: ConfigService
    ):
        """Can update multiple history cleaning options at once."""
        config = config_service.update(
            history_cleaning_enabled=True,
            history_cleaning_remove_filler_phrases=False,
            history_cleaning_collapse_newlines=False,
        )
        assert config.history_cleaning.enabled is True
        assert config.history_cleaning.remove_filler_phrases is False
        assert config.history_cleaning.collapse_newlines is False

    def test_update_history_cleaning_persists(
        self, config_service: ConfigService, temp_config_file: Path
    ):
        """History cleaning updates are persisted to disk."""
        config_service.update(history_cleaning_enabled=False)

        # Reload and verify
        saved_data = yaml.safe_load(temp_config_file.read_text())
        assert saved_data["history_cleaning"]["enabled"] is False


class TestConfigUpdatePrefixStripping:
    """Tests for prefix stripping in ConfigService.update().

    Regression tests for the bug where .replace("agent_", "") would strip
    ALL occurrences of "agent_", causing keys like "agent_function_agent_model"
    to resolve to "function_model" instead of "function_agent_model".
    """

    def test_update_agent_function_agent_model(self, config_service: ConfigService):
        """agent_function_agent_model strips only the first 'agent_' prefix."""
        config = config_service.update(agent_function_agent_model="test-model:7b")
        assert config.agent.function_agent_model == "test-model:7b"

    def test_update_agent_function_agent_model_persists(
        self, config_service: ConfigService, temp_config_file: Path
    ):
        """agent_function_agent_model update persists to disk."""
        config_service.update(agent_function_agent_model="persisted-model:7b")

        saved_data = yaml.safe_load(temp_config_file.read_text())
        assert saved_data["agent"]["function_agent_model"] == "persisted-model:7b"

    def test_update_agent_router_model(self, config_service: ConfigService):
        """agent_router_model still works (no repeated prefix)."""
        config = config_service.update(agent_router_model="router:3b")
        assert config.agent.router_model == "router:3b"

    def test_update_models_default_agent_reasoning_model(
        self, config_service: ConfigService
    ):
        """models_default_agent_reasoning_model strips only 'models_' prefix."""
        config = config_service.update(
            models_default_agent_reasoning_model="reasoning:14b"
        )
        assert config.models.default_agent_reasoning_model == "reasoning:14b"

    def test_update_unknown_key_is_ignored(self, config_service: ConfigService):
        """Unknown keys are silently ignored (with logging)."""
        config = config_service.update(nonexistent_key="value")
        # Should not raise, config should be saved with defaults
        assert config.ollama.base_url == "http://localhost:11434"

    def test_update_unknown_attr_under_valid_prefix(
        self, config_service: ConfigService
    ):
        """Valid prefix but unknown attribute is ignored (with logging)."""
        config = config_service.update(agent_nonexistent_field="value")
        # Should not raise
        assert config.agent.max_iterations == 10

    def test_update_all_prefix_patterns(self, config_service: ConfigService):
        """All config prefix patterns resolve correctly."""
        config = config_service.update(
            ollama_timeout=600,
            ui_default_temperature=0.5,
            rag_default_device="cuda",
            agent_max_iterations=20,
            models_default_rag_model="custom:14b",
            history_cleaning_enabled=False,
            web_search_ddg_max_results=20,
        )
        assert config.ollama.timeout == 600
        assert config.ui.default_temperature == 0.5
        assert config.rag.default_device == "cuda"
        assert config.agent.max_iterations == 20
        assert config.models.default_rag_model == "custom:14b"
        assert config.history_cleaning.enabled is False
        assert config.web_search.ddg_max_results == 20


class TestAgentReasoningModelRemoved:
    """Verify that the dead agent.reasoning_model field was removed."""

    def test_agent_config_no_reasoning_model_field(self):
        """AgentConfig should not have reasoning_model field."""
        config = AgentConfig()
        assert not hasattr(config, "reasoning_model")

    def test_default_config_no_agent_reasoning_model(
        self, config_service: ConfigService
    ):
        """Default config should not have agent.reasoning_model."""
        config = config_service.load()
        assert not hasattr(config.agent, "reasoning_model")

    def test_models_reasoning_model_still_exists(self, config_service: ConfigService):
        """models.default_agent_reasoning_model should still exist."""
        config = config_service.load()
        assert hasattr(config.models, "default_agent_reasoning_model")


class TestConfigBackwardCompatibility:
    """Verify backward compatibility with old config files."""

    def test_load_config_with_extra_agent_fields(
        self, config_service: ConfigService, temp_config_file: Path
    ):
        """Loading config with old reasoning_model field should not crash."""
        old_config = {
            "agent": {
                "max_iterations": 10,
                "reasoning_model": "old-model:7b",  # removed field
                "router_model": "llama3.2:3b",
                "function_agent_model": "llama3.1:8b",
            },
        }
        temp_config_file.write_text(yaml.safe_dump(old_config))

        config = config_service.load()
        assert config.agent.max_iterations == 10
        assert config.agent.router_model == "llama3.2:3b"
        assert not hasattr(config.agent, "reasoning_model")

    def test_from_dict_ignores_unknown_fields_in_all_sections(self):
        """from_dict ignores unknown fields in any config section."""
        data = {
            "ollama": {"base_url": "http://localhost:11434", "unknown_field": True},
            "agent": {"max_iterations": 5, "deprecated_field": "value"},
        }
        config = TensorTruthConfig.from_dict(data)
        assert config.ollama.base_url == "http://localhost:11434"
        assert config.agent.max_iterations == 5
