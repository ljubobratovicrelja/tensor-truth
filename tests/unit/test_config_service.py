"""Unit tests for ConfigService."""

import yaml
from pathlib import Path

import pytest

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
        assert config.ui.default_temperature == 0.1
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
        assert config.ui.default_temperature == 0.1
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
