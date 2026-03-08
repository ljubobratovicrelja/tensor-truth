"""
Unit tests for config migration: ollama section -> providers list.
"""

import pytest

from tensortruth.app_utils.config_schema import TensorTruthConfig


@pytest.mark.unit
class TestConfigMigration:
    """Tests for _migrate_config_data providers migration."""

    def test_migrates_ollama_to_providers(self):
        """Old ollama-only config gets a providers list synthesized."""
        data = {
            "ollama": {
                "base_url": "http://gpu:11434",
                "timeout": 600,
            },
            "llm": {"default_model": "qwen3:32b"},
        }
        migrated = TensorTruthConfig._migrate_config_data(data)
        assert "providers" in migrated
        providers = migrated["providers"]
        assert len(providers) == 1
        assert providers[0]["id"] == "ollama"
        assert providers[0]["type"] == "ollama"
        assert providers[0]["base_url"] == "http://gpu:11434"
        assert providers[0]["timeout"] == 600

    def test_preserves_existing_providers(self):
        """Config with providers already set is not migrated."""
        data = {
            "providers": [
                {
                    "id": "ollama",
                    "type": "ollama",
                    "base_url": "http://localhost:11434",
                },
                {
                    "id": "vllm",
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                },
            ],
        }
        migrated = TensorTruthConfig._migrate_config_data(data)
        assert len(migrated["providers"]) == 2

    def test_migrates_missing_ollama_section(self):
        """Config with neither ollama nor providers gets default provider."""
        data = {"llm": {"default_model": "llama3.2"}}
        migrated = TensorTruthConfig._migrate_config_data(data)
        assert "providers" in migrated
        assert len(migrated["providers"]) == 1
        assert migrated["providers"][0]["base_url"] == "http://localhost:11434"

    def test_default_ollama_values(self):
        """Empty ollama section produces default values."""
        data = {"ollama": {}}
        migrated = TensorTruthConfig._migrate_config_data(data)
        p = migrated["providers"][0]
        assert p["base_url"] == "http://localhost:11434"
        assert p["timeout"] == 300


@pytest.mark.unit
class TestProviderConfigParsing:
    """Tests for ProviderConfig parsing in TensorTruthConfig."""

    def test_from_dict_parses_providers(self):
        data = {
            "providers": [
                {
                    "id": "ollama",
                    "type": "ollama",
                    "base_url": "http://localhost:11434",
                },
                {
                    "id": "vllm",
                    "type": "openai_compatible",
                    "base_url": "http://localhost:8000/v1",
                    "api_key": "test",
                    "models": [
                        {
                            "name": "llama-70b",
                            "capabilities": ["tools"],
                            "context_window": 131072,
                        },
                    ],
                },
            ],
        }
        config = TensorTruthConfig.from_dict(data)
        assert len(config.providers) == 2
        assert config.providers[0].id == "ollama"
        assert config.providers[0].type == "ollama"
        assert config.providers[1].id == "vllm"
        assert config.providers[1].type == "openai_compatible"
        assert config.providers[1].api_key == "test"
        assert len(config.providers[1].models) == 1
        assert config.providers[1].models[0]["name"] == "llama-70b"

    def test_to_dict_serializes_providers(self):
        config = TensorTruthConfig.create_default()
        d = config.to_dict()
        assert "providers" in d
        assert len(d["providers"]) >= 1
        assert d["providers"][0]["id"] == "ollama"

    def test_create_default_has_ollama_provider(self):
        config = TensorTruthConfig.create_default()
        assert len(config.providers) >= 1
        ollama = config.providers[0]
        assert ollama.id == "ollama"
        assert ollama.type == "ollama"

    def test_backward_compat_ollama_accessor(self):
        """The config.ollama accessor should return config from first Ollama provider."""
        config = TensorTruthConfig.create_default()
        assert config.ollama.base_url == "http://localhost:11434"
