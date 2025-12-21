"""Unit tests for configuration system."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from tensortruth.app_utils.config import (
    compute_config_hash,
    load_config,
    save_config,
    update_config,
)
from tensortruth.app_utils.config_schema import (
    OllamaConfig,
    RAGConfig,
    TensorTruthConfig,
    UIConfig,
)


@pytest.fixture
def temp_config_dir(monkeypatch):
    """Create a temporary directory for config files during tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch the CONFIG_DIR and CONFIG_FILE in the config module
        from tensortruth.app_utils import config as config_module

        config_module.CONFIG_DIR = Path(tmpdir)
        config_module.CONFIG_FILE = Path(tmpdir) / "config.yaml"
        yield Path(tmpdir)


class TestConfigSchema:
    """Test configuration schema classes."""

    def test_ollama_config_defaults(self):
        """Test OllamaConfig default values."""
        config = OllamaConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.timeout == 300

    def test_ui_config_defaults(self):
        """Test UIConfig default values."""
        config = UIConfig()
        assert config.default_temperature == 0.3
        assert config.default_context_window == 16384
        assert config.default_reranker == "BAAI/bge-reranker-v2-m3"
        assert config.default_top_n == 3
        assert config.default_confidence_threshold == 0.3

    def test_rag_config_defaults(self):
        """Test RAGConfig default values."""
        config = RAGConfig()
        assert config.default_device == "cpu"

    def test_config_to_dict(self):
        """Test TensorTruthConfig serialization to dict."""
        config = TensorTruthConfig(
            ollama=OllamaConfig(),
            ui=UIConfig(),
            rag=RAGConfig(default_device="cuda"),
        )
        data = config.to_dict()

        assert data["ollama"]["base_url"] == "http://localhost:11434"
        assert data["ui"]["default_temperature"] == 0.3
        assert data["rag"]["default_device"] == "cuda"

    def test_config_from_dict(self):
        """Test TensorTruthConfig deserialization from dict."""
        data = {
            "ollama": {"base_url": "http://192.168.1.100:11434", "timeout": 600},
            "ui": {
                "default_temperature": 0.5,
                "default_context_window": 8192,
                "default_reranker": "BAAI/bge-reranker-base",
                "default_top_n": 5,
                "default_confidence_threshold": 0.4,
            },
            "rag": {"default_device": "mps"},
        }
        config = TensorTruthConfig.from_dict(data)

        assert config.ollama.base_url == "http://192.168.1.100:11434"
        assert config.ollama.timeout == 600
        assert config.ui.default_temperature == 0.5
        assert config.ui.default_context_window == 8192
        assert config.rag.default_device == "mps"

    def test_config_from_dict_with_missing_keys(self):
        """Test TensorTruthConfig handles missing keys gracefully."""
        data = {
            "ollama": {"base_url": "http://custom:11434"},
            # Missing timeout, should use default
            "ui": {},  # All missing, should use defaults
            "rag": {},  # Missing, should use default
        }
        config = TensorTruthConfig.from_dict(data)

        assert config.ollama.base_url == "http://custom:11434"
        assert config.ollama.timeout == 300  # Default
        assert config.ui.default_temperature == 0.3  # Default
        assert config.rag.default_device == "cpu"  # Default


class TestDeviceDetection:
    """Test smart device detection for different platforms."""

    def test_detect_mps_device(self):
        """Test detection of MPS device on Apple Silicon."""
        # Mock torch module inside the function where it's imported
        with patch("builtins.__import__") as mock_import:
            # Create mock torch module
            mock_torch = Mock()
            mock_torch.backends.mps.is_available.return_value = True
            mock_torch.cuda.is_available.return_value = False

            # Make __import__ return our mock when importing torch
            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    return mock_torch
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            device = TensorTruthConfig._detect_default_device()
            assert device == "mps"

    def test_detect_cuda_device(self):
        """Test detection of CUDA device on NVIDIA GPU systems."""
        with patch("builtins.__import__") as mock_import:
            mock_torch = Mock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.backends.mps.is_available.return_value = False

            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    return mock_torch
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            device = TensorTruthConfig._detect_default_device()
            assert device == "cuda"

    def test_detect_cpu_fallback(self):
        """Test fallback to CPU when no GPU is available."""
        with patch("builtins.__import__") as mock_import:
            mock_torch = Mock()
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False

            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    return mock_torch
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            device = TensorTruthConfig._detect_default_device()
            assert device == "cpu"

    def test_detect_device_cuda_priority_over_cpu(self):
        """Test that MPS is preferred over CUDA when both available."""
        with patch("builtins.__import__") as mock_import:
            mock_torch = Mock()
            # Both available - MPS checked first
            mock_torch.backends.mps.is_available.return_value = True
            mock_torch.cuda.is_available.return_value = True

            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    return mock_torch
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            device = TensorTruthConfig._detect_default_device()
            # MPS is checked first, so should return mps
            assert device == "mps"

    def test_detect_device_with_import_error(self):
        """Test graceful fallback when torch import fails."""
        # Temporarily hide torch
        import sys

        original_torch = sys.modules.get("torch")
        if "torch" in sys.modules:
            del sys.modules["torch"]

        try:
            # This should not crash, just return cpu
            device = TensorTruthConfig._detect_default_device()
            assert device == "cpu"
        finally:
            # Restore torch
            if original_torch:
                sys.modules["torch"] = original_torch


class TestConfigFileOperations:
    """Test config file loading, saving, and updating."""

    def test_create_default_config(self, temp_config_dir):
        """Test that default config is created on first load."""
        from tensortruth.app_utils import config as config_module

        config_file = config_module.CONFIG_FILE

        # Ensure no config exists
        assert not config_file.exists()

        # Load config should create default
        config = load_config()

        # Verify file was created
        assert config_file.exists()

        # Verify config has expected defaults
        assert config.ollama.base_url == "http://localhost:11434"
        assert config.ui.default_temperature == 0.3

    def test_default_config_with_mps(self, temp_config_dir):
        """Test that default config detects MPS and saves it to file."""
        from tensortruth.app_utils import config as config_module

        with patch("builtins.__import__") as mock_import:
            mock_torch = Mock()
            mock_torch.backends.mps.is_available.return_value = True
            mock_torch.cuda.is_available.return_value = False

            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    return mock_torch
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            config_file = config_module.CONFIG_FILE

            # Load config (creates default with MPS)
            config = load_config()

            # Verify MPS was detected
            assert config.rag.default_device == "mps"

            # Verify file contains MPS
            with open(config_file, "r") as f:
                data = yaml.safe_load(f)
            assert data["rag"]["default_device"] == "mps"

    def test_default_config_with_cuda(self, temp_config_dir):
        """Test that default config detects CUDA and saves it to file."""
        from tensortruth.app_utils import config as config_module

        with patch("builtins.__import__") as mock_import:
            mock_torch = Mock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.backends.mps.is_available.return_value = False

            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    return mock_torch
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            config_file = config_module.CONFIG_FILE

            # Load config (creates default with CUDA)
            config = load_config()

            # Verify CUDA was detected
            assert config.rag.default_device == "cuda"

            # Verify file contains CUDA
            with open(config_file, "r") as f:
                data = yaml.safe_load(f)
            assert data["rag"]["default_device"] == "cuda"

    def test_save_and_load_config(self, temp_config_dir):
        """Test saving and loading config maintains data integrity."""
        # Create custom config
        config = TensorTruthConfig(
            ollama=OllamaConfig(base_url="http://custom:11434", timeout=600),
            ui=UIConfig(default_temperature=0.7, default_top_n=5),
            rag=RAGConfig(default_device="cuda"),
        )

        # Save it
        save_config(config)

        # Load it back
        loaded_config = load_config()

        # Verify all values match
        assert loaded_config.ollama.base_url == "http://custom:11434"
        assert loaded_config.ollama.timeout == 600
        assert loaded_config.ui.default_temperature == 0.7
        assert loaded_config.ui.default_top_n == 5
        assert loaded_config.rag.default_device == "cuda"

    def test_update_config_ollama(self, temp_config_dir):
        """Test updating Ollama config values."""
        # Create initial config
        load_config()  # Creates default

        # Update ollama values
        update_config(ollama_base_url="http://192.168.1.50:11434", ollama_timeout=900)

        # Load and verify
        config = load_config()
        assert config.ollama.base_url == "http://192.168.1.50:11434"
        assert config.ollama.timeout == 900

    def test_update_config_ui(self, temp_config_dir):
        """Test updating UI config values."""
        # Create initial config
        load_config()

        # Update UI values
        update_config(
            ui_default_temperature=0.8,
            ui_default_top_n=10,
            ui_default_context_window=8192,
        )

        # Load and verify
        config = load_config()
        assert config.ui.default_temperature == 0.8
        assert config.ui.default_top_n == 10
        assert config.ui.default_context_window == 8192

    def test_update_config_rag(self, temp_config_dir):
        """Test updating RAG config values."""
        # Create initial config
        load_config()

        # Update RAG values
        update_config(rag_default_device="cuda")

        # Load and verify
        config = load_config()
        assert config.rag.default_device == "cuda"

    def test_update_config_mixed(self, temp_config_dir):
        """Test updating multiple config sections at once."""
        load_config()

        # Update across sections
        update_config(
            ollama_base_url="http://custom:11434",
            ui_default_temperature=0.5,
            rag_default_device="mps",
        )

        # Load and verify
        config = load_config()
        assert config.ollama.base_url == "http://custom:11434"
        assert config.ui.default_temperature == 0.5
        assert config.rag.default_device == "mps"

    def test_update_config_ignores_invalid_keys(self, temp_config_dir):
        """Test that update_config ignores invalid keys gracefully."""
        load_config()

        # Try to update with invalid keys - should not crash
        update_config(
            ollama_base_url="http://valid:11434",
            invalid_section_key="should_be_ignored",
            ollama_nonexistent="also_ignored",
        )

        # Valid update should still work
        config = load_config()
        assert config.ollama.base_url == "http://valid:11434"

    def test_load_corrupted_config_returns_default(self, temp_config_dir):
        """Test that corrupted config file returns defaults."""
        from tensortruth.app_utils import config as config_module

        config_file = config_module.CONFIG_FILE

        # Write corrupted YAML
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            f.write("this is not valid: yaml: content: [[[")

        # Load should return defaults without crashing
        config = load_config()
        assert config.ollama.base_url == "http://localhost:11434"

    def test_config_file_structure(self, temp_config_dir):
        """Test that saved config file has correct YAML structure."""
        from tensortruth.app_utils import config as config_module

        config_file = config_module.CONFIG_FILE

        # Create and save config
        load_config()

        # Read raw YAML and verify structure
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)

        # Verify top-level sections exist
        assert "ollama" in data
        assert "ui" in data
        assert "rag" in data

        # Verify nested structure
        assert "base_url" in data["ollama"]
        assert "timeout" in data["ollama"]
        assert "default_temperature" in data["ui"]
        assert "default_device" in data["rag"]


class TestConfigHash:
    """Test config hash computation for cache invalidation."""

    def test_compute_config_hash_basic(self):
        """Test basic config hash computation."""
        modules = ["pytorch", "numpy"]
        params = {"model": "deepseek-r1:8b", "temperature": 0.3}
        has_pdf_index = False
        session_id = "sess_123"

        hash_val = compute_config_hash(modules, params, has_pdf_index, session_id)

        # Should return a tuple
        assert isinstance(hash_val, tuple)
        assert len(hash_val) == 4

        # Should contain sorted modules
        assert hash_val[0] == ("numpy", "pytorch")
        assert hash_val[2] is False
        assert hash_val[3] == "sess_123"

    def test_compute_config_hash_with_pdf_index(self):
        """Test config hash with PDF index."""
        modules = ["pytorch"]
        params = {"model": "qwen", "temperature": 0.5}
        has_pdf_index = True
        session_id = "sess_abc"

        hash_val = compute_config_hash(modules, params, has_pdf_index, session_id)

        assert hash_val[2] is True
        assert hash_val[3] == "sess_abc"

    def test_compute_config_hash_different_sessions(self):
        """Test that different session IDs produce different hashes."""
        modules = []
        params = {"model": "qwen"}
        has_pdf_index = True

        hash1 = compute_config_hash(modules, params, has_pdf_index, "sess_abc")
        hash2 = compute_config_hash(modules, params, has_pdf_index, "sess_xyz")

        # Should be different due to session_id
        assert hash1 != hash2
        assert hash1[3] == "sess_abc"
        assert hash2[3] == "sess_xyz"

    def test_compute_config_hash_no_modules_no_pdf(self):
        """Test that hash is None when no modules and no PDF index."""
        modules = []
        params = {"model": "qwen"}
        has_pdf_index = False
        session_id = "sess_123"

        hash_val = compute_config_hash(modules, params, has_pdf_index, session_id)

        # Should return None
        assert hash_val is None

    def test_compute_config_hash_module_order_invariant(self):
        """Test that module order doesn't affect hash."""
        params = {"model": "qwen"}
        session_id = "sess_123"

        hash1 = compute_config_hash(["pytorch", "numpy"], params, False, session_id)
        hash2 = compute_config_hash(["numpy", "pytorch"], params, False, session_id)

        # Should be identical (modules are sorted)
        assert hash1 == hash2

    def test_compute_config_hash_param_changes(self):
        """Test that parameter changes produce different hashes."""
        modules = ["pytorch"]
        session_id = "sess_123"

        hash1 = compute_config_hash(modules, {"temperature": 0.3}, False, session_id)
        hash2 = compute_config_hash(modules, {"temperature": 0.5}, False, session_id)

        # Should be different
        assert hash1 != hash2
