"""
Unit tests for tensortruth.app_utils.commands module.
"""

from unittest.mock import MagicMock, patch

import pytest

from tensortruth.app_utils.commands import process_command


@pytest.fixture
def base_session():
    """Sample session data for testing."""
    return {
        "messages": [],
        "modules": ["pytorch"],
        "params": {
            "model": "deepseek-r1:8b",
            "temperature": 0.3,
            "context_window": 4096,
            "confidence_cutoff": 0.3,
            "rag_device": "cuda",
            "llm_device": "gpu",
        },
    }


@pytest.fixture
def available_modules():
    """Available knowledge base modules."""
    return ["pytorch", "tensorflow", "jax"]


@pytest.fixture
def mock_streamlit():
    """Mock streamlit session_state."""
    mock_st = MagicMock()
    mock_st.session_state = {"loaded_config": None}
    return mock_st


# ============================================================================
# Tests for /list and /status commands
# ============================================================================


@pytest.mark.unit
class TestListStatusCommands:
    """Tests for /list, /ls, and /status commands."""

    @patch("tensortruth.app_utils.helpers.get_ollama_ps")
    @patch("tensortruth.app_utils.commands.get_system_devices")
    def test_list_command(
        self, mock_get_devices, mock_get_ps, base_session, available_modules
    ):
        """Test /list command shows status correctly."""
        mock_get_devices.return_value = ["cpu", "cuda"]
        mock_get_ps.return_value = []

        is_cmd, response, state_modifier = process_command(
            "/list", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "Knowledge Base & System Status" in response
        assert "pytorch" in response
        assert "✅" in response  # Active module marker
        assert "deepseek-r1:8b" in response

    @patch("tensortruth.app_utils.helpers.get_ollama_ps")
    @patch("tensortruth.app_utils.commands.get_system_devices")
    def test_status_command(
        self, mock_get_devices, mock_get_ps, base_session, available_modules
    ):
        """Test /status command (alias for /list)."""
        mock_get_devices.return_value = ["cpu", "cuda"]
        mock_get_ps.return_value = []

        is_cmd, response, state_modifier = process_command(
            "/status", base_session, available_modules
        )

        assert is_cmd is True
        assert "Knowledge Base & System Status" in response

    @patch("tensortruth.app_utils.helpers.get_ollama_ps")
    @patch("tensortruth.app_utils.commands.get_system_devices")
    def test_list_with_running_model(
        self, mock_get_devices, mock_get_ps, base_session, available_modules
    ):
        """Test /list command with running Ollama model."""
        mock_get_devices.return_value = ["cpu", "cuda"]
        mock_get_ps.return_value = [
            {
                "name": "deepseek-r1:8b",
                "size_vram": 5500000000,  # 5.5GB
                "size": 5500000000,
                "details": {"parameter_size": "8B"},
            }
        ]

        is_cmd, response, state_modifier = process_command(
            "/list", base_session, available_modules
        )

        assert is_cmd is True
        assert "Ollama Runtime" in response
        assert "5.12 GB" in response or "5.13 GB" in response  # VRAM usage
        assert "8B" in response  # Parameters


# ============================================================================
# Tests for /help command
# ============================================================================


@pytest.mark.unit
class TestHelpCommand:
    """Tests for /help command."""

    def test_help_command(self, base_session, available_modules):
        """Test /help command shows all commands."""
        is_cmd, response, state_modifier = process_command(
            "/help", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "Command Reference" in response
        assert "/list" in response
        assert "/model" in response
        assert "/load" in response
        assert "/unload" in response
        assert "/reload" in response
        assert "/device" in response
        assert "/conf" in response
        assert "/search" in response or "/web" in response


# ============================================================================
# Tests for /model command
# ============================================================================


@pytest.mark.unit
class TestModelCommand:
    """Tests for /model command."""

    @patch("tensortruth.app_utils.commands.get_ollama_models")
    @patch("tensortruth.app_utils.helpers.get_ollama_ps")
    def test_model_no_args_shows_info(
        self, mock_get_ps, mock_get_models, base_session, available_modules
    ):
        """Test /model without args shows current model info."""
        mock_get_models.return_value = ["deepseek-r1:8b", "llama2:7b"]
        mock_get_ps.return_value = [
            {
                "name": "deepseek-r1:8b",
                "size_vram": 5500000000,
                "size": 5500000000,
                "details": {"parameter_size": "8B"},
            }
        ]

        is_cmd, response, state_modifier = process_command(
            "/model", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "Current Model Configuration" in response
        assert "deepseek-r1:8b" in response
        assert "Available Models" in response
        assert "llama2:7b" in response
        assert "✅" in response  # Current model marker

    @patch("tensortruth.app_utils.commands.get_ollama_models")
    @patch("tensortruth.app_utils.commands.st")
    def test_model_switch_valid(
        self, mock_st, mock_get_models, base_session, available_modules
    ):
        """Test /model <name> switches to valid model."""
        mock_get_models.return_value = ["deepseek-r1:8b", "llama2:7b"]
        mock_st.session_state = MagicMock()
        mock_st.session_state.loaded_config = None

        is_cmd, response, state_modifier = process_command(
            "/model llama2:7b", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is not None
        assert "Model switched to" in response
        assert "llama2:7b" in response

        # Test state modifier applies the change
        state_modifier()
        assert base_session["params"]["model"] == "llama2:7b"
        assert mock_st.session_state.loaded_config is None

    @patch("tensortruth.app_utils.commands.get_ollama_models")
    def test_model_switch_invalid(
        self, mock_get_models, base_session, available_modules
    ):
        """Test /model <name> with invalid model name."""
        mock_get_models.return_value = ["deepseek-r1:8b", "llama2:7b"]

        is_cmd, response, state_modifier = process_command(
            "/model nonexistent:1b", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "not found" in response
        assert "nonexistent:1b" in response

    @patch("tensortruth.app_utils.commands.get_ollama_models")
    @patch("tensortruth.app_utils.commands.st")
    def test_model_switch_when_ollama_unavailable(
        self, mock_st, mock_get_models, base_session, available_modules
    ):
        """Test /model <name> when Ollama service is down."""
        mock_get_models.side_effect = Exception("Connection refused")
        mock_st.session_state = MagicMock()
        mock_st.session_state.loaded_config = None

        is_cmd, response, state_modifier = process_command(
            "/model llama2:7b", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is not None  # Still attempts to switch
        assert "Could not verify" in response
        assert "Attempting to switch" in response


# ============================================================================
# Tests for /load command
# ============================================================================


@pytest.mark.unit
class TestLoadCommand:
    """Tests for /load command."""

    @patch("tensortruth.app_utils.commands.st")
    def test_load_valid_module(self, mock_st, base_session, available_modules):
        """Test /load with valid module name."""
        mock_st.session_state = MagicMock()
        mock_st.session_state.loaded_config = None
        base_session["modules"] = []  # Start with empty modules

        is_cmd, response, state_modifier = process_command(
            "/load pytorch", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is not None
        assert "Loaded" in response
        assert "pytorch" in response

        # Test state modifier applies the change
        state_modifier()
        assert "pytorch" in base_session["modules"]
        assert mock_st.session_state.loaded_config is None

    def test_load_invalid_module(self, base_session, available_modules):
        """Test /load with invalid module name."""
        is_cmd, response, state_modifier = process_command(
            "/load nonexistent", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "not found" in response
        assert "nonexistent" in response

    def test_load_already_active(self, base_session, available_modules):
        """Test /load with already active module."""
        is_cmd, response, state_modifier = process_command(
            "/load pytorch", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "active" in response.lower()

    def test_load_no_args(self, base_session, available_modules):
        """Test /load without arguments."""
        is_cmd, response, state_modifier = process_command(
            "/load", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "Usage" in response


# ============================================================================
# Tests for /unload command
# ============================================================================


@pytest.mark.unit
class TestUnloadCommand:
    """Tests for /unload command."""

    @patch("tensortruth.app_utils.commands.st")
    def test_unload_active_module(self, mock_st, base_session, available_modules):
        """Test /unload with active module."""
        mock_st.session_state = MagicMock()
        mock_st.session_state.loaded_config = None

        is_cmd, response, state_modifier = process_command(
            "/unload pytorch", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is not None
        assert "Unloaded" in response
        assert "pytorch" in response

        # Test state modifier applies the change
        state_modifier()
        assert "pytorch" not in base_session["modules"]
        assert mock_st.session_state.loaded_config is None

    def test_unload_inactive_module(self, base_session, available_modules):
        """Test /unload with inactive module."""
        is_cmd, response, state_modifier = process_command(
            "/unload tensorflow", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "not active" in response.lower()

    def test_unload_no_args(self, base_session, available_modules):
        """Test /unload without arguments."""
        is_cmd, response, state_modifier = process_command(
            "/unload", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "Usage" in response


# ============================================================================
# Tests for /reload command
# ============================================================================


@pytest.mark.unit
class TestReloadCommand:
    """Tests for /reload command."""

    @patch("tensortruth.app_utils.commands.free_memory")
    @patch("tensortruth.app_utils.commands.st")
    def test_reload_command(
        self, mock_st, mock_free_memory, base_session, available_modules
    ):
        """Test /reload command flushes memory."""
        mock_st.session_state = MagicMock()
        mock_st.session_state.loaded_config = "something"

        is_cmd, response, state_modifier = process_command(
            "/reload", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is not None
        assert "System Reload" in response
        assert "Memory flushed" in response

        # Test state modifier applies the change
        state_modifier()
        mock_free_memory.assert_called_once()
        assert mock_st.session_state.loaded_config is None


# ============================================================================
# Tests for /conf command
# ============================================================================


@pytest.mark.unit
class TestConfCommand:
    """Tests for /conf and /confidence commands."""

    @patch("tensortruth.app_utils.commands.st")
    def test_conf_valid_value(self, mock_st, base_session, available_modules):
        """Test /conf with valid confidence value."""
        mock_st.session_state = MagicMock()
        mock_st.session_state.loaded_config = None

        is_cmd, response, state_modifier = process_command(
            "/conf 0.5", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is not None
        assert "Confidence Warning" in response
        assert "0.5" in response

        # Test state modifier applies the change
        state_modifier()
        assert base_session["params"]["confidence_cutoff"] == 0.5
        assert mock_st.session_state.loaded_config is None

    def test_conf_invalid_value(self, base_session, available_modules):
        """Test /conf with invalid value."""
        is_cmd, response, state_modifier = process_command(
            "/conf 1.5", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "between 0.0 and 1.0" in response

    def test_conf_non_numeric(self, base_session, available_modules):
        """Test /conf with non-numeric value."""
        is_cmd, response, state_modifier = process_command(
            "/conf abc", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "Invalid number" in response

    def test_conf_no_args(self, base_session, available_modules):
        """Test /conf without arguments."""
        is_cmd, response, state_modifier = process_command(
            "/conf", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None

    @patch("tensortruth.app_utils.commands.st")
    def test_conf_with_hard_cutoff(self, mock_st, base_session, available_modules):
        """Test /conf with both warning and hard cutoff values."""
        mock_st.session_state = MagicMock()
        mock_st.session_state.loaded_config = None

        is_cmd, response, state_modifier = process_command(
            "/conf 0.3 0.15", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is not None
        assert "Confidence Warning" in response
        assert "0.3" in response
        assert "Confidence Cutoff (Hard)" in response
        assert "0.15" in response

        # Test state modifier applies both changes
        state_modifier()
        assert base_session["params"]["confidence_cutoff"] == 0.3
        assert base_session["params"]["confidence_cutoff_hard"] == 0.15
        assert mock_st.session_state.loaded_config is None

    def test_conf_hard_cutoff_greater_than_warning(
        self, base_session, available_modules
    ):
        """Test /conf with hard cutoff greater than warning threshold."""
        is_cmd, response, state_modifier = process_command(
            "/conf 0.2 0.5", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "must be <=" in response


# ============================================================================
# Tests for /device command
# ============================================================================


@pytest.mark.unit
class TestDeviceCommand:
    """Tests for /device command."""

    @patch("tensortruth.app_utils.commands.get_system_devices")
    @patch("tensortruth.app_utils.commands.st")
    def test_device_rag_valid(
        self, mock_st, mock_get_devices, base_session, available_modules
    ):
        """Test /device rag with valid device."""
        mock_get_devices.return_value = ["cpu", "cuda", "mps"]
        mock_st.session_state = MagicMock()
        mock_st.session_state.loaded_config = None

        is_cmd, response, state_modifier = process_command(
            "/device rag mps", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is not None
        assert "Pipeline Switched" in response
        assert "MPS" in response  # Check for uppercase MPS in response

        # Test state modifier applies the change
        state_modifier()
        assert base_session["params"]["rag_device"] == "mps"
        assert mock_st.session_state.loaded_config is None

    @patch("tensortruth.app_utils.commands.get_system_devices")
    def test_device_rag_invalid(
        self, mock_get_devices, base_session, available_modules
    ):
        """Test /device rag with invalid device."""
        mock_get_devices.return_value = ["cpu", "cuda"]

        is_cmd, response, state_modifier = process_command(
            "/device rag mps", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "not available" in response

    @patch("tensortruth.app_utils.commands.st")
    def test_device_llm_valid(self, mock_st, base_session, available_modules):
        """Test /device llm with valid device."""
        mock_st.session_state = MagicMock()
        mock_st.session_state.loaded_config = None

        is_cmd, response, state_modifier = process_command(
            "/device llm cpu", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is not None
        assert "LLM Switched" in response
        assert "CPU" in response

        # Test state modifier applies the change
        state_modifier()
        assert base_session["params"]["llm_device"] == "cpu"
        assert mock_st.session_state.loaded_config is None

    def test_device_llm_invalid(self, base_session, available_modules):
        """Test /device llm with invalid device."""
        is_cmd, response, state_modifier = process_command(
            "/device llm cuda", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "cpu" in response or "gpu" in response

    def test_device_unknown_target(self, base_session, available_modules):
        """Test /device with unknown target type."""
        is_cmd, response, state_modifier = process_command(
            "/device foo bar", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "Unknown target" in response

    def test_device_no_args(self, base_session, available_modules):
        """Test /device without arguments."""
        is_cmd, response, state_modifier = process_command(
            "/device", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "Usage" in response


# ============================================================================
# Tests for /search command
# ============================================================================


@pytest.mark.unit
class TestWebSearchCommand:
    """Tests for /search and /web commands."""

    @patch("tensortruth.utils.web_search.web_search")
    @patch("tensortruth.core.ollama.get_ollama_url")
    def test_search_command_basic(
        self, mock_get_url, mock_web_search, base_session, available_modules
    ):
        """Test /search command with basic query."""
        mock_get_url.return_value = "http://localhost:11434"
        mock_web_search.return_value = (
            '## Web Search: "PyTorch features"\n\n### Summary\nTest summary\n\n'
            "### Sources\n1. [Test](https://example.com)"
        )

        is_cmd, response, state_modifier = process_command(
            "/search PyTorch features", base_session, available_modules
        )

        # Websearch returns is_cmd=False to appear as assistant message
        assert is_cmd is False
        assert state_modifier is None
        assert "Web Search" in response
        assert "Test summary" in response
        assert "Sources" in response
        mock_web_search.assert_called_once()

        # Verify it uses session's model
        call_kwargs = mock_web_search.call_args[1]
        assert call_kwargs["model_name"] == "deepseek-r1:8b"

    @patch("tensortruth.utils.web_search.web_search")
    @patch("tensortruth.core.ollama.get_ollama_url")
    def test_search_web_alias(
        self, mock_get_url, mock_web_search, base_session, available_modules
    ):
        """Test /web alias for /search command."""
        mock_get_url.return_value = "http://localhost:11434"
        mock_web_search.return_value = "## Web Search Results"

        is_cmd, response, state_modifier = process_command(
            "/web test query", base_session, available_modules
        )

        # Websearch returns is_cmd=False to appear as assistant message
        assert is_cmd is False
        assert "Web Search" in response
        mock_web_search.assert_called_once()

    def test_search_no_args(self, base_session, available_modules):
        """Test /search without arguments."""
        is_cmd, response, state_modifier = process_command(
            "/search", base_session, available_modules
        )

        # Error case returns is_cmd=True (not added to history)
        assert is_cmd is True
        assert state_modifier is None
        assert "Usage" in response
        assert "/search <query>" in response

    @patch("tensortruth.utils.web_search.web_search")
    @patch("tensortruth.core.ollama.get_ollama_url")
    def test_search_multi_word_query(
        self, mock_get_url, mock_web_search, base_session, available_modules
    ):
        """Test /search with multi-word query."""
        mock_get_url.return_value = "http://localhost:11434"
        mock_web_search.return_value = "## Results"

        is_cmd, response, state_modifier = process_command(
            "/search how to use PyTorch 2.9", base_session, available_modules
        )

        assert is_cmd is False
        # Verify query is properly joined
        call_kwargs = mock_web_search.call_args[1]
        assert call_kwargs["query"] == "how to use PyTorch 2.9"

    @patch("tensortruth.utils.web_search.web_search")
    @patch("tensortruth.core.ollama.get_ollama_url")
    def test_search_uses_session_config(
        self, mock_get_url, mock_web_search, base_session, available_modules
    ):
        """Test /search uses session configuration parameters."""
        mock_get_url.return_value = "http://localhost:11434"
        mock_web_search.return_value = "## Results"

        # Add web search config to session
        base_session["params"]["web_search_max_results"] = 10
        base_session["params"]["web_search_pages_to_fetch"] = 7

        is_cmd, response, state_modifier = process_command(
            "/search test", base_session, available_modules
        )

        assert is_cmd is False
        call_kwargs = mock_web_search.call_args[1]
        assert call_kwargs["max_results"] == 10
        assert call_kwargs["max_pages"] == 7

    @patch("tensortruth.utils.web_search.web_search")
    @patch("tensortruth.core.ollama.get_ollama_url")
    def test_search_uses_defaults_when_no_config(
        self, mock_get_url, mock_web_search, base_session, available_modules
    ):
        """Test /search uses default values when not in session config."""
        mock_get_url.return_value = "http://localhost:11434"
        mock_web_search.return_value = "## Results"

        is_cmd, response, state_modifier = process_command(
            "/search test", base_session, available_modules
        )

        assert is_cmd is False
        call_kwargs = mock_web_search.call_args[1]
        assert call_kwargs["max_results"] == 10  # Default
        assert call_kwargs["max_pages"] == 5  # Default

    @patch("tensortruth.utils.web_search.web_search")
    @patch("tensortruth.core.ollama.get_ollama_url")
    def test_search_handles_error(
        self, mock_get_url, mock_web_search, base_session, available_modules
    ):
        """Test /search handles errors gracefully."""
        mock_get_url.return_value = "http://localhost:11434"
        mock_web_search.side_effect = Exception("Network error")

        is_cmd, response, state_modifier = process_command(
            "/search test", base_session, available_modules
        )

        # Errors return is_cmd=True (not added to history)
        assert is_cmd is True
        assert state_modifier is None
        assert "Web search failed" in response
        assert "Network error" in response

    @patch("tensortruth.utils.web_search.web_search")
    @patch("tensortruth.core.ollama.get_ollama_url")
    def test_search_reuses_session_model(
        self, mock_get_url, mock_web_search, base_session, available_modules
    ):
        """Test /search uses the current session model (reuses VRAM)."""
        mock_get_url.return_value = "http://localhost:11434"
        mock_web_search.return_value = "## Results"

        # Change model in session
        base_session["params"]["model"] = "llama2:7b"

        is_cmd, response, state_modifier = process_command(
            "/search test", base_session, available_modules
        )

        assert is_cmd is False
        call_kwargs = mock_web_search.call_args[1]
        assert call_kwargs["model_name"] == "llama2:7b"
        assert call_kwargs["ollama_url"] == "http://localhost:11434"


# ============================================================================
# Tests for unknown commands
# ============================================================================


@pytest.mark.unit
class TestUnknownCommand:
    """Tests for unknown command handling."""

    def test_unknown_command(self, base_session, available_modules):
        """Test unknown command shows error and help."""
        is_cmd, response, state_modifier = process_command(
            "/xyz", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "Unknown command" in response
        assert "/xyz" in response
        assert "Available Commands" in response
        assert "/list" in response
        assert "/model" in response

    def test_typo_command(self, base_session, available_modules):
        """Test typo in command name."""
        is_cmd, response, state_modifier = process_command(
            "/laod pytorch", base_session, available_modules
        )

        assert is_cmd is True
        assert state_modifier is None
        assert "Unknown command" in response
        assert "/laod" in response
        assert "Available Commands" in response
