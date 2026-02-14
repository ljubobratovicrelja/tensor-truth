"""Unit tests for project config inheritance (3-tier merge).

Verifies: global defaults < project config < user params
"""

from unittest.mock import MagicMock

import pytest

from tensortruth.services.models import SessionData
from tensortruth.services.session_service import SessionService


@pytest.fixture
def session_service(tmp_path):
    """Create a SessionService with temp paths."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return SessionService(
        sessions_file=tmp_path / "chat_sessions.json",
        sessions_dir=sessions_dir,
    )


@pytest.fixture
def mock_config_service():
    """Create a mock ConfigService that returns known defaults."""
    mock = MagicMock()
    config = MagicMock()

    # Set up config defaults
    config.ui.default_temperature = 0.7
    config.ui.default_context_window = 4096
    config.ui.default_max_tokens = 2048
    config.ui.default_top_n = 5
    config.ui.default_confidence_threshold = 0.3
    config.ui.default_confidence_cutoff_hard = 0.1
    config.rag.default_reranker = "global-reranker"
    config.rag.default_device = "cpu"
    config.rag.default_balance_strategy = "balanced"
    config.rag.default_embedding_model = "BAAI/bge-m3"
    config.agent.router_model = "global-router"

    mock.load.return_value = config
    return mock


class TestConfigInheritance:
    """Test 3-tier config merge: global defaults < project config < user params."""

    def test_global_defaults_applied_when_no_overrides(
        self, session_service, mock_config_service
    ):
        """Session gets global defaults when no project/user overrides."""
        data = SessionData(current_id=None, sessions={})

        new_id, new_data = session_service.create(
            modules=None,
            params={},
            data=data,
            config_service=mock_config_service,
        )

        session = new_data.sessions[new_id]
        assert session["params"]["temperature"] == 0.7
        assert session["params"]["context_window"] == 4096
        assert session["params"]["reranker_model"] == "global-reranker"

    def test_project_config_overrides_global_defaults(
        self, session_service, mock_config_service
    ):
        """Project config overrides global defaults.

        Simulates: merged_params = {**project_config, **user_params}
        passed to session_service.create() as params.
        """
        data = SessionData(current_id=None, sessions={})

        # Project config overrides temperature and reranker
        project_config = {"temperature": 0.3, "reranker_model": "project-reranker"}
        merged_params = {**project_config}  # No user params

        new_id, new_data = session_service.create(
            modules=None,
            params=merged_params,
            data=data,
            config_service=mock_config_service,
        )

        session = new_data.sessions[new_id]
        # Project config wins over global defaults
        assert session["params"]["temperature"] == 0.3
        assert session["params"]["reranker_model"] == "project-reranker"
        # Global defaults fill in non-overridden keys
        assert session["params"]["context_window"] == 4096

    def test_user_params_override_project_config(
        self, session_service, mock_config_service
    ):
        """User params override project config which overrides global defaults."""
        data = SessionData(current_id=None, sessions={})

        project_config = {"temperature": 0.3, "reranker_model": "project-reranker"}
        user_params = {"temperature": 0.9}  # User overrides project temperature
        merged_params = {**project_config, **user_params}

        new_id, new_data = session_service.create(
            modules=None,
            params=merged_params,
            data=data,
            config_service=mock_config_service,
        )

        session = new_data.sessions[new_id]
        # User param wins
        assert session["params"]["temperature"] == 0.9
        # Project config wins over global defaults
        assert session["params"]["reranker_model"] == "project-reranker"
        # Global defaults fill in the rest
        assert session["params"]["context_window"] == 4096

    def test_non_overlapping_keys_all_present(
        self, session_service, mock_config_service
    ):
        """Non-overlapping keys from all tiers are all present."""
        data = SessionData(current_id=None, sessions={})

        project_config = {"custom_project_key": "project_value"}
        user_params = {"custom_user_key": "user_value"}
        merged_params = {**project_config, **user_params}

        new_id, new_data = session_service.create(
            modules=None,
            params=merged_params,
            data=data,
            config_service=mock_config_service,
        )

        session = new_data.sessions[new_id]
        # All three tiers present
        assert session["params"]["temperature"] == 0.7  # global default
        assert session["params"]["custom_project_key"] == "project_value"
        assert session["params"]["custom_user_key"] == "user_value"
