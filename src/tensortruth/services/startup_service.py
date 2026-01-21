"""Startup service - handles application initialization and resource checks."""

import logging
from typing import Dict, List

from tensortruth.app_utils.paths import (
    get_indexes_dir,
    get_sessions_data_dir,
    get_user_data_dir,
)
from tensortruth.core.ollama import get_available_models
from tensortruth.services.config_service import ConfigService

logger = logging.getLogger(__name__)


class StartupService:
    """Service for application startup and initialization.

    Handles:
    - Directory creation
    - Configuration initialization
    - Resource availability checks (indexes, Ollama models)
    """

    def __init__(self, config_service: ConfigService):
        """Initialize startup service.

        Args:
            config_service: ConfigService instance for config management.
        """
        self.config_service = config_service

    def initialize_directories(self, log: bool = True) -> bool:
        """Create all required directories for TensorTruth.

        Creates:
        - ~/.tensortruth/
        - ~/.tensortruth/indexes/
        - ~/.tensortruth/sessions/

        Args:
            log: If True, log the result (use for actual startup, not status checks).

        Returns:
            True if all directories exist (created or already present).
        """
        try:
            # These functions create directories if they don't exist
            get_user_data_dir()
            get_indexes_dir()
            get_sessions_data_dir()

            if log:
                logger.info("✓ All required directories initialized")
            return True

        except Exception as e:
            if log:
                logger.error(f"Failed to initialize directories: {e}")
            return False

    def initialize_config(self, log: bool = True) -> bool:
        """Initialize configuration file.

        Creates default config.yaml if it doesn't exist.

        Args:
            log: If True, log the result (use for actual startup, not status checks).

        Returns:
            True if config loaded successfully.
        """
        try:
            config = self.config_service.load()
            if log:
                logger.info(
                    f"✓ Configuration loaded (device: {config.rag.default_device})"
                )
            return True

        except Exception as e:
            if log:
                logger.error(f"Failed to initialize config: {e}")
            return False

    def check_indexes(self) -> Dict[str, bool]:
        """Check if vector indexes exist and have content.

        Returns:
            Dict with keys:
            - exists: True if indexes directory exists
            - has_content: True if indexes directory contains valid ChromaDB databases
        """
        indexes_dir = get_indexes_dir()
        exists = indexes_dir.exists()
        has_content = False

        if exists:
            # Check for valid ChromaDB indexes (chroma.sqlite3 files)
            subdirs = [path for path in indexes_dir.iterdir() if path.is_dir()]
            has_content = any((path / "chroma.sqlite3").exists() for path in subdirs)

        logger.debug(
            f"Indexes check: exists={exists}, has_content={has_content}, "
            f"path={indexes_dir}"
        )

        return {"exists": exists, "has_content": has_content}

    def check_ollama_models(self) -> Dict[str, List[str]]:
        """Check if required Ollama models are available.

        Returns:
            Dict with keys:
            - required: List of required model names
            - available: List of all available models
            - missing: List of missing required models
        """
        try:
            config = self.config_service.load()

            # Get unique required models (remove duplicates)
            required_models = list(
                dict.fromkeys(
                    [
                        config.models.default_rag_model,
                        config.models.default_fallback_model,
                        config.models.default_agent_reasoning_model,
                    ]
                )
            )

            available_models = get_available_models()
            missing_models = [m for m in required_models if m not in available_models]

            logger.debug(
                f"Ollama models check: required={len(required_models)}, "
                f"available={len(available_models)}, missing={len(missing_models)}"
            )

            return {
                "required": required_models,
                "available": available_models,
                "missing": missing_models,
            }

        except Exception as e:
            logger.warning(f"Failed to check Ollama models: {e}")
            return {
                "required": [],
                "available": [],
                "missing": [],
            }

    def check_startup_status(self, log: bool = False) -> Dict:
        """Perform all startup checks and build comprehensive status.

        Args:
            log: If True, log the status summary (use for actual startup, not status checks).

        Returns:
            Dict with keys:
            - directories_ok: bool
            - config_ok: bool
            - indexes_ok: bool (has content)
            - models_ok: bool (no missing models)
            - indexes_status: dict from check_indexes()
            - models_status: dict from check_ollama_models()
            - ready: bool (app can run, but may have warnings)
            - warnings: list of warning strings
        """
        directories_ok = self.initialize_directories(log=log)
        config_ok = self.initialize_config(log=log)

        indexes_status = self.check_indexes()
        models_status = self.check_ollama_models()

        indexes_ok = indexes_status["has_content"]
        models_ok = len(models_status["missing"]) == 0

        # Build warnings list
        warnings = []

        if not indexes_ok:
            if not indexes_status["exists"]:
                warnings.append(
                    "Indexes directory does not exist. "
                    "Download indexes to enable knowledge base queries."
                )
            else:
                warnings.append(
                    "Knowledge base indexes are missing or empty. "
                    "Download indexes from HuggingFace Hub to enable RAG queries."
                )

        if not models_ok:
            missing_list = ", ".join(models_status["missing"])
            warnings.append(
                f"Missing Ollama models: {missing_list}. "
                f"Pull these models to enable all features."
            )

        # App is "ready" if critical infrastructure is ok (directories + config)
        # Indexes and models are optional (can work without them)
        ready = directories_ok and config_ok

        status = {
            "directories_ok": directories_ok,
            "config_ok": config_ok,
            "indexes_ok": indexes_ok,
            "models_ok": models_ok,
            "indexes_status": indexes_status,
            "models_status": models_status,
            "ready": ready,
            "warnings": warnings,
        }

        # Log summary only if requested (for actual startup, not status polling)
        if log:
            if ready:
                if warnings:
                    logger.warning(
                        f"⚠️  Startup complete with {len(warnings)} warnings"
                    )
                    for warning in warnings:
                        logger.warning(f"  - {warning}")
                else:
                    logger.info("✓ Startup checks passed - all resources available")
            else:
                logger.error(
                    "✗ Startup checks failed - critical infrastructure missing"
                )

        return status
