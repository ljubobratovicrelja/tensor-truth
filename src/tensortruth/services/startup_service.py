"""Startup service - handles application initialization and resource checks."""

import logging
from typing import Any, Dict, List, Optional

from tensortruth.app_utils.paths import (
    get_indexes_dir,
    get_sessions_data_dir,
    get_user_data_dir,
)
from tensortruth.core.ollama import get_available_models
from tensortruth.indexing.metadata import (
    get_available_embedding_models,
    sanitize_model_id,
)
from tensortruth.indexing.migration import check_and_migrate_on_startup
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

    def check_indexes(self) -> Dict[str, Any]:
        """Check if vector indexes exist and have content.

        Returns:
            Dict with keys:
            - exists: True if indexes directory exists
            - has_content: True if indexes directory contains valid ChromaDB databases
            - available_models: List of embedding models that have indexes
            - migration_status: Dict with legacy index migration status
        """
        indexes_dir = get_indexes_dir()
        exists = indexes_dir.exists()
        has_content = False
        available_models: List[Dict[str, Any]] = []

        if exists:
            # First, check for and migrate any legacy indexes
            check_and_migrate_on_startup(indexes_dir)

            # Get available embedding models with their indexes
            available_models = get_available_embedding_models(indexes_dir)
            has_content = len(available_models) > 0

        logger.debug(
            f"Indexes check: exists={exists}, has_content={has_content}, "
            f"models={[m['model_id'] for m in available_models]}, "
            f"path={indexes_dir}"
        )

        return {
            "exists": exists,
            "has_content": has_content,
            "available_models": available_models,
        }

    def check_embedding_model_mismatch(self) -> Optional[Dict[str, Any]]:
        """Detect when config embedding model differs from available indexes.

        Returns:
            None if no mismatch, otherwise dict with:
            - config_model: The embedding model set in config
            - config_model_id: Sanitized model ID from config
            - available_model_ids: List of model IDs with available indexes
            - message: Human-readable warning message
        """
        try:
            config = self.config_service.load()
            config_model = config.rag.default_embedding_model
            config_model_id = sanitize_model_id(config_model)

            indexes_dir = get_indexes_dir()
            available_models = get_available_embedding_models(indexes_dir)
            available_model_ids = [m["model_id"] for m in available_models]

            # No mismatch if:
            # 1. No indexes at all (fresh install - not a mismatch)
            # 2. Config model has indexes available
            if not available_model_ids or config_model_id in available_model_ids:
                return None

            message = (
                f"Configured embedding model '{config_model}' has no local indexes. "
                f"Available: {', '.join(available_model_ids)}. "
                f"Either download indexes for '{config_model}' or change your config."
            )

            logger.warning(f"Embedding model mismatch: {message}")

            return {
                "config_model": config_model,
                "config_model_id": config_model_id,
                "available_model_ids": available_model_ids,
                "message": message,
            }

        except Exception as e:
            logger.warning(f"Failed to check embedding model mismatch: {e}")
            return None

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
                        config.llm.default_model,
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
            - embedding_mismatch: dict or None (from check_embedding_model_mismatch)
            - ready: bool (app can run, but may have warnings)
            - warnings: list of warning strings
        """
        directories_ok = self.initialize_directories(log=log)
        config_ok = self.initialize_config(log=log)

        indexes_status = self.check_indexes()
        models_status = self.check_ollama_models()
        embedding_mismatch = self.check_embedding_model_mismatch()

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

        # Add embedding model mismatch warning if applicable
        if embedding_mismatch:
            warnings.append(embedding_mismatch["message"])

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
            "embedding_mismatch": embedding_mismatch,
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
