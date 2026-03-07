"""Startup service - handles application initialization and resource checks."""

import logging
from typing import Any, Dict, List, Optional

from tensortruth.app_utils.paths import (
    get_indexes_dir,
    get_sessions_data_dir,
    get_user_data_dir,
)
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

    def check_ollama_models(self) -> Dict[str, Any]:
        """Check Ollama reachability and available models.

        Also performs a basic connectivity check for openai_compatible providers.

        Returns:
            Dict with keys:
            - ollama_running: True if Ollama is reachable
            - available: List of all available model names
            - models_ok: True if ANY provider has reachable models
            - providers_ok: True if any provider is reachable
        """
        ollama_running = False
        available_models: List[str] = []
        providers_ok = False

        try:
            import requests as _requests

            from tensortruth.core.providers import ProviderRegistry

            registry = ProviderRegistry.get_instance()

            for provider in registry.providers:
                if provider.type == "ollama":
                    try:
                        base = provider.base_url.rstrip("/")
                        response = _requests.get(f"{base}/api/tags", timeout=2)
                        if response.status_code == 200:
                            ollama_running = True
                            providers_ok = True
                            data = response.json()
                            available_models.extend(
                                sorted(
                                    m["name"] for m in data.get("models", [])
                                )
                            )
                    except Exception:
                        pass

                elif provider.type == "llama_cpp":
                    try:
                        from tensortruth.core.llama_cpp import (
                            check_health,
                            get_available_models as get_llama_models,
                        )

                        base = provider.base_url.rstrip("/")
                        if check_health(base):
                            providers_ok = True
                            server_models = get_llama_models(base)
                            available_models.extend(
                                m["id"] for m in server_models if m.get("id")
                            )
                    except Exception:
                        pass

                elif provider.type == "openai_compatible":
                    # Basic connectivity check
                    try:
                        base = provider.base_url.rstrip("/")
                        headers: Dict[str, str] = {}
                        if provider.api_key:
                            headers["Authorization"] = f"Bearer {provider.api_key}"
                        response = _requests.get(
                            f"{base}/models", headers=headers, timeout=3
                        )
                        if response.status_code == 200:
                            providers_ok = True
                            # Count statically configured models as available
                            for m in provider.models:
                                name = m.get("name", "")
                                if name:
                                    available_models.append(name)
                    except Exception:
                        # Even without connectivity, statically configured models
                        # are "available" in config
                        if provider.models:
                            providers_ok = True
                            for m in provider.models:
                                name = m.get("name", "")
                                if name:
                                    available_models.append(name)

        except Exception as e:
            logger.warning(f"Failed to check providers: {e}")

        models_ok = len(available_models) > 0

        logger.debug(
            f"Provider check: ollama_running={ollama_running}, "
            f"available={len(available_models)}, models_ok={models_ok}, "
            f"providers_ok={providers_ok}"
        )

        return {
            "ollama_running": ollama_running,
            "available": available_models,
            "models_ok": models_ok,
            "providers_ok": providers_ok,
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
        models_ok = models_status["models_ok"]

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

        if not models_status["ollama_running"]:
            warnings.append(
                "Ollama is not running or not reachable. "
                "Start Ollama to enable LLM features."
            )
        elif not models_ok:
            warnings.append(
                "No Ollama models found. "
                "Pull a model (e.g. `ollama pull llama3.2`) to get started."
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
            "ollama_running": models_status["ollama_running"],
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
