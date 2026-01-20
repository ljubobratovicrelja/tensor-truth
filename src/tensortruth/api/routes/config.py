"""Configuration endpoints."""

from fastapi import APIRouter

from tensortruth.api.deps import ConfigServiceDep
from tensortruth.api.schemas import (
    AgentConfigSchema,
    ConfigResponse,
    ConfigUpdateRequest,
    ModelsConfigSchema,
    OllamaConfigSchema,
    RAGConfigSchema,
    UIConfigSchema,
)

router = APIRouter()


def _config_to_response(config) -> ConfigResponse:
    """Convert TensorTruthConfig to response schema."""
    return ConfigResponse(
        ollama=OllamaConfigSchema(
            base_url=config.ollama.base_url,
            timeout=config.ollama.timeout,
        ),
        ui=UIConfigSchema(
            default_temperature=config.ui.default_temperature,
            default_context_window=config.ui.default_context_window,
            default_max_tokens=config.ui.default_max_tokens,
            default_reranker=config.ui.default_reranker,
            default_top_n=config.ui.default_top_n,
            default_confidence_threshold=config.ui.default_confidence_threshold,
            default_confidence_cutoff_hard=config.ui.default_confidence_cutoff_hard,
        ),
        rag=RAGConfigSchema(
            default_device=config.rag.default_device,
        ),
        models=ModelsConfigSchema(
            default_rag_model=config.models.default_rag_model,
            default_fallback_model=config.models.default_fallback_model,
            default_agent_reasoning_model=config.models.default_agent_reasoning_model,
        ),
        agent=AgentConfigSchema(
            max_iterations=config.agent.max_iterations,
            min_pages_required=config.agent.min_pages_required,
            reasoning_model=config.agent.reasoning_model,
            enable_natural_language_agents=config.agent.enable_natural_language_agents,
            intent_classifier_model=config.agent.intent_classifier_model,
        ),
    )


@router.get("", response_model=ConfigResponse)
async def get_config(config_service: ConfigServiceDep) -> ConfigResponse:
    """Get current configuration."""
    config = config_service.load()
    return _config_to_response(config)


@router.patch("", response_model=ConfigResponse)
async def update_config(
    body: ConfigUpdateRequest, config_service: ConfigServiceDep
) -> ConfigResponse:
    """Update configuration values.

    Supports nested updates using prefixed keys:
    - ollama_*: Updates ollama config (e.g., ollama_base_url)
    - ui_*: Updates UI config (e.g., ui_default_temperature)
    - rag_*: Updates RAG config (e.g., rag_default_device)
    - agent_*: Updates agent config (e.g., agent_max_iterations)
    - models_*: Updates models config (e.g., models_default_rag_model)
    """
    config = config_service.update(**body.updates)
    return _config_to_response(config)


@router.get("/defaults", response_model=ConfigResponse)
async def get_default_config() -> ConfigResponse:
    """Get default configuration values."""
    from tensortruth.app_utils.config_schema import TensorTruthConfig

    config = TensorTruthConfig.create_default()
    return _config_to_response(config)


@router.get("/devices")
async def get_available_devices() -> dict:
    """Get list of available compute devices for this system."""
    from tensortruth.app_utils.helpers import get_system_devices

    devices = get_system_devices()
    return {"devices": devices}
