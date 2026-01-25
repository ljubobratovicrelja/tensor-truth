"""Factory for creating BrowseAgent instances."""

import logging
from typing import Any, Dict, Sequence

from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.base import Agent
from tensortruth.agents.config import AgentConfig
from tensortruth.agents.factory import register_agent_factory
from tensortruth.agents.router.browse.agent import BrowseAgent

logger = logging.getLogger(__name__)


def create_browse_agent(
    config: AgentConfig,
    tools: Sequence[FunctionTool],
    synthesis_llm: Ollama,
    session_params: Dict[str, Any],
) -> Agent:
    """Factory for BrowseAgent instances.

    Args:
        config: Agent configuration
        tools: Tools (must include search_web, fetch_pages_batch)
        synthesis_llm: Session model for synthesis
        session_params: router_model, min_pages_required, context_window,
                       ollama_url, reranker_model, rag_device

    Returns:
        BrowseAgent instance

    Raises:
        ValueError: If required tools are missing
    """
    # Validate tools
    tool_names = {t.metadata.name for t in tools}
    required = {"search_web", "fetch_pages_batch"}
    missing = required - tool_names
    if missing:
        raise ValueError(f"BrowseAgent requires tools: {required}. Missing: {missing}")

    # Convert to dict
    tools_dict = {t.metadata.name: t for t in tools if t.metadata.name}

    # Extract parameters
    router_model = session_params.get("router_model", "llama3.2:3b")
    min_pages_required = session_params.get("min_pages_required", 5)
    context_window = session_params.get("context_window", 16384)
    ollama_url = session_params.get("ollama_url", "http://localhost:11434")
    reranker_model = session_params.get("reranker_model")
    rag_device = session_params.get("rag_device", "cpu")

    # Merge factory_params from config
    min_pages_required = config.factory_params.get(
        "min_pages_required", min_pages_required
    )

    logger.info(
        f"Creating BrowseAgent: router_model={router_model}, "
        f"min_pages={min_pages_required}, context_window={context_window}"
    )

    # Create router LLM (fast small model)
    router_llm = Ollama(
        model=router_model,
        base_url=ollama_url,
        temperature=0.2,
        context_window=context_window,
        additional_kwargs={"num_ctx": context_window},
        request_timeout=120.0,
    )

    # Create BrowseAgent
    return BrowseAgent(
        router_llm=router_llm,
        synthesis_llm=synthesis_llm,
        tools=tools_dict,
        min_pages_required=min_pages_required,
        max_iterations=config.max_iterations,
        reranker_model=reranker_model,
        rag_device=rag_device,
        context_window=context_window,
    )


# Self-registration on import
register_agent_factory("router", create_browse_agent)
logger.info("Registered BrowseAgent factory for agent_type='router'")
