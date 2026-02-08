"""Factory for creating FunctionAgent instances."""

import logging
from typing import Any, Dict, Sequence

from llama_index.core.agent.workflow.function_agent import (
    FunctionAgent as LIFunctionAgent,
)
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

from tensortruth.agents.base import Agent
from tensortruth.agents.config import AgentConfig
from tensortruth.agents.factory import register_agent_factory
from tensortruth.agents.function.wrapper import FunctionAgentWrapper

logger = logging.getLogger(__name__)


def create_function_agent(
    config: AgentConfig,
    tools: Sequence[FunctionTool],
    llm: Ollama,
    session_params: Dict[str, Any],
) -> Agent:
    """Factory for creating FunctionAgent instances.

    Args:
        config: Agent configuration
        tools: List of tools for the agent
        llm: Ollama LLM instance
        session_params: Session parameters (not used by FunctionAgent)

    Returns:
        FunctionAgentWrapper instance
    """
    logger.info(
        f"Creating FunctionAgent: name={config.name}, tools={config.tools}, "
        f"model={llm.model}"
    )

    # Enhance system prompt with explicit tool list to prevent hallucination.
    # Small models may "know" about tools from training and try to call them
    # even when they aren't provided in the function-calling schema.
    base_prompt = config.system_prompt or "You are a helpful assistant."
    tool_names = [t.metadata.name for t in tools]
    tool_list_str = ", ".join(tool_names)
    enhanced_prompt = (
        f"{base_prompt}\n\n"
        f"You have access to ONLY these tools: {tool_list_str}. "
        f"Do NOT call any tool not in this list."
    )

    # Create LlamaIndex FunctionAgent
    function_agent = LIFunctionAgent(
        tools=list(tools),
        llm=llm,
        system_prompt=enhanced_prompt,
    )

    # Wrap in our Agent interface
    return FunctionAgentWrapper(function_agent=function_agent, agent_name=config.name)


# Self-registration on import
register_agent_factory("function", create_function_agent)
logger.info("Registered FunctionAgent factory for agent_type='function'")
