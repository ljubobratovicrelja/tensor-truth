"""API routes for tools and agents.

Provides endpoints for listing available tools and agents.
"""

from typing import Any, Dict, List

from fastapi import APIRouter

from tensortruth.api.deps import AgentServiceDep, ToolServiceDep

router = APIRouter(tags=["tools"])


@router.get("/tools")
async def list_tools(tool_service: ToolServiceDep) -> Dict[str, List[Dict[str, Any]]]:
    """List all available tools.

    Returns metadata for all tools loaded from MCP servers.

    Returns:
        Dictionary with "tools" key containing list of tool metadata.
    """
    return {"tools": tool_service.list_tools()}


@router.get("/agents")
async def list_agents(
    agent_service: AgentServiceDep,
) -> Dict[str, List[Dict[str, Any]]]:
    """List all available agents.

    Returns metadata for all registered agents.

    Returns:
        Dictionary with "agents" key containing list of agent metadata.
    """
    return {"agents": agent_service.list_agents()}
