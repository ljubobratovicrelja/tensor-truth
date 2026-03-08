"""REST endpoints for MCP server management."""

import logging

from fastapi import APIRouter, HTTPException

from tensortruth.api.schemas.mcp_server import (
    MCPServerCreateRequest,
    MCPServerListResponse,
    MCPServerResponse,
    MCPServerToggleRequest,
    MCPServerUpdateRequest,
)
from tensortruth.services.mcp_server_service import MCPServerService

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_service() -> MCPServerService:
    return MCPServerService()


def _raise_for_value_error(e: ValueError) -> None:
    """Map ValueError to appropriate HTTP status code."""
    msg = str(e)
    if "not found" in msg:
        raise HTTPException(status_code=404, detail=msg)
    if "built-in" in msg:
        raise HTTPException(status_code=403, detail=msg)
    raise HTTPException(status_code=400, detail=msg)


@router.get("/", response_model=MCPServerListResponse)
async def list_mcp_servers():
    """List all MCP servers (built-in + user-configured)."""
    service = _get_service()
    servers = service.list_all()
    return {"servers": servers}


@router.get("/presets")
async def get_presets():
    """Get available server presets."""
    return {"presets": MCPServerService.get_presets()}


@router.post("/", response_model=MCPServerResponse, status_code=201)
async def add_mcp_server(request: MCPServerCreateRequest):
    """Add a new MCP server configuration."""
    service = _get_service()
    try:
        server = service.add(request.model_dump())
        return server
    except ValueError as e:
        _raise_for_value_error(e)


@router.patch("/{name}", response_model=MCPServerResponse)
async def update_mcp_server(name: str, request: MCPServerUpdateRequest):
    """Update an MCP server configuration (partial update)."""
    service = _get_service()
    try:
        server = service.update(name, request.model_dump(exclude_none=True))
        return server
    except ValueError as e:
        _raise_for_value_error(e)


@router.delete("/{name}", status_code=204)
async def delete_mcp_server(name: str):
    """Remove an MCP server configuration."""
    service = _get_service()
    try:
        service.remove(name)
    except ValueError as e:
        _raise_for_value_error(e)


@router.patch("/{name}/toggle", response_model=MCPServerResponse)
async def toggle_mcp_server(name: str, request: MCPServerToggleRequest):
    """Toggle an MCP server's enabled state."""
    service = _get_service()
    try:
        server = service.toggle(name, request.enabled)
        return server
    except ValueError as e:
        _raise_for_value_error(e)
