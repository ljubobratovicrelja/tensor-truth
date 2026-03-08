"""Pydantic schemas for MCP server API."""

from typing import Literal, Optional

from pydantic import BaseModel


class MCPServerResponse(BaseModel):
    """Response schema for an MCP server."""

    name: str
    type: str
    command: Optional[str] = None
    args: list[str] = []
    url: Optional[str] = None
    description: Optional[str] = None
    env: Optional[dict[str, str]] = None
    enabled: bool = True
    builtin: bool = False
    env_status: dict[str, bool] = {}


class MCPServerCreateRequest(BaseModel):
    """Request schema for creating an MCP server."""

    name: str
    type: Literal["stdio", "sse"]
    command: Optional[str] = None
    args: list[str] = []
    url: Optional[str] = None
    description: Optional[str] = None
    env: Optional[dict[str, str]] = None
    enabled: bool = True


class MCPServerUpdateRequest(BaseModel):
    """Request schema for updating an MCP server."""

    type: Optional[Literal["stdio", "sse"]] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    url: Optional[str] = None
    description: Optional[str] = None
    env: Optional[dict[str, str]] = None
    enabled: Optional[bool] = None


class MCPServerToggleRequest(BaseModel):
    """Request schema for toggling an MCP server."""

    enabled: bool


class MCPServerListResponse(BaseModel):
    """Response schema for listing MCP servers."""

    servers: list[MCPServerResponse]
