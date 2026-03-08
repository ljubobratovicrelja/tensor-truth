"""Pydantic schemas for extensions API."""

from typing import Literal, Optional

from pydantic import BaseModel


class ExtensionResponse(BaseModel):
    """Response schema for an installed extension."""

    name: str
    type: str  # "command" | "agent"
    description: str
    filename: str
    requires_mcp: Optional[str] = None
    mcp_available: bool = True


class ExtensionListResponse(BaseModel):
    """Response schema for listing extensions."""

    extensions: list[ExtensionResponse]


class LibraryExtensionResponse(BaseModel):
    """Response schema for a library extension."""

    name: str
    type: str  # "command" | "agent"
    description: str
    filename: str
    requires_mcp: Optional[str] = None
    mcp_available: bool = True
    installed: bool = False


class ExtensionLibraryResponse(BaseModel):
    """Response schema for the extension library."""

    extensions: list[LibraryExtensionResponse]


class ExtensionInstallRequest(BaseModel):
    """Request schema for installing extensions."""

    type: Literal["command", "agent"]
    filename: str


class ExtensionInstallResponse(BaseModel):
    """Response schema for installing an extension."""

    installed: list[str]
    errors: list[str] = []
