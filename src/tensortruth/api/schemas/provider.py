"""Provider management schemas."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ProviderResponse(BaseModel):
    """Single provider with live status."""

    id: str = Field(..., description="Provider identifier")
    type: str = Field(
        ..., description="Provider type (ollama, openai_compatible, llama_cpp)"
    )
    base_url: str = Field(..., description="Base URL for the provider")
    api_key: str = Field("", description="Masked API key (*** if set, empty if not)")
    timeout: int = Field(300, description="Request timeout in seconds")
    models: List[Dict[str, Any]] = Field(
        default_factory=list, description="Static model configs"
    )
    status: str = Field(
        "unknown", description="Connection status (connected, unreachable)"
    )
    model_count: int = Field(0, description="Number of available models")


class ProviderListResponse(BaseModel):
    """List of configured providers."""

    providers: List[ProviderResponse]


class ProviderCreateRequest(BaseModel):
    """Request to add a new provider."""

    id: str = Field(
        ...,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$",
        description="Unique provider identifier",
    )
    type: Literal["ollama", "openai_compatible", "llama_cpp"] = Field(
        ..., description="Provider type (ollama, openai_compatible, llama_cpp)"
    )
    base_url: str = Field(..., min_length=1, description="Base URL for the provider")
    api_key: str = Field("", description="API key (optional)")
    timeout: Optional[int] = Field(
        default=None, ge=1, description="Request timeout in seconds"
    )
    models: List[Dict[str, Any]] = Field(
        default_factory=list, description="Static model configs"
    )


class ProviderUpdateRequest(BaseModel):
    """Request to update an existing provider (all fields optional)."""

    base_url: Optional[str] = Field(
        default=None, min_length=1, description="Base URL for the provider"
    )
    api_key: Optional[str] = Field(None, description="API key")
    timeout: Optional[int] = Field(
        default=None, ge=1, description="Request timeout in seconds"
    )
    models: Optional[List[Dict[str, Any]]] = Field(
        None, description="Static model configs"
    )


class ProviderTestRequest(BaseModel):
    """Request to test an arbitrary provider URL."""

    type: Literal["ollama", "openai_compatible", "llama_cpp"] = Field(
        ..., description="Provider type to probe as"
    )
    base_url: str = Field(..., min_length=1, description="Base URL to test")
    api_key: str = Field("", description="API key for authentication")


class ProviderTestResponse(BaseModel):
    """Result of a provider connectivity test."""

    success: bool = Field(..., description="Whether the connection succeeded")
    message: str = Field(..., description="Human-readable result message")
    models: List[str] = Field(
        default_factory=list, description="Discovered model names"
    )


class DiscoveredServer(BaseModel):
    """A locally discovered LLM server."""

    type: str = Field(..., description="Detected provider type")
    base_url: str = Field(..., description="Server URL")
    suggested_id: str = Field(..., description="Suggested provider ID")
    model_count: int = Field(0, description="Number of models found")
    models: List[str] = Field(default_factory=list, description="Model names")


class DiscoverResponse(BaseModel):
    """Result of local server discovery."""

    servers: List[DiscoveredServer]
