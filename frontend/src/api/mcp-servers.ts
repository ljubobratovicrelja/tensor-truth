import { apiDelete, apiGet, apiPatch, apiPost } from "./client";
import type {
  MCPServerResponse,
  MCPServerListResponse,
  MCPServerCreateRequest,
  MCPServerUpdateRequest,
  MCPServerToggleRequest,
  MCPServerPresetsResponse,
} from "./types";

export async function getMcpServers(): Promise<MCPServerListResponse> {
  return apiGet<MCPServerListResponse>("/mcp-servers/");
}

export async function getMcpServerPresets(): Promise<MCPServerPresetsResponse> {
  return apiGet<MCPServerPresetsResponse>("/mcp-servers/presets");
}

export async function addMcpServer(
  request: MCPServerCreateRequest
): Promise<MCPServerResponse> {
  return apiPost<MCPServerResponse, MCPServerCreateRequest>("/mcp-servers/", request);
}

export async function updateMcpServer(
  name: string,
  request: MCPServerUpdateRequest
): Promise<MCPServerResponse> {
  return apiPatch<MCPServerResponse, MCPServerUpdateRequest>(
    `/mcp-servers/${encodeURIComponent(name)}`,
    request
  );
}

export async function deleteMcpServer(name: string): Promise<void> {
  return apiDelete(`/mcp-servers/${encodeURIComponent(name)}`);
}

export async function toggleMcpServer(
  name: string,
  request: MCPServerToggleRequest
): Promise<MCPServerResponse> {
  return apiPatch<MCPServerResponse, MCPServerToggleRequest>(
    `/mcp-servers/${encodeURIComponent(name)}/toggle`,
    request
  );
}
