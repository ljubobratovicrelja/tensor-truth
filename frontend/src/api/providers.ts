import { apiGet, apiPost, apiPatch, apiDelete } from "./client";
import type {
  ProviderListResponse,
  ProviderResponse,
  ProviderCreateRequest,
  ProviderUpdateRequest,
  ProviderTestRequest,
  ProviderTestResponse,
  DiscoverResponse,
} from "./types";

export async function getProviders(): Promise<ProviderListResponse> {
  return apiGet<ProviderListResponse>("/providers");
}

export async function addProvider(
  request: ProviderCreateRequest
): Promise<ProviderResponse> {
  return apiPost<ProviderResponse, ProviderCreateRequest>("/providers", request);
}

export async function updateProvider(
  id: string,
  request: ProviderUpdateRequest
): Promise<ProviderResponse> {
  return apiPatch<ProviderResponse, ProviderUpdateRequest>(`/providers/${id}`, request);
}

export async function removeProvider(id: string): Promise<void> {
  return apiDelete(`/providers/${id}`);
}

export async function testProviderUrl(
  request: ProviderTestRequest
): Promise<ProviderTestResponse> {
  return apiPost<ProviderTestResponse, ProviderTestRequest>("/providers/test", request);
}

export async function discoverServers(): Promise<DiscoverResponse> {
  return apiGet<DiscoverResponse>("/providers/discover");
}
