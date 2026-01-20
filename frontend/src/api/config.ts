import { apiGet, apiPatch } from "./client";
import type { ConfigResponse, ConfigUpdateRequest } from "./types";

export async function getConfig(): Promise<ConfigResponse> {
  return apiGet<ConfigResponse>("/config");
}

export async function updateConfig(
  updates: Record<string, unknown>
): Promise<ConfigResponse> {
  return apiPatch<ConfigResponse, ConfigUpdateRequest>("/config", { updates });
}

export async function getConfigDefaults(): Promise<ConfigResponse> {
  return apiGet<ConfigResponse>("/config/defaults");
}
