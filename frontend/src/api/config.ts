import { apiGet, apiPatch } from "./client";
import type {
  ConfigResponse,
  ConfigUpdateRequest,
  ModelCapabilitiesResponse,
} from "./types";

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

export async function getAvailableDevices(): Promise<string[]> {
  const response = await apiGet<{ devices: string[] }>("/config/devices");
  return response.devices;
}

export async function getModelCapabilities(
  model: string
): Promise<ModelCapabilitiesResponse> {
  return apiGet<ModelCapabilitiesResponse>(
    `/config/model-capabilities?model=${encodeURIComponent(model)}`
  );
}
