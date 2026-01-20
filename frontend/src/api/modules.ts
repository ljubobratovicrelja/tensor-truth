import { apiGet } from "./client";
import type { ModulesResponse, ModelsResponse, PresetsResponse } from "./types";

export async function listModules(): Promise<ModulesResponse> {
  return apiGet<ModulesResponse>("/modules");
}

export async function listModels(): Promise<ModelsResponse> {
  return apiGet<ModelsResponse>("/models");
}

export async function listPresets(): Promise<PresetsResponse> {
  return apiGet<PresetsResponse>("/presets");
}
