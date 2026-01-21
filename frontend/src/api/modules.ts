import { apiGet } from "./client";
import type {
  ModulesResponse,
  ModelsResponse,
  PresetsResponse,
  EmbeddingModelsResponse,
} from "./types";

export async function listModules(): Promise<ModulesResponse> {
  return apiGet<ModulesResponse>("/modules");
}

export async function listModels(): Promise<ModelsResponse> {
  return apiGet<ModelsResponse>("/models");
}

export async function listEmbeddingModels(): Promise<EmbeddingModelsResponse> {
  return apiGet<EmbeddingModelsResponse>("/embedding-models");
}

export async function listPresets(): Promise<PresetsResponse> {
  return apiGet<PresetsResponse>("/presets");
}

export async function listFavoritePresets(): Promise<PresetsResponse> {
  return apiGet<PresetsResponse>("/presets/favorites");
}
