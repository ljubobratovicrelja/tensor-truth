import { apiGet } from "./client";
import type { ModulesResponse, ModelsResponse, EmbeddingModelsResponse } from "./types";

export async function listModules(): Promise<ModulesResponse> {
  return apiGet<ModulesResponse>("/modules");
}

export async function listModels(): Promise<ModelsResponse> {
  return apiGet<ModelsResponse>("/models");
}

export async function listEmbeddingModels(): Promise<EmbeddingModelsResponse> {
  return apiGet<EmbeddingModelsResponse>("/embedding-models");
}
