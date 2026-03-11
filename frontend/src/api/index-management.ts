import { apiGet, apiPost, apiDelete } from "./client";
import type {
  SourcesResponse,
  SourceCreateRequest,
  ModulesOverviewResponse,
  FetchRequest,
  BuildRequest,
  BatchBuildRequest,
  BatchBuildResponse,
  TaskSubmitResponse,
  ValidateResponse,
} from "./types";

const BASE = "/index-management";

export async function getSources(): Promise<SourcesResponse> {
  return apiGet<SourcesResponse>(`${BASE}/sources`);
}

export async function addSource(data: SourceCreateRequest): Promise<void> {
  await apiPost(`${BASE}/sources`, data);
}

export async function updateSource(
  sourceType: string,
  name: string,
  config: Record<string, unknown>
): Promise<void> {
  const response = await fetch(`/api${BASE}/sources/${sourceType}/${name}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
}

export async function deleteSource(sourceType: string, name: string): Promise<void> {
  await apiDelete(`${BASE}/sources/${sourceType}/${name}`);
}

export async function getModulesOverview(
  embeddingModelId?: string
): Promise<ModulesOverviewResponse> {
  const params = embeddingModelId
    ? `?embedding_model_id=${encodeURIComponent(embeddingModelId)}`
    : "";
  return apiGet<ModulesOverviewResponse>(`${BASE}/modules${params}`);
}

export async function fetchSource(data: FetchRequest): Promise<TaskSubmitResponse> {
  return apiPost<TaskSubmitResponse>(`${BASE}/fetch`, data);
}

export async function buildIndex(data: BuildRequest): Promise<TaskSubmitResponse> {
  return apiPost<TaskSubmitResponse>(`${BASE}/build`, data);
}

export async function batchBuild(data: BatchBuildRequest): Promise<BatchBuildResponse> {
  return apiPost<BatchBuildResponse>(`${BASE}/build/batch`, data);
}

export async function deleteIndex(
  embeddingModelId: string,
  moduleName: string
): Promise<void> {
  await apiDelete(`${BASE}/indexes/${embeddingModelId}/${moduleName}`);
}

export async function validateSources(): Promise<ValidateResponse> {
  return apiPost<ValidateResponse>(`${BASE}/validate`);
}
