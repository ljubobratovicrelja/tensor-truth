import { apiGet, apiPost } from "./client";
import type {
  StartupStatusResponse,
  IndexDownloadRequest,
  IndexDownloadResponse,
  ModelPullRequest,
  ModelPullResponse,
  ReinitializeIndexesResponse,
} from "./types";

export async function getStartupStatus(): Promise<StartupStatusResponse> {
  return apiGet<StartupStatusResponse>("/startup/status");
}

export async function downloadIndexes(
  request?: IndexDownloadRequest
): Promise<IndexDownloadResponse> {
  return apiPost<IndexDownloadResponse, IndexDownloadRequest>(
    "/startup/download-indexes",
    request || {}
  );
}

export async function pullModel(request: ModelPullRequest): Promise<ModelPullResponse> {
  return apiPost<ModelPullResponse, ModelPullRequest>("/startup/pull-model", request);
}

export async function reinitializeIndexes(): Promise<ReinitializeIndexesResponse> {
  // Note: apiDelete doesn't return a value, so we need to use fetch directly
  const response = await fetch("/api/startup/reinitialize-indexes", {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.json() as Promise<ReinitializeIndexesResponse>;
}
