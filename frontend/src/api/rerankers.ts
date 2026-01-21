import { apiGet, apiPost } from "./client";
import type {
  RerankerAddRequest,
  RerankerAddResponse,
  RerankerListResponse,
  RerankerRemoveResponse,
} from "./types";

export async function getRerankers(): Promise<RerankerListResponse> {
  return apiGet<RerankerListResponse>("/rerankers");
}

export async function addReranker(model: string): Promise<RerankerAddResponse> {
  return apiPost<RerankerAddResponse, RerankerAddRequest>("/rerankers", {
    model,
  });
}

export async function removeReranker(model: string): Promise<RerankerRemoveResponse> {
  // URL-encode the model path (e.g., "BAAI/bge-reranker-v2-m3")
  const encodedModel = encodeURIComponent(model);
  const response = await fetch(`/api/rerankers/${encodedModel}`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const error = await response.json();
      detail = error.detail || detail;
    } catch {
      // Ignore JSON parsing errors
    }
    throw new Error(detail);
  }
  return response.json() as Promise<RerankerRemoveResponse>;
}
