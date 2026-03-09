import type { ApiError } from "./types";

const API_BASE = "/api";

export class ApiRequestError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiRequestError";
    this.status = status;
    this.detail = detail;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const error = (await response.json()) as ApiError;
      detail = error.detail || detail;
    } catch {
      // Ignore JSON parsing errors
    }
    throw new ApiRequestError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  return handleResponse<T>(response);
}

export async function apiPost<T, D = unknown>(path: string, data?: D): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: data ? JSON.stringify(data) : undefined,
  });
  return handleResponse<T>(response);
}

export async function apiPatch<T, D = unknown>(path: string, data: D): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  return handleResponse<T>(response);
}

export async function apiDelete(path: string): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const error = (await response.json()) as ApiError;
      detail = error.detail || detail;
    } catch {
      // Ignore JSON parsing errors
    }
    throw new ApiRequestError(response.status, detail);
  }
  // 204 No Content - don't try to parse JSON
}

export async function apiPostFormData<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<T>(response);
}

// MCP proposal approval/rejection
export interface McpProposalStatus {
  proposal_id: string;
  action: string;
  config: Record<string, unknown>;
  target_name: string;
  status: "pending" | "approved" | "rejected";
  summary: string;
}

export async function getMcpProposalStatus(
  proposalId: string
): Promise<McpProposalStatus | null> {
  try {
    const response = await fetch(`${API_BASE}/mcp-server-proposals/${proposalId}`);
    if (!response.ok) return null;
    return (await response.json()) as McpProposalStatus;
  } catch {
    return null;
  }
}

export async function approveMcpProposal(proposalId: string): Promise<void> {
  await apiPost(`/mcp-server-proposals/${proposalId}/approve`);
}

export async function rejectMcpProposal(proposalId: string): Promise<void> {
  await apiPost(`/mcp-server-proposals/${proposalId}/reject`);
}

// WebSocket factory
export function createWebSocket(sessionId: string): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return new WebSocket(`${protocol}//${host}/ws/chat/${sessionId}`);
}
