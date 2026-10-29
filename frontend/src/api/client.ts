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

/**
 * Human-readable text from a FastAPI error body. `detail` is usually a string
 * but conflict responses carry an object (`{"detail": ..., "run_id": ...}`);
 * rendering that raw produced "[object Object]" in the UI.
 */
export function extractDetail(error: ApiError): string | undefined {
  const d = error.detail;
  if (!d) return undefined;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d.map((item: { msg?: string }) => item.msg || String(item)).join("; ");
  }
  if (typeof d === "object") {
    const nested = d as { detail?: unknown; message?: unknown; msg?: unknown };
    for (const value of [nested.detail, nested.message, nested.msg]) {
      if (typeof value === "string" && value.trim()) return value;
    }
    try {
      return JSON.stringify(d);
    } catch {
      return String(d);
    }
  }
  return String(d);
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const error = (await response.json()) as ApiError;
      detail = extractDetail(error) || detail;
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
      detail = extractDetail(error) || detail;
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

// Tool confirmation approval/rejection
export interface ToolConfirmationStatus {
  confirmation_id: string;
  tool_name: string;
  action_type: string;
  title: string;
  summary: string;
  details: Record<string, unknown>;
  status: "pending" | "approved" | "rejected" | "expired";
}

export async function getToolConfirmationStatus(
  confirmationId: string
): Promise<ToolConfirmationStatus | null> {
  try {
    const response = await fetch(`${API_BASE}/tool-confirmations/${confirmationId}`);
    if (!response.ok) return null;
    return (await response.json()) as ToolConfirmationStatus;
  } catch {
    return null;
  }
}

export async function approveToolConfirmation(confirmationId: string): Promise<void> {
  await apiPost(`/tool-confirmations/${confirmationId}/approve`);
}

export async function rejectToolConfirmation(confirmationId: string): Promise<void> {
  await apiPost(`/tool-confirmations/${confirmationId}/reject`);
}

/**
 * Resolve a confirmation with a tool-defined outcome (e.g. "context" for
 * add_arxiv_paper, which offers an alternative to indexing).
 */
export async function resolveToolConfirmation(
  confirmationId: string,
  outcome: string
): Promise<void> {
  await apiPost(`/tool-confirmations/${confirmationId}/resolve`, { outcome });
}

// WebSocket factory
export function createWebSocket(sessionId: string): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return new WebSocket(`${protocol}//${host}/ws/chat/${sessionId}`);
}
