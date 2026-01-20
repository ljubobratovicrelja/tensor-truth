import { apiGet, apiPost, apiPatch, apiDelete } from "./client";
import type {
  SessionCreate,
  SessionResponse,
  SessionListResponse,
  SessionUpdate,
  MessagesResponse,
  MessageCreate,
  MessageResponse,
} from "./types";

export async function listSessions(): Promise<SessionListResponse> {
  return apiGet<SessionListResponse>("/sessions");
}

export async function createSession(data: SessionCreate = {}): Promise<SessionResponse> {
  return apiPost<SessionResponse, SessionCreate>("/sessions", data);
}

export async function getSession(sessionId: string): Promise<SessionResponse> {
  return apiGet<SessionResponse>(`/sessions/${sessionId}`);
}

export async function updateSession(
  sessionId: string,
  data: SessionUpdate
): Promise<SessionResponse> {
  return apiPatch<SessionResponse, SessionUpdate>(`/sessions/${sessionId}`, data);
}

export async function deleteSession(sessionId: string): Promise<void> {
  return apiDelete(`/sessions/${sessionId}`);
}

export async function getSessionMessages(sessionId: string): Promise<MessagesResponse> {
  return apiGet<MessagesResponse>(`/sessions/${sessionId}/messages`);
}

export async function addSessionMessage(
  sessionId: string,
  data: MessageCreate
): Promise<MessageResponse> {
  return apiPost<MessageResponse, MessageCreate>(`/sessions/${sessionId}/messages`, data);
}
