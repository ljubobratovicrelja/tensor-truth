import { apiGet, apiPost, apiPatch, apiDelete } from "./client";
import type {
  SessionCreate,
  SessionResponse,
  SessionListResponse,
  SessionUpdate,
  MessagesResponse,
  MessageCreate,
  MessageResponse,
  TitlePreviewRequest,
  TitlePreviewResponse,
} from "./types";

export async function listSessions(): Promise<SessionListResponse> {
  return apiGet<SessionListResponse>("/sessions");
}

export async function createSession(data: SessionCreate = {}): Promise<SessionResponse> {
  return apiPost<SessionResponse, SessionCreate>("/sessions", data);
}

export async function previewSessionTitle(
  data: TitlePreviewRequest
): Promise<TitlePreviewResponse> {
  return apiPost<TitlePreviewResponse, TitlePreviewRequest>(
    "/sessions/title-preview",
    data
  );
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

export interface SessionStatsResponse {
  // Total session history (all messages stored)
  history_messages: number;
  history_chars: number;
  history_tokens_estimate: number;

  // Compiled history (what's actually sent to LLM per config)
  compiled_history_messages: number;
  compiled_history_chars: number;
  compiled_history_tokens_estimate: number;

  // Config limit: max conversation turns (1 turn = user query + assistant response)
  max_history_turns: number;

  model_name: string | null;
  context_length: number;
}

export async function deleteMessage(
  sessionId: string,
  messageIndex: number
): Promise<void> {
  return apiDelete(`/sessions/${sessionId}/messages/${messageIndex}`);
}

export async function getSessionStats(sessionId: string): Promise<SessionStatsResponse> {
  return apiGet<SessionStatsResponse>(`/sessions/${sessionId}/stats`);
}

export interface ContextCategory {
  id: string;
  label: string;
  hint: string;
  chars: number;
  tokens: number;
  share: number;
}

export interface SessionContextResponse {
  measured: boolean;
  categories: ContextCategory[];
  total_chars: number;
  total_tokens: number;
  context_window: number;
  used_percent: number;
  remaining_tokens: number | null;
  model_name: string | null;
  messages_included: number;
  messages_total: number;
  measured_at: number | null;
}

export interface CompactResponse {
  compacted: boolean;
  message: string;
  messages_before: number;
  messages_after: number;
  messages_compacted: number;
  chars_before: number;
  chars_after: number;
  tokens_saved: number;
  summary_chars: number;
  backup_path: string | null;
}

export async function getSessionContext(
  sessionId: string
): Promise<SessionContextResponse> {
  return apiGet<SessionContextResponse>(`/sessions/${sessionId}/context`);
}

export async function compactSession(
  sessionId: string,
  keepLast = 6
): Promise<CompactResponse> {
  return apiPost<CompactResponse, { keep_last: number }>(
    `/sessions/${sessionId}/compact`,
    { keep_last: keepLast }
  );
}
