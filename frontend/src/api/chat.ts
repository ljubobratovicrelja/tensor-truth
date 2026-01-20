import { apiPost } from "./client";
import type { ChatRequest, ChatResponse, IntentRequest, IntentResponse } from "./types";

export async function sendChatMessage(
  sessionId: string,
  prompt: string
): Promise<ChatResponse> {
  return apiPost<ChatResponse, ChatRequest>(`/sessions/${sessionId}/chat`, { prompt });
}

export async function classifyIntent(
  sessionId: string,
  data: IntentRequest
): Promise<IntentResponse> {
  return apiPost<IntentResponse, IntentRequest>(`/sessions/${sessionId}/intent`, data);
}
