import { apiPost } from "./client";
import type { ChatRequest, ChatResponse } from "./types";

export async function sendChatMessage(
  sessionId: string,
  prompt: string
): Promise<ChatResponse> {
  return apiPost<ChatResponse, ChatRequest>(`/sessions/${sessionId}/chat`, { prompt });
}
