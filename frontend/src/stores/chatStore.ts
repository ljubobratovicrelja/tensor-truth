import { create } from "zustand";
import type { SourceNode } from "@/api/types";

interface ChatStore {
  isStreaming: boolean;
  streamingContent: string;
  streamingSources: SourceNode[];
  confidenceLevel: string | null;
  error: string | null;
  pendingUserMessage: string | null;

  startStreaming: (userMessage: string) => void;
  appendToken: (token: string) => void;
  setSources: (sources: SourceNode[]) => void;
  finishStreaming: (content: string, confidenceLevel: string) => void;
  clearPendingUserMessage: () => void;
  setError: (error: string) => void;
  reset: () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  isStreaming: false,
  streamingContent: "",
  streamingSources: [],
  confidenceLevel: null,
  error: null,
  pendingUserMessage: null,

  startStreaming: (userMessage: string) =>
    set({
      isStreaming: true,
      streamingContent: "",
      streamingSources: [],
      confidenceLevel: null,
      error: null,
      pendingUserMessage: userMessage,
    }),

  appendToken: (token) =>
    set((state) => ({
      streamingContent: state.streamingContent + token,
    })),

  setSources: (sources) => set({ streamingSources: sources }),

  finishStreaming: (content, confidenceLevel) =>
    set({
      isStreaming: false,
      streamingContent: content,
      confidenceLevel,
      pendingUserMessage: null,
    }),

  clearPendingUserMessage: () => set({ pendingUserMessage: null }),

  setError: (error) =>
    set({
      isStreaming: false,
      error,
      pendingUserMessage: null,
    }),

  reset: () =>
    set({
      isStreaming: false,
      streamingContent: "",
      streamingSources: [],
      confidenceLevel: null,
      error: null,
      pendingUserMessage: null,
    }),
}));
