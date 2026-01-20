import { create } from "zustand";
import type { SourceNode } from "@/api/types";

interface ChatStore {
  isStreaming: boolean;
  streamingContent: string;
  streamingSources: SourceNode[];
  confidenceLevel: string | null;
  error: string | null;

  startStreaming: () => void;
  appendToken: (token: string) => void;
  setSources: (sources: SourceNode[]) => void;
  finishStreaming: (content: string, confidenceLevel: string) => void;
  setError: (error: string) => void;
  reset: () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  isStreaming: false,
  streamingContent: "",
  streamingSources: [],
  confidenceLevel: null,
  error: null,

  startStreaming: () =>
    set({
      isStreaming: true,
      streamingContent: "",
      streamingSources: [],
      confidenceLevel: null,
      error: null,
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
    }),

  setError: (error) =>
    set({
      isStreaming: false,
      error,
    }),

  reset: () =>
    set({
      isStreaming: false,
      streamingContent: "",
      streamingSources: [],
      confidenceLevel: null,
      error: null,
    }),
}));
