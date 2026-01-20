import { create } from "zustand";
import type { SourceNode } from "@/api/types";

export type PipelineStatus = "retrieving" | "thinking" | "generating" | null;

interface ChatStore {
  isStreaming: boolean;
  streamingContent: string;
  streamingThinking: string;
  streamingSources: SourceNode[];
  confidenceLevel: string | null;
  pipelineStatus: PipelineStatus;
  error: string | null;
  pendingUserMessage: string | null;

  startStreaming: (userMessage: string) => void;
  appendToken: (token: string) => void;
  appendThinking: (thinking: string) => void;
  setStatus: (status: PipelineStatus) => void;
  setSources: (sources: SourceNode[]) => void;
  finishStreaming: (content: string, confidenceLevel: string) => void;
  clearPendingUserMessage: () => void;
  setError: (error: string) => void;
  reset: () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  isStreaming: false,
  streamingContent: "",
  streamingThinking: "",
  streamingSources: [],
  confidenceLevel: null,
  pipelineStatus: null,
  error: null,
  pendingUserMessage: null,

  startStreaming: (userMessage: string) =>
    set({
      isStreaming: true,
      streamingContent: "",
      streamingThinking: "",
      streamingSources: [],
      confidenceLevel: null,
      pipelineStatus: null,
      error: null,
      pendingUserMessage: userMessage,
    }),

  appendToken: (token) =>
    set((state) => ({
      streamingContent: state.streamingContent + token,
    })),

  appendThinking: (thinking) =>
    set((state) => ({
      streamingThinking: state.streamingThinking + thinking,
    })),

  setStatus: (status) => set({ pipelineStatus: status }),

  setSources: (sources) => set({ streamingSources: sources }),

  finishStreaming: (content, confidenceLevel) =>
    set({
      isStreaming: false,
      streamingContent: content,
      confidenceLevel,
      pipelineStatus: null,
      pendingUserMessage: null,
    }),

  clearPendingUserMessage: () => set({ pendingUserMessage: null }),

  setError: (error) =>
    set({
      isStreaming: false,
      error,
      pipelineStatus: null,
      pendingUserMessage: null,
    }),

  reset: () =>
    set({
      isStreaming: false,
      streamingContent: "",
      streamingThinking: "",
      streamingSources: [],
      confidenceLevel: null,
      pipelineStatus: null,
      error: null,
      pendingUserMessage: null,
    }),
}));
