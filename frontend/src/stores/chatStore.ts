import { create } from "zustand";
import type {
  RetrievalMetrics,
  SourceNode,
  StreamToolProgress,
  StreamAgentProgress,
} from "@/api/types";

export type PipelineStatus =
  | "loading_models"
  | "retrieving"
  | "reranking"
  | "thinking"
  | "generating"
  | null;

interface ChatStore {
  isStreaming: boolean;
  streamingContent: string;
  streamingThinking: string;
  streamingSources: SourceNode[];
  streamingMetrics: RetrievalMetrics | null;
  confidenceLevel: string | null;
  pipelineStatus: PipelineStatus;
  error: string | null;
  pendingUserMessage: string | null;

  // Agent/tool progress
  toolProgress: StreamToolProgress | null;
  agentProgress: StreamAgentProgress | null;

  startStreaming: (userMessage: string) => void;
  appendToken: (token: string) => void;
  appendThinking: (thinking: string) => void;
  setStatus: (status: PipelineStatus) => void;
  setSources: (sources: SourceNode[]) => void;
  setMetrics: (metrics: RetrievalMetrics | null) => void;
  finishStreaming: (content: string, confidenceLevel: string) => void;
  setPendingUserMessage: (message: string | null) => void;
  clearPendingUserMessage: () => void;
  setError: (error: string) => void;
  reset: () => void;

  // Agent/tool progress setters
  setToolProgress: (progress: StreamToolProgress | null) => void;
  setAgentProgress: (progress: StreamAgentProgress | null) => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  isStreaming: false,
  streamingContent: "",
  streamingThinking: "",
  streamingSources: [],
  streamingMetrics: null,
  confidenceLevel: null,
  pipelineStatus: null,
  error: null,
  pendingUserMessage: null,
  toolProgress: null,
  agentProgress: null,

  startStreaming: (userMessage: string) =>
    set({
      isStreaming: true,
      streamingContent: "",
      streamingThinking: "",
      streamingSources: [],
      streamingMetrics: null,
      confidenceLevel: null,
      pipelineStatus: null,
      error: null,
      pendingUserMessage: userMessage,
      toolProgress: null,
      agentProgress: null,
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

  setMetrics: (metrics) => set({ streamingMetrics: metrics }),

  finishStreaming: (content, confidenceLevel) =>
    set({
      isStreaming: false,
      streamingContent: content,
      confidenceLevel,
      pipelineStatus: null,
      pendingUserMessage: null,
    }),

  setPendingUserMessage: (message) => set({ pendingUserMessage: message }),

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
      streamingMetrics: null,
      confidenceLevel: null,
      pipelineStatus: null,
      error: null,
      toolProgress: null,
      agentProgress: null,
      // Note: pendingUserMessage is NOT cleared here - it's needed for auto-send
      // from welcome page. It's cleared by finishStreaming, setError, or explicitly.
    }),

  setToolProgress: (progress) => set({ toolProgress: progress }),
  setAgentProgress: (progress) => set({ agentProgress: progress }),
}));
