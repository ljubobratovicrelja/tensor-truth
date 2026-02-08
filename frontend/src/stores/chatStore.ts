import { create } from "zustand";
import type {
  RetrievalMetrics,
  SourceNode,
  StreamToolProgress,
  StreamAgentProgress,
  ToolStep,
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
  streamingToolSteps: (ToolStep & { status: "calling" | "completed" | "failed" })[];
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
  addToolStep: (progress: StreamToolProgress) => void;
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
  streamingToolSteps: [],
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
      streamingToolSteps: [],
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
      streamingToolSteps: [],
      agentProgress: null,
      // Note: pendingUserMessage is NOT cleared here - it's needed for auto-send
      // from welcome page. It's cleared by finishStreaming, setError, or explicitly.
    }),

  addToolStep: (progress) =>
    set((state) => {
      if (progress.action === "calling") {
        return {
          streamingToolSteps: [
            ...state.streamingToolSteps,
            {
              tool: progress.tool,
              params: progress.params,
              output: "",
              is_error: false,
              status: "calling" as const,
            },
          ],
        };
      }
      // "completed" or "failed": find last matching "calling" step and update it
      const steps = [...state.streamingToolSteps];
      for (let i = steps.length - 1; i >= 0; i--) {
        if (steps[i].tool === progress.tool && steps[i].status === "calling") {
          steps[i] = {
            ...steps[i],
            output: progress.output ?? "",
            is_error: progress.is_error ?? false,
            status: progress.action,
          };
          break;
        }
      }
      return { streamingToolSteps: steps };
    }),
  setAgentProgress: (progress) => set({ agentProgress: progress }),
}));
