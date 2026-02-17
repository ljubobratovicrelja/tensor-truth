import { create } from "zustand";
import type {
  RetrievalMetrics,
  SourceNode,
  StreamToolProgress,
  StreamToolPhase,
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
  streamingSourceTypes: string[] | null;
  confidenceLevel: string | null;
  pipelineStatus: PipelineStatus;
  error: string | null;
  pendingUserMessage: string | null;

  // Agent/tool progress
  streamingToolSteps: (ToolStep & { status: "calling" | "completed" | "failed" })[];
  agentProgress: StreamAgentProgress | null;
  toolPhase: StreamToolPhase | null;
  streamingReasoning: string;

  startStreaming: (userMessage: string) => void;
  appendToken: (token: string) => void;
  appendThinking: (thinking: string) => void;
  setStatus: (status: PipelineStatus) => void;
  setSources: (sources: SourceNode[], sourceTypes?: string[]) => void;
  setMetrics: (metrics: RetrievalMetrics | null) => void;
  finishStreaming: (content: string, confidenceLevel: string) => void;
  setPendingUserMessage: (message: string | null) => void;
  clearPendingUserMessage: () => void;
  setError: (error: string) => void;
  reset: () => void;

  // Agent/tool progress setters
  addToolStep: (progress: StreamToolProgress) => void;
  setAgentProgress: (progress: StreamAgentProgress | null) => void;
  setToolPhase: (phase: StreamToolPhase | null) => void;
  appendReasoning: (reasoning: string) => void;
  clearReasoning: () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  isStreaming: false,
  streamingContent: "",
  streamingThinking: "",
  streamingSources: [],
  streamingMetrics: null,
  streamingSourceTypes: null,
  confidenceLevel: null,
  pipelineStatus: null,
  error: null,
  pendingUserMessage: null,
  streamingToolSteps: [],
  agentProgress: null,
  toolPhase: null,
  streamingReasoning: "",

  startStreaming: (userMessage: string) =>
    set({
      isStreaming: true,
      streamingContent: "",
      streamingThinking: "",
      streamingReasoning: "",
      streamingSources: [],
      streamingMetrics: null,
      streamingSourceTypes: null,
      confidenceLevel: null,
      pipelineStatus: null,
      error: null,
      pendingUserMessage: userMessage,
      streamingToolSteps: [],
      agentProgress: null,
      toolPhase: null,
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

  setSources: (sources, sourceTypes) =>
    set({ streamingSources: sources, streamingSourceTypes: sourceTypes ?? null }),

  setMetrics: (metrics) => set({ streamingMetrics: metrics }),

  finishStreaming: (content, confidenceLevel) =>
    set({
      isStreaming: false,
      streamingContent: content,
      confidenceLevel,
      pipelineStatus: null,
      toolPhase: null,
      streamingReasoning: "",
      pendingUserMessage: null,
    }),

  setPendingUserMessage: (message) => set({ pendingUserMessage: message }),

  clearPendingUserMessage: () => set({ pendingUserMessage: null }),

  setError: (error) =>
    set({
      isStreaming: false,
      error,
      pipelineStatus: null,
      toolPhase: null,
      streamingReasoning: "",
      pendingUserMessage: null,
    }),

  reset: () =>
    set({
      isStreaming: false,
      streamingContent: "",
      streamingThinking: "",
      streamingReasoning: "",
      streamingSources: [],
      streamingMetrics: null,
      streamingSourceTypes: null,
      confidenceLevel: null,
      pipelineStatus: null,
      error: null,
      streamingToolSteps: [],
      agentProgress: null,
      toolPhase: null,
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
              tool_id: progress.tool_id,
            },
          ],
        };
      }
      // "completed" or "failed": match by tool_id (unique) when available,
      // fall back to backward search by tool name for older messages
      const steps = [...state.streamingToolSteps];
      const idx = progress.tool_id
        ? steps.findIndex((s) => s.tool_id === progress.tool_id && s.status === "calling")
        : steps.findLastIndex((s) => s.tool === progress.tool && s.status === "calling");
      if (idx !== -1) {
        steps[idx] = {
          ...steps[idx],
          output: progress.output ?? "",
          is_error: progress.is_error ?? false,
          status: progress.action,
        };
      }
      return { streamingToolSteps: steps };
    }),
  setAgentProgress: (progress) => set({ agentProgress: progress }),
  setToolPhase: (phase) => set({ toolPhase: phase }),
  appendReasoning: (reasoning) =>
    set((state) => ({
      streamingReasoning: state.streamingReasoning + reasoning,
    })),
  clearReasoning: () => set({ streamingReasoning: "" }),
}));
