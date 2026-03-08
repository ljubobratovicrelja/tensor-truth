import { create } from "zustand";
import type {
  ImageRef,
  RetrievalMetrics,
  SourceNode,
  StreamToolProgress,
  StreamToolPhase,
  StreamAgentProgress,
  ToolStep,
} from "@/api/types";
import type { AttachedImage } from "@/hooks/useWebSocket";

export interface ResponseStats {
  inputTokens: number;
  outputTokens: number;
  totalTimeSec: number;
  tps: number | null;
}

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
  pendingUserImages: (ImageRef & { previewUrl?: string })[] | null;
  /** Full AttachedImage objects for auto-send from welcome page */
  pendingAttachedImages: AttachedImage[] | null;

  // Agent/tool progress
  streamingToolSteps: (ToolStep & { status: "calling" | "completed" | "failed" })[];
  agentProgress: StreamAgentProgress | null;
  toolPhase: StreamToolPhase | null;
  streamingReasoning: string;

  // Response stats tracking
  streamingRequestTime: number | null;
  streamingStartTime: number | null;
  streamingCharCount: number;
  streamingInputCharCount: number;
  lastResponseStats: ResponseStats | null;

  startStreaming: (
    userMessage: string,
    images?: (ImageRef & { previewUrl?: string })[]
  ) => void;
  appendToken: (token: string) => void;
  appendThinking: (thinking: string) => void;
  setStatus: (status: PipelineStatus) => void;
  setSources: (sources: SourceNode[], sourceTypes?: string[]) => void;
  setMetrics: (metrics: RetrievalMetrics | null) => void;
  finishStreaming: (
    content: string,
    confidenceLevel: string,
    backendTokens?: { inputTokens: number; outputTokens: number }
  ) => void;
  setPendingUserMessage: (message: string | null) => void;
  setPendingAttachedImages: (images: AttachedImage[] | null) => void;
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
  pendingUserImages: null,
  pendingAttachedImages: null,
  streamingToolSteps: [],
  agentProgress: null,
  toolPhase: null,
  streamingReasoning: "",
  streamingRequestTime: null,
  streamingStartTime: null,
  streamingCharCount: 0,
  streamingInputCharCount: 0,
  lastResponseStats: null,

  startStreaming: (
    userMessage: string,
    images?: (ImageRef & { previewUrl?: string })[]
  ) =>
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
      pendingUserImages: images ?? null,
      streamingToolSteps: [],
      agentProgress: null,
      toolPhase: null,
      streamingRequestTime: Date.now(),
      streamingStartTime: null,
      streamingCharCount: 0,
      streamingInputCharCount: userMessage.length,
      lastResponseStats: null,
    }),

  appendToken: (token) =>
    set((state) => ({
      streamingContent: state.streamingContent + token,
      streamingStartTime: state.streamingStartTime ?? Date.now(),
      streamingCharCount: state.streamingCharCount + token.length,
    })),

  appendThinking: (thinking) =>
    set((state) => ({
      streamingThinking: state.streamingThinking + thinking,
    })),

  setStatus: (status) => set({ pipelineStatus: status }),

  setSources: (sources, sourceTypes) =>
    set({ streamingSources: sources, streamingSourceTypes: sourceTypes ?? null }),

  setMetrics: (metrics) => set({ streamingMetrics: metrics }),

  finishStreaming: (content, confidenceLevel, backendTokens) =>
    set((state) => {
      const now = Date.now();
      const genElapsed = state.streamingStartTime
        ? (now - state.streamingStartTime) / 1000
        : null;
      const totalTimeSec = state.streamingRequestTime
        ? Math.round(((now - state.streamingRequestTime) / 1000) * 10) / 10
        : 0;
      // Prefer backend-provided counts; fall back to char estimation
      const outputTokens =
        backendTokens?.outputTokens ?? Math.round(state.streamingCharCount / 4);
      const inputTokens =
        backendTokens?.inputTokens ?? Math.round(state.streamingInputCharCount / 4);
      const tps =
        genElapsed && outputTokens > 0
          ? Math.round((outputTokens / genElapsed) * 10) / 10
          : null;
      return {
        isStreaming: false,
        streamingContent: content,
        confidenceLevel,
        pipelineStatus: null,
        toolPhase: null,
        streamingThinking: "",
        streamingReasoning: "",
        pendingUserMessage: null,
        pendingUserImages: null,
        pendingAttachedImages: null,
        streamingRequestTime: null,
        streamingStartTime: null,
        streamingCharCount: 0,
        streamingInputCharCount: 0,
        lastResponseStats: { inputTokens, outputTokens, totalTimeSec, tps },
      };
    }),

  setPendingUserMessage: (message) => set({ pendingUserMessage: message }),

  setPendingAttachedImages: (images) => set({ pendingAttachedImages: images }),

  clearPendingUserMessage: () =>
    set({ pendingUserMessage: null, pendingAttachedImages: null }),

  setError: (error) =>
    set({
      isStreaming: false,
      error,
      pipelineStatus: null,
      toolPhase: null,
      streamingThinking: "",
      streamingReasoning: "",
      pendingUserMessage: null,
      pendingUserImages: null,
      pendingAttachedImages: null,
      lastResponseStats: null,
      streamingRequestTime: null,
      streamingStartTime: null,
      streamingCharCount: 0,
      streamingInputCharCount: 0,
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
      lastResponseStats: null,
      streamingRequestTime: null,
      streamingStartTime: null,
      streamingCharCount: 0,
      streamingInputCharCount: 0,
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
