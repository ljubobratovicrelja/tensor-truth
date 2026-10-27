import { create } from "zustand";
import type {
  ImageRef,
  RetrievalMetrics,
  SourceNode,
  StreamConfirmationRequest,
  StreamToolProgress,
  StreamToolPhase,
  StreamAgentProgress,
  ToolStep,
} from "@/api/types";
import type { AttachedImage } from "@/hooks/useWebSocket";

export type ToolStepWithStatus = ToolStep & {
  status: "calling" | "completed" | "failed";
};

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
  /** Intermediate narration segments that must render outside the answer. */
  narrationSegments: string[];
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
  streamingToolSteps: ToolStepWithStatus[];
  agentProgress: StreamAgentProgress | null;
  toolPhase: StreamToolPhase | null;
  streamingReasoning: string;

  // Tool confirmation requests
  confirmationRequests: StreamConfirmationRequest[];

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
  /** Move a completed intermediate step out of the answer into narration. */
  commitNarrationSegment: (text: string) => void;
  setStatus: (status: PipelineStatus) => void;
  setSources: (sources: SourceNode[], sourceTypes?: string[]) => void;
  setMetrics: (metrics: RetrievalMetrics | null) => void;
  finishStreaming: (
    content: string,
    confidenceLevel: string,
    backendTokens?: { inputTokens: number; outputTokens: number },
    narrationSegments?: string[]
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

  // Tool confirmation request setters
  addConfirmationRequest: (req: StreamConfirmationRequest) => void;
  resolveConfirmationRequest: (
    confirmationId: string,
    outcome: "approved" | "rejected"
  ) => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  isStreaming: false,
  streamingContent: "",
  streamingThinking: "",
  narrationSegments: [],
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
  confirmationRequests: [],
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
      narrationSegments: [],
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
      confirmationRequests: [],
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

  commitNarrationSegment: (text) =>
    set((state) => {
      const segment = text.trim();
      if (!segment) return state;

      // The segment's tokens already went into streamingContent before the
      // backend knew this step would call a tool. Remove that exact text from
      // the tail so the final answer and narration cannot overlap.
      let content = state.streamingContent;
      if (content.endsWith(text)) {
        content = content.slice(0, -text.length);
      } else {
        const idx = content.lastIndexOf(text);
        if (idx !== -1) {
          content = content.slice(0, idx) + content.slice(idx + text.length);
        }
      }

      return {
        streamingContent: content.replace(/\s+$/, ""),
        narrationSegments: [...state.narrationSegments, segment],
      };
    }),

  setStatus: (status) => set({ pipelineStatus: status }),

  setSources: (sources, sourceTypes) =>
    set({ streamingSources: sources, streamingSourceTypes: sourceTypes ?? null }),

  setMetrics: (metrics) => set({ streamingMetrics: metrics }),

  finishStreaming: (content, confidenceLevel, backendTokens, narrationSegments) =>
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
        narrationSegments: narrationSegments ?? state.narrationSegments,
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
      narrationSegments: [],
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
      narrationSegments: [],
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
  // Phases carrying a numeric `percent` are in-step progress: attach them to
  // the running step for that tool (so each step renders its own bar) while
  // still keeping the latest phase for the global indicator. Same-tool
  // concurrent calls are disambiguated by the `scope` param when present;
  // otherwise the most recently started still-running call owns the phase.
  setToolPhase: (phase) =>
    set((state) => {
      if (!phase) return { toolPhase: null };

      const rawPercent = phase.metadata?.percent;
      const percent =
        typeof rawPercent === "number" && Number.isFinite(rawPercent)
          ? Math.max(0, Math.min(100, rawPercent))
          : null;

      if (percent === null) {
        return { toolPhase: phase };
      }

      const scope = phase.metadata?.scope;
      const isRunningForTool = (s: (typeof state.streamingToolSteps)[number]) =>
        s.status === "calling" &&
        (s.tool === phase.tool_id || s.tool_id === phase.tool_id);

      let idx = -1;
      if (scope !== undefined) {
        idx = state.streamingToolSteps.findLastIndex(
          (s) => isRunningForTool(s) && s.params?.scope === scope
        );
      }
      if (idx === -1) {
        idx = state.streamingToolSteps.findLastIndex(isRunningForTool);
      }
      if (idx === -1) {
        return { toolPhase: phase };
      }

      const steps = [...state.streamingToolSteps];
      steps[idx] = {
        ...steps[idx],
        progress: {
          percent,
          message: phase.message,
          stage: phase.phase,
        },
      };
      return { toolPhase: phase, streamingToolSteps: steps };
    }),
  appendReasoning: (reasoning) =>
    set((state) => ({
      streamingReasoning: state.streamingReasoning + reasoning,
    })),
  clearReasoning: () => set({ streamingReasoning: "" }),

  // Idempotent by confirmation_id: a duplicate frame must never render a
  // second card for the same confirmation, otherwise the second Approve
  // fails with "already approved" and the waiting tool hangs.
  addConfirmationRequest: (req) =>
    set((state) =>
      state.confirmationRequests.some((r) => r.confirmation_id === req.confirmation_id)
        ? state
        : { confirmationRequests: [...state.confirmationRequests, req] }
    ),
  resolveConfirmationRequest: (confirmationId, outcome) =>
    set((state) => ({
      confirmationRequests: state.confirmationRequests.map((r) =>
        r.confirmation_id === confirmationId
          ? ({ ...r, _outcome: outcome } as StreamConfirmationRequest)
          : r
      ),
    })),
}));
