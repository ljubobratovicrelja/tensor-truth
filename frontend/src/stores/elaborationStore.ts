import { create } from "zustand";

/** A passage of one assistant message the reader asked to have explained. */
export interface ElaborationAnchor {
  messageIndex: number;
  selectedText: string;
  occurrenceIndex: number;
}

interface ElaborationStore {
  open: boolean;
  /** The selection currently being explained (null when closed). */
  anchor: ElaborationAnchor | null;
  /** Id of the persisted elaboration, once the backend assigned one. */
  activeElaborationId: string | null;
  content: string;
  thinking: string;
  isStreaming: boolean;
  error: string | null;
  runId: string | null;

  openFor: (anchor: ElaborationAnchor) => void;
  setRunId: (runId: string | null) => void;
  appendToken: (token: string) => void;
  appendThinking: (thinking: string) => void;
  finish: (content: string, elaborationId: string | null) => void;
  setError: (error: string) => void;
  /** Close the panel; the anchor is kept so the highlight can settle back. */
  close: () => void;
  reset: () => void;
}

const INITIAL = {
  open: false,
  anchor: null,
  activeElaborationId: null,
  content: "",
  thinking: "",
  isStreaming: false,
  error: null,
  runId: null,
} as const;

export const useElaborationStore = create<ElaborationStore>((set) => ({
  ...INITIAL,

  // Sequential elaborations replace the previous panel content outright.
  openFor: (anchor) =>
    set({
      ...INITIAL,
      open: true,
      anchor,
      isStreaming: true,
    }),

  setRunId: (runId) => set({ runId }),

  appendToken: (token) => set((state) => ({ content: state.content + token })),

  appendThinking: (thinking) => set((state) => ({ thinking: state.thinking + thinking })),

  finish: (content, elaborationId) =>
    set({
      content,
      activeElaborationId: elaborationId,
      isStreaming: false,
      thinking: "",
      error: null,
      runId: null,
    }),

  // Streaming failures are terminal for this panel but do not close it.
  setError: (error) => set({ error, isStreaming: false, runId: null }),

  close: () =>
    set({
      open: false,
      isStreaming: false,
      error: null,
      runId: null,
    }),

  reset: () => set({ ...INITIAL }),
}));
