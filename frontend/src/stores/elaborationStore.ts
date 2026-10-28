import { create } from "zustand";
import type { ThreadMessage } from "@/api/types";

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
  /** The original explanation (never the follow-up reply). */
  content: string;
  thinking: string;
  isStreaming: boolean;
  error: string | null;
  runId: string | null;

  /** Persisted follow-up turns, oldest first. */
  thread: ThreadMessage[];
  /** Partial follow-up answer currently streaming (empty when idle). */
  reply: string;
  isReplying: boolean;
  followUpError: string | null;

  openFor: (anchor: ElaborationAnchor) => void;
  /** Re-open an already persisted elaboration without starting a new run. */
  openStored: (
    anchor: ElaborationAnchor,
    elaborationId: string,
    content: string,
    messages?: ThreadMessage[] | null
  ) => void;
  setRunId: (runId: string | null) => void;
  appendToken: (token: string) => void;
  appendThinking: (thinking: string) => void;
  finish: (content: string, elaborationId: string | null) => void;
  setError: (error: string) => void;

  /** Begin a follow-up: the question shows immediately, before the run id. */
  startFollowUp: (userText: string) => void;
  appendReplyToken: (token: string) => void;
  /** Persisted answer arrived: move it into the thread, stop streaming. */
  finishFollowUp: (content: string) => void;
  /** Failure or cancel: drop the partial reply and surface the message. */
  failFollowUp: (error: string | null) => void;

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
  thread: [] as ThreadMessage[],
  reply: "",
  isReplying: false,
  followUpError: null,
} as const;

export const useElaborationStore = create<ElaborationStore>((set) => ({
  ...INITIAL,
  // `INITIAL.thread` is shared; always hand out a fresh array.
  thread: [],

  // Sequential elaborations replace the previous panel content outright.
  openFor: (anchor) =>
    set({
      ...INITIAL,
      thread: [],
      open: true,
      anchor,
      isStreaming: true,
    }),

  openStored: (anchor, elaborationId, content, messages) =>
    set({
      ...INITIAL,
      thread: messages ? [...messages] : [],
      open: true,
      anchor,
      activeElaborationId: elaborationId,
      content,
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

  startFollowUp: (userText) =>
    set((state) => ({
      thread: [...state.thread, { role: "user", content: userText }],
      reply: "",
      isReplying: true,
      followUpError: null,
      runId: null,
    })),

  appendReplyToken: (token) => set((state) => ({ reply: state.reply + token })),

  finishFollowUp: (content) =>
    set((state) => ({
      thread: [...state.thread, { role: "assistant", content }],
      reply: "",
      isReplying: false,
      followUpError: null,
      runId: null,
    })),

  failFollowUp: (error) =>
    set({
      reply: "",
      isReplying: false,
      followUpError: error,
      runId: null,
    }),

  close: () =>
    set({
      open: false,
      isStreaming: false,
      isReplying: false,
      reply: "",
      error: null,
      followUpError: null,
      runId: null,
    }),

  reset: () => set({ ...INITIAL, thread: [] }),
}));
