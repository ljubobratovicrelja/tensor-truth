import { useCallback, useRef, useEffect } from "react";
import { useQueryClient, type QueryClient } from "@tanstack/react-query";
import { createWebSocket } from "@/api/client";
import { cancelRun, getActiveRun, getRunEvents, startRun } from "@/api/runs";
import { useChatStore } from "@/stores";
import { QUERY_KEYS } from "@/lib/constants";
import type { RunImagePayload, StreamMessage } from "@/api/types";

export interface AttachedImage {
  id: string;
  file: File;
  previewUrl: string;
  mimetype: string;
}

interface UseWebSocketChatOptions {
  sessionId: string | null;
  onError?: (error: string) => void;
}

/** Number of re-attach attempts after a socket drops mid-run. */
const MAX_RECONNECT_ATTEMPTS = 4;
const RECONNECT_BASE_DELAY_MS = 300;
const RECONNECT_MAX_DELAY_MS = 2000;

/** Mutable per-run accumulator shared by live and replayed events. */
export interface StreamApplyState {
  didFetchUserMessage: boolean;
  didReceiveThinking: boolean;
  fullContent: string;
  confidenceLevel: string;
  /** True once a terminal (`done` / `error`) event has been applied. */
  terminal: boolean;
}

export function createStreamApplyState(): StreamApplyState {
  return {
    didFetchUserMessage: false,
    didReceiveThinking: false,
    fullContent: "",
    confidenceLevel: "normal",
    terminal: false,
  };
}

export interface StreamApplyContext {
  sessionId: string;
  queryClient: QueryClient;
  onError?: (error: string) => void;
}

/**
 * Single pure-ish applier for stream events. Live socket frames and replayed
 * run-log events both go through here, so hydration and live streaming cannot
 * drift apart. Mutates the supplied accumulator, never owns transport state.
 */
export function applyStreamMessage(
  data: StreamMessage,
  state: StreamApplyState,
  ctx: StreamApplyContext
): void {
  const store = useChatStore.getState();

  switch (data.type) {
    case "status":
      store.setStatus(data.status);
      break;

    case "thinking":
      // On first thinking event, clear Phase 1 reasoning so the
      // unified box transitions cleanly to showing synthesis thinking.
      if (!state.didReceiveThinking) {
        state.didReceiveThinking = true;
        store.clearReasoning();
      }
      store.appendThinking(data.content);
      break;

    case "reasoning":
      store.appendReasoning(data.content);
      break;

    case "token":
      // On first token, clear agent reasoning (response generation starting)
      // and fetch messages (backend has saved user message).
      // Don't clear pendingUserMessage here - let MessageList deduplicate
      // to avoid flash when query is refetching
      if (!state.didFetchUserMessage) {
        state.didFetchUserMessage = true;
        store.clearReasoning();
        ctx.queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.messages(ctx.sessionId),
        });
      }
      state.fullContent += data.content;
      store.appendToken(data.content);
      break;

    case "sources":
      store.setSources(data.data, data.source_types);
      if (data.metrics) {
        store.setMetrics(data.metrics);
      }
      break;

    case "done":
      state.terminal = true;
      state.fullContent = data.content;
      state.confidenceLevel = data.confidence_level;
      store.finishStreaming(state.fullContent, state.confidenceLevel, {
        inputTokens: data.input_tokens ?? 0,
        outputTokens: data.output_tokens ?? 0,
      });

      // Refetch messages from server (backend already saved them)
      ctx.queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.messages(ctx.sessionId),
      });
      ctx.queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });
      // The context breakdown was just re-measured on this turn.
      ctx.queryClient.invalidateQueries({
        queryKey: ["session-context", ctx.sessionId],
      });
      break;

    case "error":
      state.terminal = true;
      store.setError(data.detail);
      ctx.onError?.(data.detail);
      break;

    case "title":
      // Title was generated - refresh session data to show new title
      ctx.queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });
      ctx.queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.session(ctx.sessionId),
      });
      break;

    case "tool_progress":
      // New tool starting — clear stale reasoning from previous gap
      if (data.action === "calling") {
        store.clearReasoning();
      }
      store.addToolStep(data);
      break;

    case "tool_phase":
      if (data.phase === "generating") {
        store.clearReasoning();
      }
      store.setToolPhase(data);
      break;

    case "agent_progress":
      store.setAgentProgress(data);
      break;

    case "confirmation_request":
      console.log("[WS] confirmation_request received:", data);
      store.addConfirmationRequest(data);
      break;
  }
}

export function useWebSocketChat({ sessionId, onError }: UseWebSocketChatOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const manualCloseRef = useRef(false);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const runIdRef = useRef<string | null>(null);
  const lastAppliedSeqRef = useRef(0);
  const applyStateRef = useRef<StreamApplyState>(createStreamApplyState());
  const queryClient = useQueryClient();
  // Kept in a ref so a fresh inline `onError` from the consumer never tears
  // down the hydration effect (which would reset the store on every render).
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  const { startStreaming, finishStreaming, setError, reset, isStreaming } =
    useChatStore();

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  /**
   * Apply one event, enforcing the per-run cursor. Returns false when the
   * event was a duplicate (seq <= lastAppliedSeq) and therefore dropped.
   */
  const handleEvent = useCallback(
    (data: StreamMessage & { seq?: number }): boolean => {
      if (!sessionId) return false;
      if (typeof data.seq === "number") {
        if (data.seq <= lastAppliedSeqRef.current) return false;
        lastAppliedSeqRef.current = data.seq;
      }
      applyStreamMessage(data, applyStateRef.current, {
        sessionId,
        queryClient,
        onError: onErrorRef.current,
      });
      return true;
    },
    [sessionId, queryClient]
  );

  /**
   * The first-turn title is generated *after* the answer completes, so its
   * event is appended to the run log after `done` — and the socket closes on
   * `done`. Without this the sidebar kept showing the provisional title until
   * an unrelated refetch. Poll the run log (REST, bounded) until the run
   * reports finished, then apply whatever arrived.
   */
  const settleTitle = useCallback(
    async (runId: string) => {
      for (let attempt = 0; attempt < 8; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 700));
        try {
          const result = await getRunEvents(runId, lastAppliedSeqRef.current);
          for (const ev of result.events ?? []) handleEvent(ev);
          if (result.status !== "running") {
            void queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });
            return;
          }
        } catch (err) {
          console.warn("[WS] title tail fetch failed:", err);
          return;
        }
      }
    },
    [handleEvent, queryClient]
  );

  /**
   * Open a socket and attach to `runId` from `after`. Re-attaches from the
   * current cursor with bounded backoff whenever the socket drops mid-run.
   * A transport failure is never rendered as an error card: the run is still
   * alive server-side, so we retry silently.
   */
  const openRunSocket = useCallback(
    (runId: string, after: number) => {
      if (!sessionId) return;

      clearReconnectTimer();

      if (wsRef.current) {
        const stale = wsRef.current;
        wsRef.current = null;
        stale.onclose = null;
        stale.close();
      }

      const connect = (cursor: number) => {
        const ws = createWebSocket(sessionId);
        wsRef.current = ws;
        manualCloseRef.current = false;

        ws.onopen = () => {
          ws.send(JSON.stringify({ type: "attach", run_id: runId, after: cursor }));
        };

        ws.onmessage = (event) => {
          let data: (StreamMessage & { seq?: number }) | null = null;
          try {
            data = JSON.parse(event.data) as StreamMessage & { seq?: number };
          } catch (error) {
            console.error("Failed to parse WebSocket message:", error);
            return;
          }
          if (!data || typeof data !== "object" || !("type" in data)) return;

          // Any delivered event is evidence the stream is healthy again.
          reconnectAttemptsRef.current = 0;
          handleEvent(data);

          if (data.type === "done" && data.title_pending) {
            void settleTitle(runId);
          }

          if (applyStateRef.current.terminal) {
            manualCloseRef.current = true;
            clearReconnectTimer();
            ws.close();
          }
        };

        // Transport errors surface as onclose next; swallowing them here is
        // what keeps an in-flight run from flashing an error card.
        ws.onerror = () => {};

        ws.onclose = () => {
          if (wsRef.current === ws) {
            wsRef.current = null;
          }
          if (manualCloseRef.current || applyStateRef.current.terminal) return;

          const attempt = reconnectAttemptsRef.current;
          if (attempt >= MAX_RECONNECT_ATTEMPTS) {
            // Bounded retries exhausted. The run keeps going server-side, so
            // catch up over REST rather than declaring the turn failed.
            console.warn(
              `[WS] re-attach failed after ${attempt} attempts; falling back to REST replay`
            );
            void getRunEvents(runId, lastAppliedSeqRef.current)
              .then((result) => {
                for (const ev of result.events ?? []) handleEvent(ev);
                if (result.status !== "running" && !applyStateRef.current.terminal) {
                  finishStreaming(applyStateRef.current.fullContent, "normal");
                }
              })
              .catch((err) => {
                console.warn("[WS] REST catch-up failed:", err);
              });
            return;
          }

          const delay = Math.min(
            RECONNECT_BASE_DELAY_MS * 2 ** attempt,
            RECONNECT_MAX_DELAY_MS
          );
          reconnectAttemptsRef.current = attempt + 1;
          reconnectTimerRef.current = setTimeout(() => {
            reconnectTimerRef.current = null;
            if (manualCloseRef.current || applyStateRef.current.terminal) return;
            connect(lastAppliedSeqRef.current);
          }, delay);
        };
      };

      connect(after);
    },
    [sessionId, clearReconnectTimer, handleEvent, finishStreaming, settleTitle]
  );

  // Hydrate any in-flight run on mount / session change: this is what makes a
  // browser reload seamless (replay the log, then attach live at the cursor).
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    const hydrate = async () => {
      let active;
      try {
        active = await getActiveRun(sessionId);
      } catch (err) {
        console.warn("[WS] active-run lookup failed:", err);
        return;
      }
      if (cancelled) return;

      const run = active;
      if (!run || run.status !== "running") return;
      if (runIdRef.current === run.run_id) return;

      runIdRef.current = run.run_id;
      lastAppliedSeqRef.current = 0;
      applyStateRef.current = createStreamApplyState();
      reconnectAttemptsRef.current = 0;
      manualCloseRef.current = false;

      startStreaming(run.prompt);
      for (const ev of run.events ?? []) {
        handleEvent(ev);
      }
      if (applyStateRef.current.terminal) return;

      openRunSocket(run.run_id, lastAppliedSeqRef.current);
    };

    void hydrate();

    return () => {
      cancelled = true;
      clearReconnectTimer();
      reconnectAttemptsRef.current = 0;
      manualCloseRef.current = true;
      if (wsRef.current) {
        const ws = wsRef.current;
        wsRef.current = null;
        ws.close();
      }
      runIdRef.current = null;
      lastAppliedSeqRef.current = 0;
      applyStateRef.current = createStreamApplyState();
      reset();
    };
  }, [sessionId, handleEvent, openRunSocket, clearReconnectTimer, startStreaming, reset]);

  const sendMessage = useCallback(
    async (message: string, images?: AttachedImage[]) => {
      if (!sessionId) return;

      // Build image refs for optimistic UI (include previewUrl for pending display)
      const imageRefs = images?.map((img) => ({
        id: img.id,
        mimetype: img.mimetype,
        filename: img.file.name,
        previewUrl: img.previewUrl,
      }));

      // Start streaming state with user message for optimistic UI
      startStreaming(message, imageRefs);

      // Convert images to base64 for the run payload
      let imagePayloads: RunImagePayload[] = [];
      if (images && images.length > 0) {
        imagePayloads = await Promise.all(
          images.map(
            (img) =>
              new Promise<RunImagePayload>((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                  const result = reader.result as string;
                  // Strip "data:image/png;base64," prefix
                  const base64 = result.split(",")[1] || result;
                  resolve({
                    data: base64,
                    mimetype: img.mimetype,
                    filename: img.file.name,
                  });
                };
                reader.onerror = reject;
                reader.readAsDataURL(img.file);
              })
          )
        );
      }

      // The turn must exist server-side before any transport is involved: if
      // the socket never connects, the run still completes and can be
      // rehydrated later.
      let runId: string;
      try {
        const result = await startRun(
          sessionId,
          message,
          imagePayloads.length > 0 ? imagePayloads : undefined
        );
        runId = result.run_id;
        if (!runId) throw new Error("Server did not return a run id");
      } catch (err) {
        const detail =
          err instanceof Error ? err.message : "Failed to start the chat run";
        setError(detail);
        onErrorRef.current?.(detail);
        return;
      }

      runIdRef.current = runId;
      lastAppliedSeqRef.current = 0;
      applyStateRef.current = createStreamApplyState();
      reconnectAttemptsRef.current = 0;
      manualCloseRef.current = false;
      openRunSocket(runId, 0);
    },
    [sessionId, startStreaming, setError, openRunSocket]
  );

  const cancelStreaming = useCallback(() => {
    const runId = runIdRef.current;
    // Mark terminal before closing so onclose does not schedule a re-attach.
    applyStateRef.current.terminal = true;
    manualCloseRef.current = true;
    clearReconnectTimer();
    if (runId) {
      void cancelRun(runId).catch((err) => {
        console.warn("[WS] run cancel failed:", err);
      });
    }
    if (wsRef.current) {
      const ws = wsRef.current;
      wsRef.current = null;
      ws.close();
    }
    runIdRef.current = null;
    reset();
  }, [clearReconnectTimer, reset]);

  return {
    sendMessage,
    cancelStreaming,
    isStreaming,
  };
}
