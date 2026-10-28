import { useCallback, useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { createWebSocket } from "@/api/client";
import { cancelRun, getRunEvents, startRun } from "@/api/runs";
import { QUERY_KEYS } from "@/lib/constants";
import { useElaborationStore } from "@/stores/elaborationStore";
import type { ElaborationAnchor } from "@/stores/elaborationStore";
import type { StreamMessage } from "@/api/types";

/** Bounded re-attach policy for a socket that drops mid-run. */
const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_BASE_DELAY_MS = 300;
const RECONNECT_MAX_DELAY_MS = 1500;

/**
 * Streams one elaboration into the dedicated elaboration store.
 *
 * Deliberately independent of `useChatStore`: an explanation must never touch
 * the main chat's streaming state.
 */
export function useElaboration(sessionId: string | null) {
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const runIdRef = useRef<string | null>(null);
  const lastSeqRef = useRef(0);
  const contentRef = useRef("");
  const terminalRef = useRef(false);
  const manualCloseRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const closeSocket = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onclose = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.close();
    }
  }, []);

  const applyEvent = useCallback(
    (event: StreamMessage & { seq?: number }): boolean => {
      if (typeof event.seq === "number") {
        if (event.seq <= lastSeqRef.current) return false;
        lastSeqRef.current = event.seq;
      }
      const store = useElaborationStore.getState();

      switch (event.type) {
        case "token":
          contentRef.current += event.content;
          store.appendToken(event.content);
          return true;
        case "thinking":
        case "reasoning":
          store.appendThinking(event.content);
          return true;
        case "done": {
          terminalRef.current = true;
          const content = event.content || contentRef.current;
          contentRef.current = content;
          store.finish(content, event.elaboration_id ?? null);
          if (sessionId) {
            const key = QUERY_KEYS.messages(sessionId);
            const invalidate = () => {
              void queryClient.invalidateQueries({ queryKey: key });
            };
            invalidate();
            // `done` is emitted before the runner persists the elaboration, so
            // one immediate refetch can race the write. A second, later pass
            // makes the underline appear without a manual reload.
            window.setTimeout(invalidate, 800);
          }
          return true;
        }
        case "error":
          terminalRef.current = true;
          store.setError(event.detail);
          return true;
        default:
          // sources/tool/status events carry nothing the panel renders.
          return true;
      }
    },
    [queryClient, sessionId]
  );

  /** Bounded REST replay once socket re-attach attempts are exhausted. */
  const drainRest = useCallback(
    async (runId: string) => {
      try {
        const result = await getRunEvents(runId, lastSeqRef.current);
        for (const event of result.events ?? []) applyEvent(event);
        if (result.status !== "running" && !terminalRef.current) {
          terminalRef.current = true;
          useElaborationStore.getState().finish(contentRef.current, null);
        }
      } catch (err) {
        console.warn("[elaboration] REST catch-up failed:", err);
        useElaborationStore
          .getState()
          .setError("Lost connection while explaining this passage.");
      }
    },
    [applyEvent]
  );

  const openSocket = useCallback(
    (runId: string, after: number) => {
      if (!sessionId) return;

      clearReconnectTimer();
      closeSocket();

      const connect = (cursor: number) => {
        const ws = createWebSocket(sessionId);
        wsRef.current = ws;
        manualCloseRef.current = false;

        ws.onopen = () => {
          ws.send(JSON.stringify({ type: "attach", run_id: runId, after: cursor }));
        };

        ws.onmessage = (message) => {
          let event: (StreamMessage & { seq?: number }) | null = null;
          try {
            event = JSON.parse(message.data) as StreamMessage & { seq?: number };
          } catch (err) {
            console.error("[elaboration] bad event:", err);
            return;
          }
          if (!event || typeof event !== "object" || !("type" in event)) return;

          reconnectAttemptsRef.current = 0;
          applyEvent(event);

          if (terminalRef.current) {
            manualCloseRef.current = true;
            clearReconnectTimer();
            ws.close();
          }
        };

        // The run is server-side; a transport hiccup is retried, not surfaced.
        ws.onerror = () => {};

        ws.onclose = () => {
          if (wsRef.current === ws) wsRef.current = null;
          if (manualCloseRef.current || terminalRef.current) return;

          const attempt = reconnectAttemptsRef.current;
          if (attempt >= MAX_RECONNECT_ATTEMPTS) {
            void drainRest(runId);
            return;
          }
          const delay = Math.min(
            RECONNECT_BASE_DELAY_MS * 2 ** attempt,
            RECONNECT_MAX_DELAY_MS
          );
          reconnectAttemptsRef.current = attempt + 1;
          reconnectTimerRef.current = setTimeout(() => {
            reconnectTimerRef.current = null;
            if (manualCloseRef.current || terminalRef.current) return;
            connect(lastSeqRef.current);
          }, delay);
        };
      };

      connect(after);
    },
    [sessionId, clearReconnectTimer, closeSocket, applyEvent, drainRest]
  );

  const start = useCallback(
    async (anchor: ElaborationAnchor) => {
      if (!sessionId) return;

      clearReconnectTimer();
      closeSocket();
      runIdRef.current = null;
      lastSeqRef.current = 0;
      contentRef.current = "";
      terminalRef.current = false;
      reconnectAttemptsRef.current = 0;

      // Open first so the panel appears (and streams) immediately.
      useElaborationStore.getState().openFor(anchor);

      try {
        const { run_id: runId } = await startRun(
          sessionId,
          anchor.selectedText,
          undefined,
          {
            kind: "elaboration",
            message_index: anchor.messageIndex,
            selected_text: anchor.selectedText,
            occurrence_index: anchor.occurrenceIndex,
          }
        );
        if (!runId) throw new Error("Server did not return a run id");

        runIdRef.current = runId;
        useElaborationStore.getState().setRunId(runId);
        openSocket(runId, 0);
      } catch (err) {
        const detail =
          err instanceof Error ? err.message : "Failed to start the explanation";
        useElaborationStore.getState().setError(detail);
      }
    },
    [sessionId, clearReconnectTimer, closeSocket, openSocket]
  );

  const cancel = useCallback(() => {
    const runId = runIdRef.current ?? useElaborationStore.getState().runId;
    terminalRef.current = true;
    manualCloseRef.current = true;
    clearReconnectTimer();
    closeSocket();
    runIdRef.current = null;

    if (runId) {
      void cancelRun(runId).catch((err) => {
        console.warn("[elaboration] cancel failed:", err);
      });
    }

    const store = useElaborationStore.getState();
    if (store.isStreaming) {
      // Keep whatever was streamed so far; nothing was persisted.
      store.finish(contentRef.current, store.activeElaborationId);
    }
  }, [clearReconnectTimer, closeSocket]);

  // Session switches and unmounts must not leave a panel bound to a stale run.
  useEffect(() => {
    return () => {
      terminalRef.current = true;
      manualCloseRef.current = true;
      clearReconnectTimer();
      closeSocket();
      runIdRef.current = null;
      lastSeqRef.current = 0;
      contentRef.current = "";
      reconnectAttemptsRef.current = 0;
      useElaborationStore.getState().reset();
    };
  }, [sessionId, clearReconnectTimer, closeSocket]);

  return { start, cancel };
}
