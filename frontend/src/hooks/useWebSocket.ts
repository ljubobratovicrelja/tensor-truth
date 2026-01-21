import { useCallback, useRef, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { createWebSocket } from "@/api/client";
import { useChatStore } from "@/stores";
import { QUERY_KEYS } from "@/lib/constants";
import type { StreamMessage } from "@/api/types";

interface UseWebSocketChatOptions {
  sessionId: string | null;
  onError?: (error: string) => void;
}

export function useWebSocketChat({ sessionId, onError }: UseWebSocketChatOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const manualCloseRef = useRef(false);
  const queryClient = useQueryClient();

  const {
    startStreaming,
    appendToken,
    appendThinking,
    setStatus,
    setSources,
    setMetrics,
    finishStreaming,
    setError,
    reset,
    isStreaming,
  } = useChatStore();

  // Cleanup on unmount or session change
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      reset();
    };
  }, [sessionId, reset]);

  const sendMessage = useCallback(
    async (message: string) => {
      if (!sessionId) return;

      // Close any existing connection
      if (wsRef.current) {
        manualCloseRef.current = true;
        wsRef.current.close();
      }

      // Reset manual close flag for new connection
      manualCloseRef.current = false;

      // Start streaming state with user message for optimistic UI
      startStreaming(message);

      // Create new WebSocket connection
      const ws = createWebSocket(sessionId);
      wsRef.current = ws;

      let fullContent = "";
      let confidenceLevel = "normal";
      let didFetchUserMessage = false;

      ws.onopen = () => {
        // Send the prompt
        ws.send(JSON.stringify({ prompt: message }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as StreamMessage;

          switch (data.type) {
            case "status":
              setStatus(data.status);
              break;

            case "thinking":
              appendThinking(data.content);
              break;

            case "token":
              // On first token, fetch messages (backend has saved user message)
              // Don't clear pendingUserMessage here - let MessageList deduplicate
              // to avoid flash when query is refetching
              if (!didFetchUserMessage) {
                didFetchUserMessage = true;
                queryClient.invalidateQueries({
                  queryKey: QUERY_KEYS.messages(sessionId),
                });
              }
              fullContent += data.content;
              appendToken(data.content);
              break;

            case "sources":
              setSources(data.data);
              if (data.metrics) {
                setMetrics(data.metrics);
              }
              break;

            case "done":
              fullContent = data.content;
              confidenceLevel = data.confidence_level;
              finishStreaming(fullContent, confidenceLevel);

              // Refetch messages from server (backend already saved them)
              queryClient.invalidateQueries({
                queryKey: QUERY_KEYS.messages(sessionId),
              });
              queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });

              // Close immediately if no title pending, otherwise wait for title
              if (!data.title_pending) {
                manualCloseRef.current = true; // Suppress error on clean close
                ws.close();
              }
              break;

            case "error":
              setError(data.detail);
              onError?.(data.detail);
              manualCloseRef.current = true; // We already showed the error
              ws.close();
              break;

            case "title":
              // Title was generated - refresh session data to show new title
              queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });
              queryClient.invalidateQueries({
                queryKey: QUERY_KEYS.session(sessionId),
              });
              // Now we can close the connection
              manualCloseRef.current = true; // Suppress error on clean close
              ws.close();
              break;
          }
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      ws.onerror = () => {
        // Only report error if not a manual close
        if (!manualCloseRef.current) {
          setError("WebSocket connection error");
          onError?.("Connection error");
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
        manualCloseRef.current = false;
      };
    },
    [
      sessionId,
      queryClient,
      startStreaming,
      appendToken,
      appendThinking,
      setStatus,
      setSources,
      setMetrics,
      finishStreaming,
      setError,
      onError,
    ]
  );

  const cancelStreaming = useCallback(() => {
    if (wsRef.current) {
      manualCloseRef.current = true;
      wsRef.current.close();
      wsRef.current = null;
    }
    reset();
  }, [reset]);

  return {
    sendMessage,
    cancelStreaming,
    isStreaming,
  };
}
