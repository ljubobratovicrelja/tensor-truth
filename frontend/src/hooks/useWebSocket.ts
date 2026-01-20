import { useCallback, useRef, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { createWebSocket } from "@/api/client";
import { addSessionMessage } from "@/api/sessions";
import { useChatStore } from "@/stores";
import { QUERY_KEYS } from "@/lib/constants";
import type { StreamMessage } from "@/api/types";

interface UseWebSocketChatOptions {
  sessionId: string | null;
  onError?: (error: string) => void;
}

export function useWebSocketChat({ sessionId, onError }: UseWebSocketChatOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const queryClient = useQueryClient();

  const {
    startStreaming,
    appendToken,
    setSources,
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

      // First, add the user message to the session
      try {
        await addSessionMessage(sessionId, {
          role: "user",
          content: message,
        });
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.messages(sessionId) });
      } catch (error) {
        console.error("Failed to save user message:", error);
        onError?.("Failed to save message");
        return;
      }

      // Close any existing connection
      if (wsRef.current) {
        wsRef.current.close();
      }

      // Start streaming state
      startStreaming();

      // Create new WebSocket connection
      const ws = createWebSocket(sessionId);
      wsRef.current = ws;

      let fullContent = "";
      let confidenceLevel = "normal";

      ws.onopen = () => {
        // Send the prompt
        ws.send(JSON.stringify({ prompt: message }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as StreamMessage;

          switch (data.type) {
            case "token":
              fullContent += data.content;
              appendToken(data.content);
              break;

            case "sources":
              setSources(data.data);
              break;

            case "done":
              fullContent = data.content;
              confidenceLevel = data.confidence_level;
              finishStreaming(fullContent, confidenceLevel);

              // Save the assistant message
              addSessionMessage(sessionId, {
                role: "assistant",
                content: fullContent,
              })
                .then(() => {
                  queryClient.invalidateQueries({
                    queryKey: QUERY_KEYS.messages(sessionId),
                  });
                  queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });
                })
                .catch(console.error);

              ws.close();
              break;

            case "error":
              setError(data.detail);
              onError?.(data.detail);
              ws.close();
              break;
          }
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection error");
        onError?.("Connection error");
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    },
    [
      sessionId,
      queryClient,
      startStreaming,
      appendToken,
      setSources,
      finishStreaming,
      setError,
      onError,
    ]
  );

  const cancelStreaming = useCallback(() => {
    if (wsRef.current) {
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
