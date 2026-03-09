import { useCallback, useRef, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { createWebSocket } from "@/api/client";
import { useChatStore } from "@/stores";
import { QUERY_KEYS } from "@/lib/constants";
import type { StreamMessage } from "@/api/types";

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
    addToolStep,
    setAgentProgress,
    setToolPhase,
    appendReasoning,
    clearReasoning,
    addApprovalRequest,
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
    async (message: string, images?: AttachedImage[]) => {
      if (!sessionId) return;

      // Close any existing connection
      if (wsRef.current) {
        manualCloseRef.current = true;
        wsRef.current.close();
      }

      // Reset manual close flag for new connection
      manualCloseRef.current = false;

      // Build image refs for optimistic UI (include previewUrl for pending display)
      const imageRefs = images?.map((img) => ({
        id: img.id,
        mimetype: img.mimetype,
        filename: img.file.name,
        previewUrl: img.previewUrl,
      }));

      // Start streaming state with user message for optimistic UI
      startStreaming(message, imageRefs);

      // Convert images to base64 for WS payload
      let imagePayloads: { data: string; mimetype: string; filename: string }[] = [];
      if (images && images.length > 0) {
        imagePayloads = await Promise.all(
          images.map(
            (img) =>
              new Promise<{ data: string; mimetype: string; filename: string }>(
                (resolve, reject) => {
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
                }
              )
          )
        );
      }

      // Create new WebSocket connection
      const ws = createWebSocket(sessionId);
      wsRef.current = ws;

      let fullContent = "";
      let confidenceLevel = "normal";
      let didFetchUserMessage = false;
      let didReceiveThinking = false;

      ws.onopen = () => {
        // Send the prompt (with images if present)
        const payload: Record<string, unknown> = { prompt: message };
        if (imagePayloads.length > 0) {
          payload.images = imagePayloads;
        }
        ws.send(JSON.stringify(payload));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as StreamMessage;

          switch (data.type) {
            case "status":
              setStatus(data.status);
              break;

            case "thinking":
              // On first thinking event, clear Phase 1 reasoning so the
              // unified box transitions cleanly to showing synthesis thinking.
              if (!didReceiveThinking) {
                didReceiveThinking = true;
                clearReasoning();
              }
              appendThinking(data.content);
              break;

            case "reasoning":
              appendReasoning(data.content);
              break;

            case "token":
              // On first token, clear agent reasoning (response generation starting)
              // and fetch messages (backend has saved user message).
              // Don't clear pendingUserMessage here - let MessageList deduplicate
              // to avoid flash when query is refetching
              if (!didFetchUserMessage) {
                didFetchUserMessage = true;
                clearReasoning();
                queryClient.invalidateQueries({
                  queryKey: QUERY_KEYS.messages(sessionId),
                });
              }
              fullContent += data.content;
              appendToken(data.content);
              break;

            case "sources":
              setSources(data.data, data.source_types);
              if (data.metrics) {
                setMetrics(data.metrics);
              }
              break;

            case "done":
              fullContent = data.content;
              confidenceLevel = data.confidence_level;
              finishStreaming(fullContent, confidenceLevel, {
                inputTokens: data.input_tokens ?? 0,
                outputTokens: data.output_tokens ?? 0,
              });

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

            case "tool_progress":
              // New tool starting — clear stale reasoning from previous gap
              if (data.action === "calling") {
                clearReasoning();
              }
              addToolStep(data);
              break;

            case "tool_phase":
              if (data.phase === "generating") {
                clearReasoning();
              }
              setToolPhase(data);
              break;

            case "agent_progress":
              setAgentProgress(data);
              break;

            case "approval_request":
              addApprovalRequest(data);
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
      addToolStep,
      setAgentProgress,
      setToolPhase,
      appendReasoning,
      clearReasoning,
      addApprovalRequest,
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
