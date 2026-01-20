import { useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import { sendChatMessage } from "@/api/chat";
import { addSessionMessage } from "@/api/sessions";
import type { MessageResponse } from "@/api/types";

interface SendMessageParams {
  sessionId: string;
  message: string;
}

export function useSendMessage() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ sessionId, message }: SendMessageParams) => {
      // Add user message first
      await addSessionMessage(sessionId, {
        role: "user",
        content: message,
      });

      // Send to chat API for response
      const response = await sendChatMessage(sessionId, message);

      // Add assistant response
      const assistantMessage: MessageResponse = {
        role: "assistant",
        content: response.content,
        sources: response.sources,
      };
      await addSessionMessage(sessionId, {
        role: "assistant",
        content: response.content,
      });

      return {
        response,
        assistantMessage,
      };
    },
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.messages(sessionId) });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });
    },
  });
}
