import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  compactSession,
  getSessionContext,
  type CompactResponse,
  type SessionContextResponse,
} from "@/api/sessions";

/** Per-category context breakdown for the status bar. */
export function useSessionContext(sessionId: string | undefined, enabled = true) {
  return useQuery<SessionContextResponse>({
    queryKey: ["session-context", sessionId],
    queryFn: () => getSessionContext(sessionId as string),
    enabled: enabled && !!sessionId,
    // The breakdown only changes when a turn completes; no need to poll hard.
    staleTime: 5_000,
    retry: false,
  });
}

/** "/compact" — summarise older messages to free context. */
export function useCompactSession(sessionId: string | undefined) {
  const queryClient = useQueryClient();
  return useMutation<CompactResponse, Error, { keepLast?: number } | void>({
    mutationFn: (vars) => compactSession(sessionId as string, vars?.keepLast ?? 6),
    onSuccess: (data) => {
      if (data.compacted) {
        toast.success(
          `Compacted ${data.messages_compacted} messages · saved ~${data.tokens_saved.toLocaleString()} tokens`
        );
      } else {
        toast.info(data.message);
      }
      queryClient.invalidateQueries({ queryKey: ["session-context", sessionId] });
      queryClient.invalidateQueries({ queryKey: ["session-stats", sessionId] });
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
    },
    onError: (error) => {
      toast.error(`Compaction failed: ${error.message}`);
    },
  });
}
