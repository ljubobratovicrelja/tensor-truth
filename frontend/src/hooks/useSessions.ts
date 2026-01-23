import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import {
  listSessions,
  createSession,
  getSession,
  updateSession,
  deleteSession,
  getSessionMessages,
  getSessionStats,
  type SessionStatsResponse,
} from "@/api/sessions";
import type { SessionCreate, SessionUpdate } from "@/api/types";

export function useSessions() {
  return useQuery({
    queryKey: QUERY_KEYS.sessions,
    queryFn: listSessions,
  });
}

export function useSession(sessionId: string | null) {
  return useQuery({
    queryKey: sessionId ? QUERY_KEYS.session(sessionId) : ["sessions", "none"],
    queryFn: () => (sessionId ? getSession(sessionId) : Promise.resolve(null)),
    enabled: !!sessionId,
  });
}

export function useSessionMessages(sessionId: string | null) {
  return useQuery({
    queryKey: sessionId ? QUERY_KEYS.messages(sessionId) : ["messages", "none"],
    queryFn: () =>
      sessionId ? getSessionMessages(sessionId) : Promise.resolve({ messages: [] }),
    enabled: !!sessionId,
    // Always refetch when session changes to avoid showing stale cached data
    // from a previously visited session
    staleTime: 0,
    refetchOnMount: "always",
  });
}

export function useCreateSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: SessionCreate) => createSession(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });
    },
  });
}

export function useUpdateSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ sessionId, data }: { sessionId: string; data: SessionUpdate }) =>
      updateSession(sessionId, data),
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.session(sessionId) });
    },
  });
}

export function useDeleteSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (sessionId: string) => deleteSession(sessionId),
    onSuccess: (_, sessionId) => {
      // Remove session-specific queries to prevent 404 refetch attempts
      queryClient.removeQueries({ queryKey: QUERY_KEYS.session(sessionId) });
      queryClient.removeQueries({ queryKey: QUERY_KEYS.messages(sessionId) });
      // Refresh the sessions list
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });
    },
  });
}

/**
 * Hook to fetch session statistics (messages, chars, model info)
 *
 * Only fetches when sessionId is provided and enabled is true.
 * Polls every second when enabled for real-time updates.
 *
 * @param sessionId - Session ID to fetch stats for (null disables the query)
 * @param enabled - Whether to enable polling (default: true)
 */
export function useSessionStats(sessionId: string | null, enabled = true) {
  return useQuery<SessionStatsResponse>({
    queryKey: ["sessions", sessionId, "stats"],
    queryFn: () => getSessionStats(sessionId!),
    enabled: enabled && !!sessionId,
    refetchInterval: enabled && !!sessionId ? 1000 : false,
    staleTime: 800,
  });
}
