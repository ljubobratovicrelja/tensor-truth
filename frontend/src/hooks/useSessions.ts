import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import {
  listSessions,
  createSession,
  getSession,
  updateSession,
  deleteSession,
  getSessionMessages,
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
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.sessions });
    },
  });
}
