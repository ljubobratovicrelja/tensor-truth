import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import {
  getStartupStatus,
  downloadIndexes,
  pullModel,
  reinitializeIndexes,
} from "@/api/startup";
import type { IndexDownloadRequest, ModelPullRequest } from "@/api/types";

export function useStartupStatus(options?: { pollingInterval?: number }) {
  const pollingInterval = options?.pollingInterval;

  return useQuery({
    queryKey: QUERY_KEYS.startup,
    queryFn: getStartupStatus,
    refetchInterval: (query) => {
      const data = query.state.data;
      // Use custom polling interval if provided, otherwise default to 5s
      if (data && (!data.indexes_ok || !data.models_ok)) {
        return pollingInterval ?? 5000;
      }
      return false; // Stop polling when all resources are ready
    },
  });
}

export function useDownloadIndexes() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request?: IndexDownloadRequest) => downloadIndexes(request),
    onSuccess: () => {
      // Invalidate startup status to trigger re-check
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.startup });
    },
  });
}

export function usePullModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ModelPullRequest) => pullModel(request),
    onSuccess: () => {
      // Invalidate both startup status and models queries
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.startup });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.models });
    },
  });
}

export function useReinitializeIndexes() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => reinitializeIndexes(),
    onSuccess: () => {
      // Invalidate startup status and modules to trigger re-check
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.startup });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.modules });
    },
  });
}
