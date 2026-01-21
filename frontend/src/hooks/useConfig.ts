import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import { getConfig, updateConfig, getConfigDefaults } from "@/api/config";

export function useConfig() {
  return useQuery({
    queryKey: QUERY_KEYS.config,
    queryFn: getConfig,
  });
}

export function useConfigDefaults() {
  return useQuery({
    queryKey: [...QUERY_KEYS.config, "defaults"],
    queryFn: getConfigDefaults,
  });
}

export function useUpdateConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (updates: Record<string, unknown>) => updateConfig(updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.config });
      // Also invalidate modules since they depend on configured embedding model
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.modules });
    },
  });
}
