import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import {
  getConfig,
  updateConfig,
  getConfigDefaults,
  getModelCapabilities,
} from "@/api/config";

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

/**
 * Hook to check model capabilities (e.g., tool-calling support for agentic mode).
 *
 * @param modelName - The Ollama model name to check. If null/undefined, the query is disabled.
 */
export function useModelCapabilities(modelName: string | null | undefined) {
  return useQuery({
    queryKey: ["model-capabilities", modelName],
    queryFn: () => getModelCapabilities(modelName!),
    enabled: !!modelName,
    // Cache for 5 minutes — model capabilities don't change often
    staleTime: 5 * 60 * 1000,
  });
}
