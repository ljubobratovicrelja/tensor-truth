/**
 * React Query hooks for system status information
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getMemoryInfo,
  getDevices,
  getOllamaStatus,
  restartEngine,
  type MemoryResponse,
  type DevicesResponse,
  type OllamaStatusResponse,
  type RestartEngineResponse,
} from "@/api/system";

/**
 * Hook to fetch comprehensive memory usage
 *
 * Refetches every 5 seconds to keep memory stats up-to-date
 *
 * @param enabled - Whether to enable polling (default: true)
 */
export function useMemoryInfo(enabled = true) {
  return useQuery<MemoryResponse>({
    queryKey: ["system", "memory"],
    queryFn: getMemoryInfo,
    refetchInterval: enabled ? 5000 : false, // Only refresh when enabled
    staleTime: 4000,
    enabled, // Only fetch when enabled
  });
}

/**
 * Hook to fetch available compute devices
 *
 * Devices rarely change, so cache indefinitely
 */
export function useDevices() {
  return useQuery<DevicesResponse>({
    queryKey: ["system", "devices"],
    queryFn: getDevices,
    staleTime: Infinity,
    gcTime: Infinity,
  });
}

/**
 * Hook to fetch Ollama runtime status
 *
 * Refetches every 10 seconds to track running models
 *
 * @param enabled - Whether to enable polling (default: true)
 */
export function useOllamaStatus(enabled = true) {
  return useQuery<OllamaStatusResponse>({
    queryKey: ["system", "ollama", "status"],
    queryFn: getOllamaStatus,
    refetchInterval: enabled ? 10000 : false, // Only refresh when enabled
    staleTime: 8000,
    enabled, // Only fetch when enabled
  });
}

/**
 * Hook to fetch all system status information
 *
 * Returns memory, devices, and Ollama status
 */
export function useSystemStatus() {
  const memory = useMemoryInfo();
  const devices = useDevices();
  const ollama = useOllamaStatus();

  return {
    memory,
    devices,
    ollama,
    isLoading: memory.isLoading || devices.isLoading || ollama.isLoading,
    isError: memory.isError || devices.isError || ollama.isError,
  };
}

/**
 * Hook to restart the RAG engine
 *
 * Clears memory and caches, invalidates memory/ollama queries on success
 */
export function useRestartEngine() {
  const queryClient = useQueryClient();

  return useMutation<RestartEngineResponse, Error>({
    mutationFn: restartEngine,
    onSuccess: () => {
      // Invalidate memory and ollama queries to refresh data after restart
      queryClient.invalidateQueries({ queryKey: ["system", "memory"] });
      queryClient.invalidateQueries({ queryKey: ["system", "ollama", "status"] });
    },
  });
}
