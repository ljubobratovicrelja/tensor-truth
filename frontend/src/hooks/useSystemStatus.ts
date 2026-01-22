/**
 * React Query hooks for system status information
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getMemoryInfo,
  getDevices,
  getOllamaStatus,
  getRAGStatus,
  restartEngine,
  type MemoryResponse,
  type DevicesResponse,
  type OllamaStatusResponse,
  type RAGStatusResponse,
  type RestartEngineResponse,
} from "@/api/system";

/**
 * Hook to fetch comprehensive memory usage
 *
 * Refetches every second when enabled (panel open) for responsive updates
 *
 * @param enabled - Whether to enable polling (default: true)
 */
export function useMemoryInfo(enabled = true) {
  return useQuery<MemoryResponse>({
    queryKey: ["system", "memory"],
    queryFn: getMemoryInfo,
    refetchInterval: enabled ? 1000 : false,
    staleTime: 800,
    enabled,
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
 * Refetches every second when enabled (panel open) for responsive updates
 *
 * @param enabled - Whether to enable polling (default: true)
 */
export function useOllamaStatus(enabled = true) {
  return useQuery<OllamaStatusResponse>({
    queryKey: ["system", "ollama", "status"],
    queryFn: getOllamaStatus,
    refetchInterval: enabled ? 1000 : false,
    staleTime: 800,
    enabled,
  });
}

/**
 * Hook to fetch RAG system status (embedder and reranker)
 *
 * Refetches every second when enabled (panel open) for responsive updates
 *
 * @param enabled - Whether to enable polling (default: true)
 */
export function useRAGStatus(enabled = true) {
  return useQuery<RAGStatusResponse>({
    queryKey: ["system", "rag", "status"],
    queryFn: getRAGStatus,
    refetchInterval: enabled ? 1000 : false,
    staleTime: 800,
    enabled,
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
