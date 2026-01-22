/**
 * Hook for fetching and caching available commands
 *
 * Uses React Query to fetch command definitions from the backend.
 * Commands are cached indefinitely since they rarely change.
 */

import { useQuery } from "@tanstack/react-query";
import { getCommands } from "@/api/commands";
import type { CommandDefinition } from "@/types/commands";

/**
 * Fetch all available commands from the backend
 *
 * The query is cached indefinitely since commands rarely change during a session.
 * The cache is invalidated on page reload.
 *
 * @returns React Query result with commands array
 */
export function useCommands() {
  return useQuery<CommandDefinition[]>({
    queryKey: ["commands"],
    queryFn: getCommands,
    staleTime: Infinity, // Commands don't change during session
    gcTime: Infinity, // Keep in cache forever (was: cacheTime)
    refetchOnWindowFocus: false, // Don't refetch on window focus
    refetchOnMount: false, // Don't refetch on component mount after initial load
  });
}
