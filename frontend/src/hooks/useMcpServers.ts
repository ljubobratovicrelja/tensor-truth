import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getMcpServers,
  getMcpServerPresets,
  addMcpServer,
  updateMcpServer,
  deleteMcpServer,
  toggleMcpServer,
} from "@/api/mcp-servers";
import { reloadExtensions } from "@/api/extensions";
import { QUERY_KEYS } from "@/lib/constants";
import type {
  MCPServerCreateRequest,
  MCPServerListResponse,
  MCPServerUpdateRequest,
} from "@/api/types";

export function useMcpServers() {
  return useQuery({
    queryKey: QUERY_KEYS.mcpServers,
    queryFn: getMcpServers,
  });
}

export function useMcpServerPresets() {
  return useQuery({
    queryKey: [...QUERY_KEYS.mcpServers, "presets"],
    queryFn: getMcpServerPresets,
  });
}

function useInvalidateAndReload() {
  const queryClient = useQueryClient();
  return async () => {
    await reloadExtensions();
    queryClient.invalidateQueries({ queryKey: QUERY_KEYS.mcpServers });
    queryClient.invalidateQueries({ queryKey: QUERY_KEYS.extensions });
    queryClient.invalidateQueries({ queryKey: QUERY_KEYS.extensionLibrary });
  };
}

export function useAddMcpServer() {
  const invalidateAndReload = useInvalidateAndReload();
  return useMutation({
    mutationFn: (request: MCPServerCreateRequest) => addMcpServer(request),
    onSuccess: () => invalidateAndReload(),
  });
}

export function useUpdateMcpServer() {
  const invalidateAndReload = useInvalidateAndReload();
  return useMutation({
    mutationFn: ({ name, request }: { name: string; request: MCPServerUpdateRequest }) =>
      updateMcpServer(name, request),
    onSuccess: () => invalidateAndReload(),
  });
}

export function useDeleteMcpServer() {
  const queryClient = useQueryClient();
  const invalidateAndReload = useInvalidateAndReload();
  return useMutation({
    mutationFn: (name: string) => deleteMcpServer(name),
    onMutate: async (name) => {
      await queryClient.cancelQueries({ queryKey: QUERY_KEYS.mcpServers });
      const previous = queryClient.getQueryData(QUERY_KEYS.mcpServers);
      queryClient.setQueryData<MCPServerListResponse | undefined>(
        QUERY_KEYS.mcpServers,
        (old) => {
          if (!old?.servers) return old;
          return {
            ...old,
            servers: old.servers.filter((s) => s.name !== name),
          };
        }
      );
      return { previous };
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData(QUERY_KEYS.mcpServers, context.previous);
      }
    },
    onSettled: () => invalidateAndReload(),
  });
}

export function useToggleMcpServer() {
  const invalidateAndReload = useInvalidateAndReload();
  return useMutation({
    mutationFn: ({ name, enabled }: { name: string; enabled: boolean }) =>
      toggleMcpServer(name, { enabled }),
    onSuccess: () => invalidateAndReload(),
  });
}
