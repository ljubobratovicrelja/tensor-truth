import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getExtensions,
  getExtensionLibrary,
  installExtension,
  uninstallExtension,
  reloadExtensions,
} from "@/api/extensions";
import { QUERY_KEYS } from "@/lib/constants";
import type { ExtensionInstallRequest } from "@/api/types";

export function useExtensions() {
  return useQuery({
    queryKey: QUERY_KEYS.extensions,
    queryFn: getExtensions,
  });
}

export function useExtensionLibrary() {
  return useQuery({
    queryKey: QUERY_KEYS.extensionLibrary,
    queryFn: getExtensionLibrary,
  });
}

function useInvalidateAndReload() {
  const queryClient = useQueryClient();
  return async () => {
    await reloadExtensions();
    queryClient.invalidateQueries({ queryKey: QUERY_KEYS.extensions });
    queryClient.invalidateQueries({ queryKey: QUERY_KEYS.extensionLibrary });
    queryClient.invalidateQueries({ queryKey: QUERY_KEYS.mcpServers });
  };
}

export function useInstallExtension() {
  const invalidateAndReload = useInvalidateAndReload();
  return useMutation({
    mutationFn: (request: ExtensionInstallRequest) => installExtension(request),
    onSuccess: () => invalidateAndReload(),
  });
}

export function useUninstallExtension() {
  const invalidateAndReload = useInvalidateAndReload();
  return useMutation({
    mutationFn: ({ type, filename }: { type: string; filename: string }) =>
      uninstallExtension(type, filename),
    onSuccess: () => invalidateAndReload(),
  });
}
