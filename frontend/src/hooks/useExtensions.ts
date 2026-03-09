import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getExtensions,
  getExtensionLibrary,
  installExtension,
  uninstallExtension,
  reloadExtensions,
} from "@/api/extensions";
import { QUERY_KEYS } from "@/lib/constants";
import type { ExtensionInstallRequest, ExtensionListResponse } from "@/api/types";

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
  const queryClient = useQueryClient();
  const invalidateAndReload = useInvalidateAndReload();
  return useMutation({
    mutationFn: ({ type, filename }: { type: string; filename: string }) =>
      uninstallExtension(type, filename),
    onMutate: async ({ type, filename }) => {
      await queryClient.cancelQueries({ queryKey: QUERY_KEYS.extensions });
      const previous = queryClient.getQueryData(QUERY_KEYS.extensions);
      queryClient.setQueryData<ExtensionListResponse | undefined>(
        QUERY_KEYS.extensions,
        (old) => {
          if (!old?.extensions) return old;
          return {
            ...old,
            extensions: old.extensions.filter(
              (e) => !(e.type === type && e.filename === filename)
            ),
          };
        }
      );
      return { previous };
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData(QUERY_KEYS.extensions, context.previous);
      }
    },
    onSettled: () => invalidateAndReload(),
  });
}
