import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import {
  getProviders,
  addProvider,
  updateProvider,
  removeProvider,
  testProviderUrl,
  discoverServers,
} from "@/api/providers";
import type {
  ProviderCreateRequest,
  ProviderUpdateRequest,
  ProviderTestRequest,
} from "@/api/types";

export function useProviders() {
  return useQuery({
    queryKey: QUERY_KEYS.providers,
    queryFn: getProviders,
  });
}

export function useAddProvider() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ProviderCreateRequest) => addProvider(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.providers });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.models });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.startup });
    },
  });
}

export function useUpdateProvider() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, request }: { id: string; request: ProviderUpdateRequest }) =>
      updateProvider(id, request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.providers });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.models });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.startup });
    },
  });
}

export function useRemoveProvider() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => removeProvider(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.providers });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.models });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.startup });
    },
  });
}

export function useTestProviderUrl() {
  return useMutation({
    mutationFn: (request: ProviderTestRequest) => testProviderUrl(request),
  });
}

export function useDiscoverServers() {
  return useMutation({
    mutationFn: () => discoverServers(),
  });
}
