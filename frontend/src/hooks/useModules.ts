import { useQuery } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import { listModules, listModels, listPresets, listFavoritePresets } from "@/api/modules";

export function useModules() {
  return useQuery({
    queryKey: QUERY_KEYS.modules,
    queryFn: listModules,
  });
}

export function useModels() {
  return useQuery({
    queryKey: QUERY_KEYS.models,
    queryFn: listModels,
  });
}

export function usePresets() {
  return useQuery({
    queryKey: QUERY_KEYS.presets,
    queryFn: listPresets,
  });
}

export function useFavoritePresets() {
  return useQuery({
    queryKey: QUERY_KEYS.favoritePresets,
    queryFn: listFavoritePresets,
  });
}
