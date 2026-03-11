import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import {
  getSources,
  getModulesOverview,
  addSource,
  updateSource,
  deleteSource,
  fetchSource,
  buildIndex,
  batchBuild,
  deleteIndex,
  getSourceDocuments,
  getDocumentContent,
} from "@/api/index-management";
import type {
  SourceCreateRequest,
  FetchRequest,
  BuildRequest,
  BatchBuildRequest,
} from "@/api/types";

export function useSources() {
  return useQuery({
    queryKey: QUERY_KEYS.sources,
    queryFn: getSources,
  });
}

export function useModulesOverview(embeddingModelId?: string) {
  return useQuery({
    queryKey: [...QUERY_KEYS.modulesOverview, embeddingModelId],
    queryFn: () => getModulesOverview(embeddingModelId),
  });
}

export function useAddSource() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: SourceCreateRequest) => addSource(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEYS.sources });
      qc.invalidateQueries({ queryKey: QUERY_KEYS.modulesOverview });
    },
  });
}

export function useUpdateSource() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      sourceType: string;
      name: string;
      config: Record<string, unknown>;
    }) => updateSource(data.sourceType, data.name, data.config),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEYS.sources });
      qc.invalidateQueries({ queryKey: QUERY_KEYS.modulesOverview });
    },
  });
}

export function useDeleteSource() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { sourceType: string; name: string }) =>
      deleteSource(data.sourceType, data.name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEYS.sources });
      qc.invalidateQueries({ queryKey: QUERY_KEYS.modulesOverview });
    },
  });
}

export function useFetchSource() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: FetchRequest) => fetchSource(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEYS.modulesOverview });
    },
  });
}

export function useBuildModuleIndex() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: BuildRequest) => buildIndex(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEYS.modulesOverview });
      qc.invalidateQueries({ queryKey: QUERY_KEYS.modules });
    },
  });
}

export function useBatchBuild() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: BatchBuildRequest) => batchBuild(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEYS.modulesOverview });
      qc.invalidateQueries({ queryKey: QUERY_KEYS.modules });
    },
  });
}

export function useSourceDocuments(
  embeddingModelId: string,
  moduleName: string,
  enabled: boolean
) {
  return useQuery({
    queryKey: QUERY_KEYS.sourceDocuments(embeddingModelId, moduleName),
    queryFn: () => getSourceDocuments(embeddingModelId, moduleName),
    enabled,
  });
}

export function useDocumentContent(
  embeddingModelId: string,
  moduleName: string,
  docId: string,
  enabled: boolean
) {
  return useQuery({
    queryKey: QUERY_KEYS.documentContent(embeddingModelId, moduleName, docId),
    queryFn: () => getDocumentContent(embeddingModelId, moduleName, docId),
    enabled,
  });
}

export function useDeleteIndex() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { embeddingModelId: string; moduleName: string }) =>
      deleteIndex(data.embeddingModelId, data.moduleName),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QUERY_KEYS.modulesOverview });
      qc.invalidateQueries({ queryKey: QUERY_KEYS.modules });
    },
  });
}
