import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import {
  listDocuments,
  uploadDocument,
  uploadText,
  uploadUrl,
  deleteDocument,
  reindexDocuments,
  addCatalogModule,
  removeCatalogModule,
} from "@/api/documents";
import type { ScopeType, TextUploadRequest, UrlUploadRequest } from "@/api/types";

export function useDocuments(scopeId: string | null, scopeType: ScopeType) {
  return useQuery({
    queryKey: scopeId
      ? QUERY_KEYS.documents(scopeType, scopeId)
      : [scopeType, "none", "documents"],
    queryFn: () =>
      scopeId
        ? listDocuments(scopeId, scopeType)
        : Promise.resolve({ documents: [], has_index: false }),
    enabled: !!scopeId,
  });
}

export function useUploadDocument() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      scopeId,
      scopeType,
      file,
    }: {
      scopeId: string;
      scopeType: ScopeType;
      file: File;
    }) => uploadDocument(scopeId, scopeType, file),
    onSuccess: (_, { scopeId, scopeType }) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.documents(scopeType, scopeId),
      });
      if (scopeType === "project") {
        queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.project(scopeId),
        });
      }
    },
  });
}

export function useUploadText() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      scopeId,
      scopeType,
      data,
    }: {
      scopeId: string;
      scopeType: ScopeType;
      data: TextUploadRequest;
    }) => uploadText(scopeId, scopeType, data),
    onSuccess: (_, { scopeId, scopeType }) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.documents(scopeType, scopeId),
      });
      if (scopeType === "project") {
        queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.project(scopeId),
        });
      }
    },
  });
}

export function useUploadUrl() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      scopeId,
      scopeType,
      data,
    }: {
      scopeId: string;
      scopeType: ScopeType;
      data: UrlUploadRequest;
    }) => uploadUrl(scopeId, scopeType, data),
    onSuccess: (_, { scopeId, scopeType }) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.documents(scopeType, scopeId),
      });
      if (scopeType === "project") {
        queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.project(scopeId),
        });
      }
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      scopeId,
      scopeType,
      docId,
    }: {
      scopeId: string;
      scopeType: ScopeType;
      docId: string;
    }) => deleteDocument(scopeId, scopeType, docId),
    onSuccess: (_, { scopeId, scopeType }) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.documents(scopeType, scopeId),
      });
      if (scopeType === "project") {
        queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.project(scopeId),
        });
      }
    },
  });
}

export function useReindexDocuments() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ scopeId, scopeType }: { scopeId: string; scopeType: ScopeType }) =>
      reindexDocuments(scopeId, scopeType),
    onSuccess: (_, { scopeId, scopeType }) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.documents(scopeType, scopeId),
      });
      if (scopeType === "project") {
        queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.project(scopeId),
        });
      }
    },
  });
}

export function useAddCatalogModule() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ projectId, moduleName }: { projectId: string; moduleName: string }) =>
      addCatalogModule(projectId, moduleName),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.project(projectId),
      });
    },
  });
}

export function useRemoveCatalogModule() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ projectId, moduleName }: { projectId: string; moduleName: string }) =>
      removeCatalogModule(projectId, moduleName),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.project(projectId),
      });
    },
  });
}
