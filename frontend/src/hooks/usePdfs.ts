import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import { listPdfs, uploadPdf, deletePdf, reindexPdfs } from "@/api/pdfs";

export function usePdfs(sessionId: string | null) {
  return useQuery({
    queryKey: sessionId ? QUERY_KEYS.pdfs(sessionId) : ["pdfs", "none"],
    queryFn: () =>
      sessionId ? listPdfs(sessionId) : Promise.resolve({ pdfs: [], has_index: false }),
    enabled: !!sessionId,
  });
}

export function useUploadPdf() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ sessionId, file }: { sessionId: string; file: File }) =>
      uploadPdf(sessionId, file),
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.pdfs(sessionId) });
    },
  });
}

export function useDeletePdf() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ sessionId, pdfId }: { sessionId: string; pdfId: string }) =>
      deletePdf(sessionId, pdfId),
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.pdfs(sessionId) });
    },
  });
}

export function useReindexPdfs() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (sessionId: string) => reindexPdfs(sessionId),
    onSuccess: (_, sessionId) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.pdfs(sessionId) });
    },
  });
}
