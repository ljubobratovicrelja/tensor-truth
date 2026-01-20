import { apiGet, apiDelete, apiPost, apiPostFormData } from "./client";
import type { PDFListResponse, PDFMetadata, ReindexResponse } from "./types";

export async function listPdfs(sessionId: string): Promise<PDFListResponse> {
  return apiGet<PDFListResponse>(`/sessions/${sessionId}/pdfs`);
}

export async function uploadPdf(sessionId: string, file: File): Promise<PDFMetadata> {
  const formData = new FormData();
  formData.append("file", file);
  return apiPostFormData<PDFMetadata>(`/sessions/${sessionId}/pdfs`, formData);
}

export async function deletePdf(
  sessionId: string,
  pdfId: string
): Promise<{ message: string }> {
  return apiDelete<{ message: string }>(`/sessions/${sessionId}/pdfs/${pdfId}`);
}

export async function reindexPdfs(sessionId: string): Promise<ReindexResponse> {
  return apiPost<ReindexResponse>(`/sessions/${sessionId}/pdfs/reindex`);
}
