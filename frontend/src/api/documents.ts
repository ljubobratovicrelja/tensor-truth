import { apiGet, apiPost, apiDelete, apiPostFormData } from "./client";
import type {
  ScopeType,
  ArxivLookupResponse,
  ArxivUploadRequest,
  DocumentListResponse,
  DocumentUploadResponse,
  TextUploadRequest,
  UrlUploadRequest,
  ReindexResponse,
  CatalogModuleAddResponse,
} from "./types";

function scopePrefix(scopeType: ScopeType, scopeId: string): string {
  return `/${scopeType}s/${scopeId}/documents`;
}

export async function listDocuments(
  scopeId: string,
  scopeType: ScopeType
): Promise<DocumentListResponse> {
  return apiGet<DocumentListResponse>(scopePrefix(scopeType, scopeId));
}

export async function uploadDocument(
  scopeId: string,
  scopeType: ScopeType,
  file: File
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return apiPostFormData<DocumentUploadResponse>(
    `${scopePrefix(scopeType, scopeId)}/upload`,
    formData
  );
}

export async function uploadText(
  scopeId: string,
  scopeType: ScopeType,
  data: TextUploadRequest
): Promise<DocumentUploadResponse> {
  return apiPost<DocumentUploadResponse, TextUploadRequest>(
    `${scopePrefix(scopeType, scopeId)}/upload-text`,
    data
  );
}

export async function uploadUrl(
  scopeId: string,
  scopeType: ScopeType,
  data: UrlUploadRequest
): Promise<DocumentUploadResponse> {
  return apiPost<DocumentUploadResponse, UrlUploadRequest>(
    `${scopePrefix(scopeType, scopeId)}/upload-url`,
    data
  );
}

export async function deleteDocument(
  scopeId: string,
  scopeType: ScopeType,
  docId: string
): Promise<void> {
  return apiDelete(`${scopePrefix(scopeType, scopeId)}/${docId}`);
}

export async function reindexDocuments(
  scopeId: string,
  scopeType: ScopeType
): Promise<ReindexResponse> {
  return apiPost<ReindexResponse>(`${scopePrefix(scopeType, scopeId)}/reindex`);
}

export async function lookupArxiv(arxivId: string): Promise<ArxivLookupResponse> {
  return apiGet<ArxivLookupResponse>(`/arxiv/${arxivId}`);
}

export async function uploadArxiv(
  scopeId: string,
  scopeType: ScopeType,
  data: ArxivUploadRequest
): Promise<DocumentUploadResponse> {
  return apiPost<DocumentUploadResponse, ArxivUploadRequest>(
    `${scopePrefix(scopeType, scopeId)}/upload-arxiv`,
    data
  );
}

export async function addCatalogModule(
  projectId: string,
  moduleName: string
): Promise<CatalogModuleAddResponse> {
  return apiPost<CatalogModuleAddResponse>(`/projects/${projectId}/catalog-modules`, {
    module_name: moduleName,
  });
}

export async function removeCatalogModule(
  projectId: string,
  moduleName: string
): Promise<void> {
  // The backend returns a JSON body, but we don't need it.
  // apiDelete handles error checking, and we discard the response body.
  return apiDelete(`/projects/${projectId}/catalog-modules/${moduleName}`);
}
