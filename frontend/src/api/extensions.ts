import { apiDelete, apiGet, apiPost } from "./client";
import type {
  ExtensionListResponse,
  ExtensionLibraryResponse,
  ExtensionInstallRequest,
  ExtensionInstallResponse,
  ReloadExtensionsResponse,
} from "./types";

export async function getExtensions(): Promise<ExtensionListResponse> {
  return apiGet<ExtensionListResponse>("/extensions/");
}

export async function getExtensionLibrary(): Promise<ExtensionLibraryResponse> {
  return apiGet<ExtensionLibraryResponse>("/extensions/library");
}

export async function installExtension(
  request: ExtensionInstallRequest
): Promise<ExtensionInstallResponse> {
  return apiPost<ExtensionInstallResponse, ExtensionInstallRequest>(
    "/extensions/install",
    request
  );
}

export async function uninstallExtension(type: string, filename: string): Promise<void> {
  return apiDelete(
    `/extensions/${encodeURIComponent(type)}/${encodeURIComponent(filename)}`
  );
}

export async function reloadExtensions(): Promise<ReloadExtensionsResponse> {
  return apiPost<ReloadExtensionsResponse>("/reload-extensions");
}
