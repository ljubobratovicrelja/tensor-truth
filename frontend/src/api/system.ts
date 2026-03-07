/**
 * System information API client
 */

import { apiGet, apiPost } from "./client";

export interface MemoryInfo {
  name: string;
  allocated_gb: number;
  total_gb: number | null;
  details: string | null;
}

export interface MemoryResponse {
  memory: MemoryInfo[];
}

export interface DevicesResponse {
  devices: string[];
}

export interface OllamaModelInfo {
  name: string;
  size_vram_gb: number;
  size_gb: number;
  parameters: string | null;
  context_length: number | null;
}

export interface OllamaStatusResponse {
  running: boolean;
  models: OllamaModelInfo[];
  info_lines: string[];
}

export interface RAGModelStatus {
  loaded: boolean;
  model_name: string | null;
  device: string | null;
  memory_gb: number | null;
}

export interface RAGStatusResponse {
  active: boolean;
  embedder: RAGModelStatus;
  reranker: RAGModelStatus;
  total_memory_gb: number;
}

/**
 * Get comprehensive memory usage across all components
 */
export async function getMemoryInfo(): Promise<MemoryResponse> {
  return apiGet<MemoryResponse>("/system/memory");
}

/**
 * Get list of available compute devices
 */
export async function getDevices(): Promise<DevicesResponse> {
  return apiGet<DevicesResponse>("/system/devices");
}

/**
 * Get Ollama runtime status and running models
 */
export async function getOllamaStatus(): Promise<OllamaStatusResponse> {
  return apiGet<OllamaStatusResponse>("/system/ollama/status");
}

/**
 * Get RAG system status (embedder and reranker)
 */
export async function getRAGStatus(): Promise<RAGStatusResponse> {
  return apiGet<RAGStatusResponse>("/system/rag/status");
}

export interface LlamaCppModelInfo {
  name: string;
  display_name: string;
  status: string;
}

export interface LlamaCppStatusResponse {
  running: boolean;
  models: LlamaCppModelInfo[];
  base_url: string;
}

export interface LlamaCppActionResponse {
  success: boolean;
  message: string;
}

/**
 * Get llama.cpp runtime status and available models
 */
export async function getLlamaCppStatus(): Promise<LlamaCppStatusResponse> {
  return apiGet<LlamaCppStatusResponse>("/system/llama-cpp/status");
}

/**
 * Load a model on the llama.cpp server
 */
export async function loadLlamaCppModel(
  model: string,
  providerId?: string
): Promise<LlamaCppActionResponse> {
  return apiPost<LlamaCppActionResponse>("/system/llama-cpp/load", {
    model,
    provider_id: providerId,
  });
}

/**
 * Unload a model from the llama.cpp server
 */
export async function unloadLlamaCppModel(
  model: string,
  providerId?: string
): Promise<LlamaCppActionResponse> {
  return apiPost<LlamaCppActionResponse>("/system/llama-cpp/unload", {
    model,
    provider_id: providerId,
  });
}

export interface RestartEngineResponse {
  success: boolean;
  message: string;
}

/**
 * Restart the RAG engine by clearing memory and caches
 */
export async function restartEngine(): Promise<RestartEngineResponse> {
  return apiPost<RestartEngineResponse>("/system/restart-engine");
}
