/**
 * System information API client
 */

import { apiGet } from "./client";

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
}

export interface OllamaStatusResponse {
  running: boolean;
  models: OllamaModelInfo[];
  info_lines: string[];
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

export interface RestartEngineResponse {
  success: boolean;
  message: string;
}

/**
 * Restart the RAG engine by clearing memory and caches
 */
export async function restartEngine(): Promise<RestartEngineResponse> {
  const response = await fetch(
    `${import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"}/api/system/restart-engine`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to restart engine: ${response.statusText}`);
  }

  return response.json();
}
