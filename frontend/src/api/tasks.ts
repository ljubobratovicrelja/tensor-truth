import { apiGet } from "./client";
import type { TaskResponse, TaskListResponse } from "./types";

export async function getTask(taskId: string): Promise<TaskResponse> {
  return apiGet<TaskResponse>(`/tasks/${taskId}`);
}

export async function listTasks(params?: {
  task_type?: string;
  status?: string;
}): Promise<TaskListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.task_type) searchParams.set("task_type", params.task_type);
  if (params?.status) searchParams.set("status", params.status);
  const query = searchParams.toString();
  return apiGet<TaskListResponse>(`/tasks${query ? `?${query}` : ""}`);
}
