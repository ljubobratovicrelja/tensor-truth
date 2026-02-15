import { apiGet, apiPost, apiPatch, apiDelete } from "./client";
import type {
  ProjectCreate,
  ProjectUpdate,
  ProjectResponse,
  ProjectListResponse,
  ProjectSessionCreate,
  SessionResponse,
  SessionListResponse,
} from "./types";

export async function listProjects(): Promise<ProjectListResponse> {
  return apiGet<ProjectListResponse>("/projects");
}

export async function createProject(data: ProjectCreate): Promise<ProjectResponse> {
  return apiPost<ProjectResponse, ProjectCreate>("/projects", data);
}

export async function getProject(projectId: string): Promise<ProjectResponse> {
  return apiGet<ProjectResponse>(`/projects/${projectId}`);
}

export async function updateProject(
  projectId: string,
  data: ProjectUpdate
): Promise<ProjectResponse> {
  return apiPatch<ProjectResponse, ProjectUpdate>(`/projects/${projectId}`, data);
}

export async function deleteProject(projectId: string): Promise<void> {
  return apiDelete(`/projects/${projectId}`);
}

export async function listProjectSessions(
  projectId: string
): Promise<SessionListResponse> {
  return apiGet<SessionListResponse>(`/projects/${projectId}/sessions`);
}

export async function createProjectSession(
  projectId: string,
  data: ProjectSessionCreate = {}
): Promise<SessionResponse> {
  return apiPost<SessionResponse, ProjectSessionCreate>(
    `/projects/${projectId}/sessions`,
    data
  );
}
