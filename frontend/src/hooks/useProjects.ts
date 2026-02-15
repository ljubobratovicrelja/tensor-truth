import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { QUERY_KEYS } from "@/lib/constants";
import {
  listProjects,
  createProject,
  getProject,
  updateProject,
  deleteProject,
  listProjectSessions,
  createProjectSession,
} from "@/api/projects";
import type { ProjectCreate, ProjectUpdate, ProjectSessionCreate } from "@/api/types";

export function useProjects() {
  return useQuery({
    queryKey: QUERY_KEYS.projects,
    queryFn: listProjects,
  });
}

export function useProject(projectId: string | null) {
  return useQuery({
    queryKey: projectId ? QUERY_KEYS.project(projectId) : ["projects", "none"],
    queryFn: () => (projectId ? getProject(projectId) : Promise.resolve(null)),
    enabled: !!projectId,
  });
}

export function useCreateProject() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: ProjectCreate) => createProject(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.projects });
    },
  });
}

export function useUpdateProject() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ projectId, data }: { projectId: string; data: ProjectUpdate }) =>
      updateProject(projectId, data),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.projects });
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.project(projectId),
      });
    },
  });
}

export function useDeleteProject() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (projectId: string) => deleteProject(projectId),
    onSuccess: (_, projectId) => {
      queryClient.removeQueries({
        queryKey: QUERY_KEYS.project(projectId),
      });
      queryClient.removeQueries({
        queryKey: QUERY_KEYS.projectSessions(projectId),
      });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.projects });
    },
  });
}

export function useProjectSessions(projectId: string | null) {
  return useQuery({
    queryKey: projectId
      ? QUERY_KEYS.projectSessions(projectId)
      : ["projects", "none", "sessions"],
    queryFn: () =>
      projectId ? listProjectSessions(projectId) : Promise.resolve({ sessions: [] }),
    enabled: !!projectId,
  });
}

export function useCreateProjectSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      projectId,
      data,
    }: {
      projectId: string;
      data?: ProjectSessionCreate;
    }) => createProjectSession(projectId, data),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.projectSessions(projectId),
      });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.projects });
    },
  });
}
