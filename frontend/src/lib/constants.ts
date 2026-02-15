export const QUERY_KEYS = {
  sessions: ["sessions"] as const,
  session: (id: string) => ["sessions", id] as const,
  messages: (sessionId: string) => ["sessions", sessionId, "messages"] as const,
  modules: ["modules"] as const,
  models: ["models"] as const,
  embeddingModels: ["embedding-models"] as const,
  rerankers: ["rerankers"] as const,
  config: ["config"] as const,
  documents: (scopeType: string, scopeId: string) =>
    [scopeType, scopeId, "documents"] as const,
  startup: ["startup"] as const,
  projects: ["projects"] as const,
  project: (id: string) => ["projects", id] as const,
  projectSessions: (projectId: string) => ["projects", projectId, "sessions"] as const,
  task: (id: string) => ["tasks", id] as const,
  tasks: ["tasks"] as const,
};
