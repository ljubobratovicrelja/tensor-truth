// Session types
export interface SessionCreate {
  modules?: string[] | null;
  params?: Record<string, unknown>;
}

export interface SessionResponse {
  session_id: string;
  title: string;
  created_at: string;
  modules?: string[] | null;
  params: Record<string, unknown>;
  message_count: number;
}

export interface SessionListResponse {
  sessions: SessionResponse[];
  current_id?: string | null;
}

export interface SessionUpdate {
  title?: string | null;
  params?: Record<string, unknown> | null;
}

// Message types
export interface MessageCreate {
  role: "user" | "assistant";
  content: string;
}

export interface MessageResponse {
  role: string;
  content: string;
  sources?: SourceNode[] | null;
}

export interface MessagesResponse {
  messages: MessageResponse[];
}

// Chat types
export interface ChatRequest {
  prompt: string;
}

export interface SourceNode {
  text: string;
  score?: number | null;
  metadata: Record<string, unknown>;
}

export interface ChatResponse {
  content: string;
  sources: SourceNode[];
  confidence_level: string;
}

// WebSocket message types
export interface StreamToken {
  type: "token";
  content: string;
}

export interface StreamSources {
  type: "sources";
  data: SourceNode[];
}

export interface StreamDone {
  type: "done";
  content: string;
  confidence_level: string;
  title_pending?: boolean;
}

export interface StreamError {
  type: "error";
  detail: string;
}

export interface StreamTitle {
  type: "title";
  title: string;
}

export type StreamMessage = StreamToken | StreamSources | StreamDone | StreamError | StreamTitle;

// Intent types
export interface IntentRequest {
  message: string;
  recent_messages?: Array<{ role: string; content: string }>;
}

export interface IntentResponse {
  intent: "chat" | "browse" | "search";
  query?: string | null;
  reason: string;
}

// Config types
export interface OllamaConfig {
  base_url: string;
  timeout: number;
}

export interface UIConfig {
  default_temperature: number;
  default_context_window: number;
  default_max_tokens: number;
  default_reranker: string;
  default_top_n: number;
  default_confidence_threshold: number;
  default_confidence_cutoff_hard: number;
}

export interface RAGConfig {
  default_device: string;
}

export interface ModelsConfig {
  default_rag_model: string;
  default_fallback_model: string;
  default_agent_reasoning_model: string;
}

export interface AgentConfig {
  max_iterations: number;
  min_pages_required: number;
  reasoning_model: string;
  enable_natural_language_agents: boolean;
  intent_classifier_model: string;
}

export interface ConfigResponse {
  ollama: OllamaConfig;
  ui: UIConfig;
  rag: RAGConfig;
  models: ModelsConfig;
  agent: AgentConfig;
}

export interface ConfigUpdateRequest {
  updates: Record<string, unknown>;
}

// Module/Model types
export interface ModuleInfo {
  name: string;
  description: string;
}

export interface ModulesResponse {
  modules: ModuleInfo[];
}

export interface ModelInfo {
  name: string;
  size: number;
  modified_at: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface PresetInfo {
  name: string;
  config: Record<string, unknown>;
}

export interface PresetsResponse {
  presets: PresetInfo[];
}

// PDF types
export interface PDFMetadata {
  pdf_id: string;
  filename: string;
  path: string;
  file_size: number;
  page_count: number;
}

export interface PDFListResponse {
  pdfs: PDFMetadata[];
  has_index: boolean;
}

export interface ReindexResponse {
  success: boolean;
  message: string;
  pdf_count: number;
}

// API Error
export interface ApiError {
  detail: string;
}
