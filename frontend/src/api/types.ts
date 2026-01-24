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
  modules?: string[] | null;
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
  thinking?: string | null;
  metrics?: RetrievalMetrics | null;
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

export interface RetrievalMetrics {
  score_distribution: {
    mean: number | null;
    median: number | null;
    min: number | null;
    max: number | null;
    std: number | null;
    q1: number | null;
    q3: number | null;
    iqr: number | null;
    range: number | null;
  };
  diversity: {
    unique_sources: number;
    source_types: number;
    source_entropy: number | null;
  };
  coverage: {
    total_context_chars: number;
    avg_chunk_length: number;
    total_chunks: number;
    estimated_tokens: number;
  };
  quality: {
    high_confidence_ratio: number;
    low_confidence_ratio: number;
  };
}

export interface ChatResponse {
  content: string;
  sources: SourceNode[];
  confidence_level: string;
  metrics?: RetrievalMetrics | null;
}

// WebSocket message types
export interface StreamToken {
  type: "token";
  content: string;
}

export interface StreamSources {
  type: "sources";
  data: SourceNode[];
  metrics?: RetrievalMetrics | null;
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

export interface StreamThinking {
  type: "thinking";
  content: string;
}

export interface StreamStatus {
  type: "status";
  status: "loading_models" | "retrieving" | "reranking" | "thinking" | "generating";
}

export interface StreamToolProgress {
  type: "tool_progress";
  tool: string;
  action: "calling" | "completed" | "failed";
  params: Record<string, unknown>;
}

export type AgentPhase = "searching" | "ranking" | "fetching" | "summarizing" | "complete";

export interface StreamAgentProgress {
  type: "agent_progress";
  agent: string; // "web_search", future agents
  phase: AgentPhase;
  message: string; // Human-readable status (no emoji)

  // Phase-specific data (all optional)
  search_query?: string;
  search_hits?: number;
  pages_target?: number;
  pages_fetched?: number;
  pages_failed?: number;
  current_page?: {
    url: string;
    title: string;
    status: string;
    error?: string | null;
  };
  model_name?: string;
}

export interface WebSearchSource {
  url: string;
  title: string;
  status: "success" | "failed" | "skipped";
  error?: string | null;
  snippet?: string | null;
}

export interface StreamWebSearchSources {
  type: "web_sources";
  sources: WebSearchSource[];
}

export type StreamMessage =
  | StreamToken
  | StreamSources
  | StreamDone
  | StreamError
  | StreamTitle
  | StreamThinking
  | StreamStatus
  | StreamToolProgress
  | StreamAgentProgress
  | StreamWebSearchSources;

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
  default_top_n: number;
  default_confidence_threshold: number;
  default_confidence_cutoff_hard: number;
}

export interface RAGConfig {
  default_device: string;
  default_balance_strategy: string;
  default_embedding_model: string;
  default_reranker: string;
  // Max conversation turns (1 turn = user query + assistant response)
  max_history_turns: number;
  memory_token_limit: number;
}

export interface ModelsConfig {
  default_rag_model: string;
  default_agent_reasoning_model: string;
}

export interface AgentConfig {
  max_iterations: number;
  min_pages_required: number;
  reasoning_model: string;
  enable_natural_language_agents: boolean;
  intent_classifier_model: string;
}

export interface HistoryCleaningConfig {
  enabled: boolean;
  remove_emojis: boolean;
  remove_filler_phrases: boolean;
  normalize_whitespace: boolean;
  collapse_newlines: boolean;
}

export interface WebSearchConfig {
  ddg_max_results: number;
  max_pages_to_fetch: number;
  rerank_title_threshold: number;
  rerank_content_threshold: number;
  max_source_context_pct: number;
  input_context_pct: number;
}

export interface ConfigResponse {
  ollama: OllamaConfig;
  ui: UIConfig;
  rag: RAGConfig;
  models: ModelsConfig;
  agent: AgentConfig;
  history_cleaning: HistoryCleaningConfig;
  web_search: WebSearchConfig;
}

export interface ConfigUpdateRequest {
  updates: Record<string, unknown>;
}

// Module/Model types
export interface ModuleInfo {
  name: string;
  display_name: string;
  doc_type: string;
  sort_order: number;
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

export interface EmbeddingModelInfo {
  model_id: string;
  model_name: string | null;
  index_count: number;
  modules: string[];
}

export interface EmbeddingModelsResponse {
  models: EmbeddingModelInfo[];
  current: string;
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

// Startup types
export interface AvailableEmbeddingModel {
  model_id: string;
  model_name: string | null;
  index_count: number;
  modules: string[];
}

export interface IndexesStatus {
  exists: boolean;
  has_content: boolean;
  available_models?: AvailableEmbeddingModel[] | null;
}

export interface ModelsStatus {
  required: string[];
  available: string[];
  missing: string[];
}

export interface EmbeddingMismatch {
  config_model: string;
  config_model_id: string;
  available_model_ids: string[];
  message: string;
}

export interface StartupStatusResponse {
  directories_ok: boolean;
  config_ok: boolean;
  indexes_ok: boolean;
  models_ok: boolean;
  indexes_status: IndexesStatus;
  models_status: ModelsStatus;
  embedding_mismatch?: EmbeddingMismatch | null;
  ready: boolean;
  warnings: string[];
}

export interface IndexDownloadRequest {
  repo_id?: string;
  filename?: string;
  embedding_model?: string;
}

export interface IndexDownloadResponse {
  status: string;
  message: string;
}

export interface ModelPullRequest {
  model_name: string;
}

export interface ModelPullResponse {
  status: string;
  message: string;
}

export interface ReinitializeIndexesResponse {
  status: string;
  message: string;
}

// Reranker types
export interface RerankerModelInfo {
  model: string;
  status: string;
}

export interface RerankerListResponse {
  models: RerankerModelInfo[];
  current: string;
}

export interface RerankerAddRequest {
  model: string;
}

export interface RerankerAddResponse {
  status: string;
  model?: string | null;
  error?: string | null;
}

export interface RerankerRemoveResponse {
  status: string;
  error?: string | null;
}

// API Error
export interface ApiError {
  detail: string;
}
